"""Vocoder-only research wrappers; upstream learned parameters stay immutable."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class DirectSharedPReLU(nn.Module):
    def __init__(self, source):
        super().__init__()
        if source.weight.numel() != 1:
            raise ValueError('Expected exactly one learned shared slope')
        self.weight = nn.Parameter(source.weight.detach().reshape(1).clone(), requires_grad=False)

    def forward(self, inputs):
        return F.prelu(inputs, self.weight)


class Rank4Head(nn.Module):
    """Preserve causal padding and keep both head projections/activation rank four."""
    def __init__(self, head):
        super().__init__()
        self.pad = copy.deepcopy(head.pad)
        first, second = head.layer1.net, head.layer2
        if first.groups != 1 or second.groups != 1 or first.padding != (0,) or second.padding != (0,):
            raise ValueError('Only the explicit-pad dense head is supported')
        self.weight1 = nn.Parameter(first.weight.detach().unsqueeze(2).clone(), requires_grad=False)
        self.bias1 = None if first.bias is None else nn.Parameter(first.bias.detach().clone(), requires_grad=False)
        self.weight2 = nn.Parameter(second.weight.detach().unsqueeze(2).clone(), requires_grad=False)
        self.bias2 = None if second.bias is None else nn.Parameter(second.bias.detach().clone(), requires_grad=False)
        self.stride1, self.dilation1 = (1, first.stride[0]), (1, first.dilation[0])
        self.stride2, self.dilation2 = (1, second.stride[0]), (1, second.dilation[0])
        self.act = DirectSharedPReLU(head.act)

    def forward(self, inputs):
        hidden = F.conv2d(self.pad(inputs).unsqueeze(2), self.weight1, self.bias1,
                          stride=self.stride1, dilation=self.dilation1)
        hidden = F.conv2d(self.act(hidden), self.weight2, self.bias2,
                          stride=self.stride2, dilation=self.dilation2)
        return hidden.squeeze(2).transpose(1, 2).reshape(hidden.shape[0], -1)


def fold_input_affine(conv, scale, shift):
    """Fold a channel affine before a dense, explicitly padded convolution.

    Valid for replicate padding, which commutes with the affine. Implicit zero
    padding does not commute with a nonzero shift and is deliberately rejected.
    """
    if not isinstance(conv, nn.Conv1d) or conv.groups != 1 or conv.padding != (0,):
        raise ValueError('Require dense Conv1d with explicit affine-compatible padding')
    if scale.numel() != conv.in_channels or shift.numel() != conv.in_channels:
        raise ValueError('Affine axis must match convolution input channels')
    dtype=conv.weight.dtype
    weight=conv.weight.detach().double()
    bias=torch.zeros(conv.out_channels,dtype=torch.float64,device=weight.device) if conv.bias is None else conv.bias.detach().double()
    new_weight=weight*scale.detach().double().reshape(1,-1,1)
    new_bias=bias+(weight*shift.detach().double().reshape(1,-1,1)).sum((1,2))
    conv.weight=nn.Parameter(new_weight.to(dtype),requires_grad=False)
    conv.bias=nn.Parameter(new_bias.to(dtype),requires_grad=False)


def fold_head_bn(model):
    decoder=model.decoder;norm=decoder.final_norm.norm;conv=decoder.head.layer1.net
    if decoder.head.pad.padding!=conv.dilation[0]*(conv.kernel_size[0]-1):
        raise ValueError('Expected exact causal replicate padding')
    scale=norm.weight.detach().double()/torch.sqrt(norm.running_var.detach().double()+1e-5)
    shift=norm.bias.detach().double()-scale*norm.running_mean.detach().double()
    fold_input_affine(conv,scale,shift)
    decoder.final_norm=nn.Identity()
    return model


class InputAffineFoldedVocoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder=model.decoder
        self.latent_dim=model.latent_dim;self.compress_factor=model.compress_factor
        self.latent_channels=model.latent_channels
        conv=self.decoder.embed.net
        if self.decoder.embed_pad.padding!=conv.dilation[0]*(conv.kernel_size[0]-1):
            raise ValueError('Expected exact causal replicate padding')
        scale=model.latent_std.detach().double()/model.normalizer.scale.detach().double()
        fold_input_affine(conv,scale,model.latent_mean)

    def forward(self, latent):
        if latent.ndim!=3 or latent.shape[1]!=self.latent_channels or latent.shape[2]<=0:
            raise ValueError('Invalid packed latent shape')
        batch,_,length=latent.shape
        unpacked=latent.reshape(batch,self.latent_dim,self.compress_factor,length).transpose(2,3).reshape(batch,self.latent_dim,length*self.compress_factor)
        return self.decoder(unpacked)


class GammaFoldedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block=block
        conv=block.pwconv2;scale=block.gamma.detach().double().reshape(-1)
        conv.weight=nn.Parameter((conv.weight.detach().double()*scale[:,None,None]).to(conv.weight.dtype),requires_grad=False)
        if conv.bias is not None:
            conv.bias=nn.Parameter((conv.bias.detach().double()*scale).to(conv.bias.dtype),requires_grad=False)

    def forward(self, inputs):
        block=self.block
        hidden=block.norm.norm(block.dwconv(block.pad(inputs)).transpose(1,2))
        hidden=block.act(F.linear(hidden,block.pwconv1.weight[...,0],block.pwconv1.bias))
        hidden=F.linear(hidden,block.pwconv2.weight[...,0],block.pwconv2.bias)
        return inputs+hidden.transpose(1,2)


class Rank4DecoderBlock(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.block=wrapper.block
        self.gamma_folded=isinstance(wrapper,GammaFoldedBlock)
        conv=self.block.dwconv.net
        if conv.padding!=(0,) or conv.stride!=(1,) or conv.groups!=conv.in_channels:
            raise ValueError('Expected causal depthwise convolution')
        self.weight=nn.Parameter(conv.weight.detach().unsqueeze(2).clone(),requires_grad=False)
        self.bias=conv.bias;self.groups=conv.groups;self.dilation=(1,conv.dilation[0])
        self.padding=self.block.pad.padding

    def forward(self, inputs):
        block=self.block
        prefix=inputs[:,:,:1,:].expand(-1,-1,self.padding,-1)
        padded=torch.cat((prefix,inputs),dim=2)
        hidden=F.conv2d(padded.permute(0,3,1,2),self.weight,self.bias,groups=self.groups,dilation=self.dilation).permute(0,2,3,1)
        hidden=block.norm.norm(hidden)
        hidden=block.act(F.linear(hidden,block.pwconv1.weight[...,0],block.pwconv1.bias))
        hidden=F.linear(hidden,block.pwconv2.weight[...,0],block.pwconv2.bias)
        if not self.gamma_folded: hidden=hidden*block.gamma.reshape(1,1,1,-1)
        return inputs+hidden


class Rank4DecoderBody(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.embed_pad=decoder.embed_pad;self.embed=decoder.embed
        self.convnext=nn.ModuleList([Rank4DecoderBlock(block) for block in decoder.convnext])
        self.final_norm=decoder.final_norm;self.head=decoder.head

    def forward(self, latent):
        hidden=self.embed(self.embed_pad(latent)).transpose(1,2).unsqueeze(1)
        for block in self.convnext: hidden=block(hidden)
        hidden=hidden.squeeze(1).transpose(1,2)
        return self.head(self.final_norm(hidden))


def transform(model, mode, selection=None):
    if mode == 'baseline':
        return model
    if mode == 'direct-prelu':
        model.decoder.head.act = DirectSharedPReLU(model.decoder.head.act)
    elif mode == 'rank4-head':
        model.decoder.head = Rank4Head(model.decoder.head)
    elif mode == 'fold-head-bn':
        model=fold_head_bn(model)
    elif mode == 'fold-input-affine':
        model=InputAffineFoldedVocoder(model)
    elif mode == 'fold-gamma':
        for index,wrapper in enumerate(model.decoder.convnext):
            model.decoder.convnext[index]=GammaFoldedBlock(wrapper.block)
    elif mode == 'rank4-body':
        model.decoder=Rank4DecoderBody(model.decoder)
    elif mode == 'combined':
        if not selection: raise ValueError('Combined transform selection must be frozen first')
        for name in selection: model=transform(model,name)
    else:
        raise ValueError(mode)
    return model

// A persistent native ARM64 host for ExecuTorch's QNN/HTP delegate.
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::make_tensor_ptr;
using executorch::runtime::EValue;
using executorch::runtime::Error;
using executorch::aten::ScalarType;
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void trace_event(json& timings, const std::string& name, Clock::time_point start) {
  if (!timings.contains("trace_events")) return;
  timings["trace_events"].push_back({{"name", name}, {"cat", "host"}, {"ph", "X"},
      {"pid", 1}, {"tid", 1}, {"ts", std::chrono::duration<double, std::micro>(start.time_since_epoch()).count()},
      {"dur", elapsed(start) * 1000}});
}

TensorPtr read_tensor(const json& spec) {
  const auto shape = spec.at("shape").get<std::vector<int32_t>>();
  size_t count = 1;
  for (auto size : shape) {
    if (size <= 0) throw std::runtime_error("Invalid input shape");
    count *= size;
  }
  const auto dtype = spec.at("dtype").get<std::string>();
  if (dtype != "float32" && dtype != "int64")
    throw std::runtime_error("Unsupported input dtype: " + dtype);
  const auto bytes = count * (dtype == "float32" ? sizeof(float) : sizeof(int64_t));
  auto data = std::make_shared<std::vector<uint8_t>>(bytes);
  std::ifstream stream(spec.at("path").get<std::string>(), std::ios::binary);
  if (!stream.read(reinterpret_cast<char*>(data->data()), bytes) || stream.peek() != EOF)
    throw std::runtime_error("Input size mismatch: " + spec.at("path").get<std::string>());
  return make_tensor_ptr(shape, data->data(),
      dtype == "float32" ? ScalarType::Float : ScalarType::Long,
      executorch::aten::Device(executorch::aten::DeviceType::CPU),
      executorch::aten::TensorShapeDynamism::STATIC,
      [data](void*) {});
}

void write_tensor(const TensorPtr& tensor, const std::filesystem::path& path) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(static_cast<const char*>(tensor->const_data_ptr()), tensor->nbytes());
  if (!stream) throw std::runtime_error("Failed to write " + path.string());
}

class Runner {
  std::filesystem::path models_;
  std::map<int, std::filesystem::path> buckets_;
  std::map<int, std::filesystem::path> vocoder_buckets_;
  std::map<std::string, std::unique_ptr<Module>> modules_;
  std::map<std::string, std::pair<int, int>> flow_settings_;

  std::string model_path(const std::string& name, int bucket) const {
    const auto directory = bucket ? (name == "vocoder" && vocoder_buckets_.count(bucket)
        ? vocoder_buckets_.at(bucket) : buckets_.at(bucket)) : models_;
    return (directory / (name + ".pte")).lexically_normal().string();
  }
 public:
  explicit Runner(std::filesystem::path models) : models_(std::move(models)) {
    const auto manifest = models_ / "bundle.json";
    if (std::filesystem::exists(manifest)) {
      const auto bundle = json::parse(std::ifstream(manifest));
      for (const auto& bucket : bundle.at("buckets")) {
        buckets_.emplace(bucket.at("latent_length").get<int>(),
                         models_ / bucket.at("models").get<std::string>());
        if (bucket.contains("vocoder"))
          vocoder_buckets_.emplace(bucket.at("latent_length").get<int>(),
                                  models_ / bucket.at("vocoder").get<std::string>());
      }
      models_ /= bundle.at("base").get<std::string>();
    }
  }

  Module& load(const std::string& name, json& timings, int bucket = 0) {
    if (name != "duration_predictor" && name != "text_encoder" &&
        name != "vector_estimator" && name != "vocoder")
      throw std::runtime_error("Unknown component: " + name);
    const auto path = model_path(name, bucket);
    if (!modules_.count(path)) {
      const auto start = Clock::now();
      if (name == "vector_estimator") {
        auto metadata_path = std::filesystem::path(path);
        metadata_path.replace_extension(".json");
        const auto metadata = json::parse(std::ifstream(metadata_path));
        const int unroll = metadata.value("flow_unroll", 1);
        const int schedule = metadata.value("flow_total_steps", 0);
        if ((unroll != 1 && unroll != 2 && unroll != 4 && unroll != 8) ||
            schedule < 0 || schedule > 100 || (unroll > 1 && schedule == 0) ||
            (schedule && schedule % unroll))
          throw std::runtime_error("Invalid compiled flow schedule");
        flow_settings_[path] = {unroll, schedule};
      }
      auto module = std::make_unique<Module>(path);
      const auto error = module->load_forward();
      if (error != Error::Ok)
        throw std::runtime_error("Loading " + name + " failed: " + std::to_string(int(error)));
      modules_[path] = std::move(module);
      timings[name + "_load_ms"] = elapsed(start);
      trace_event(timings, name + " load", start);
    }
    return *modules_.at(path);
  }

  TensorPtr run(const std::string& name, const std::vector<TensorPtr>& inputs, json& timings, int bucket = 0) {
    auto& module = load(name, timings, bucket);
    if (name == "vector_estimator") {
      const auto [unroll, schedule] = flow_settings_.at(model_path(name, bucket));
      if (schedule) {
        if (inputs.size() != 7 || inputs[5]->numel() != 1 || inputs[6]->numel() != 1 ||
            inputs[5]->scalar_type() != ScalarType::Float || inputs[6]->scalar_type() != ScalarType::Float)
          throw std::runtime_error("Grouped flow requires scalar float32 schedule inputs");
        const float step = inputs[5]->const_data_ptr<float>()[0];
        const float total = inputs[6]->const_data_ptr<float>()[0];
        if (!std::isfinite(step) || step < 0 || step + unroll > schedule ||
            std::fmod(step, float(unroll)) != 0 || total != schedule)
          throw std::runtime_error("Grouped flow requires an aligned step within the compiled schedule");
      }
    }
    std::vector<EValue> values;
    for (const auto& input : inputs) values.emplace_back(*input);
    const auto start = Clock::now();
    auto result = module.forward(values);
    const auto ms = elapsed(start);
    trace_event(timings, name + " forward (host wall time)", start);
    if (!result.ok())
      throw std::runtime_error("Executing " + name + " failed: " + std::to_string(int(result.error())));
    timings[name + "_ms"].push_back(ms);
    const auto& tensor = result->at(0).toTensor();
    if (tensor.scalar_type() != ScalarType::Float)
      throw std::runtime_error("Expected float32 model output");
    const auto copy_start = Clock::now();
    std::vector<float> data(tensor.const_data_ptr<float>(), tensor.const_data_ptr<float>() + tensor.numel());
    std::vector<int32_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
    auto output = make_tensor_ptr(std::move(sizes), std::move(data));
    trace_event(timings, name + " output copy (CPU)", copy_start);
    return output;
  }

  json request(const json& request) {
    json timings = json::object();
    if (request.value("trace", false)) timings["trace_events"] = json::array();
    const auto start = Clock::now();
    if (request.at("command") == "infer") {
      std::vector<TensorPtr> inputs;
      for (const auto& spec : request.at("inputs")) inputs.push_back(read_tensor(spec));
      TensorPtr output;
      const auto name = request.at("component").get<std::string>();
      const int count = request.value("repeat", 1);
      if (count < 1 || count > 1000) throw std::runtime_error("Invalid repeat count");
      for (int i = 0; i < count; ++i) output = run(name, inputs, timings);
      write_tensor(output, request.at("output").get<std::string>());
    } else if (request.at("command") == "synthesize") {
      // Keep all four modules loaded and the flow loop inside this process.
      for (const auto* name : {"duration_predictor", "text_encoder"})
        load(name, timings);
      if (buckets_.empty()) {
        load("vector_estimator", timings);
        load("vocoder", timings);
      }
      std::map<std::string, TensorPtr> inputs;
      for (auto it = request.at("inputs").begin(); it != request.at("inputs").end(); ++it)
        inputs[it.key()] = read_tensor(it.value());
      const auto out_dir = std::filesystem::path(request.at("output_dir").get<std::string>());
      std::filesystem::create_directories(out_dir);
      const auto inference_start = Clock::now();
      const auto duration = run("duration_predictor", {inputs.at("text_ids"), inputs.at("style_dp"), inputs.at("text_mask")}, timings);
      const float speed = request.value("speed", 1.05f);
      const float seconds = duration->const_data_ptr<float>()[0] / speed;
      if (!std::isfinite(seconds) || seconds <= 0 || speed <= 0)
        throw std::runtime_error("Invalid predicted duration or speed");
      const int sample_rate = request.at("sample_rate");
      const int chunk = request.at("samples_per_latent");
      const int wav_length = static_cast<int>(seconds * sample_rate);
      const int latent_length = (wav_length + chunk - 1) / chunk;
      const auto noise = inputs.at("noise");
      int capacity = noise->size(2);
      int bucket = 0;
      double selected_load_ms = 0;
      if (!buckets_.empty()) {
        const auto selected = buckets_.lower_bound(latent_length);
        if (selected == buckets_.end()) throw std::runtime_error("Speech exceeds largest compiled bucket");
        bucket = capacity = selected->first;
        if (capacity > noise->size(2)) throw std::runtime_error("Noise is shorter than selected bucket");
        const auto loading_start = Clock::now();
        load("vector_estimator", timings, bucket);
        load("vocoder", timings, bucket);
        selected_load_ms = elapsed(loading_start);
      }
      if (latent_length > capacity)
        throw std::runtime_error("Speech exceeds compiled latent capacity; shorten text or export a larger bucket");
      auto mask = make_tensor_ptr({1, 1, capacity}, std::vector<float>(capacity, 0));
      for (int t = 0; t < latent_length; ++t) mask->mutable_data_ptr<float>()[t] = 1;
      auto latent = make_tensor_ptr({1, static_cast<int32_t>(noise->size(1)), capacity},
                                    std::vector<float>(noise->size(1) * capacity, 0));
      for (int c = 0; c < noise->size(1); ++c)
        std::copy_n(noise->const_data_ptr<float>() + c * noise->size(2), latent_length,
                    latent->mutable_data_ptr<float>() + c * capacity);
      const int steps = request.value("steps", 5);
      if (steps < 1 || steps > 100) throw std::runtime_error("Invalid flow step count");
      const auto [unroll, schedule] = flow_settings_.at(model_path("vector_estimator", bucket));
      if ((schedule && steps != schedule) || steps % unroll)
        throw std::runtime_error("Flow step count does not match the compiled schedule: " + std::to_string(schedule));
      const auto embedding = run("text_encoder", {inputs.at("text_ids"), inputs.at("style_ttl"), inputs.at("text_mask")}, timings);
      auto current = make_tensor_ptr({1}, std::vector<float>{0});
      auto total = make_tensor_ptr({1}, std::vector<float>{float(steps)});
      for (int step = 0; step < steps; step += unroll) {
        current->mutable_data_ptr<float>()[0] = float(step);
        latent = run("vector_estimator", {latent, embedding, inputs.at("style_ttl"), mask, inputs.at("text_mask"), current, total}, timings, bucket);
      }
      const auto audio = run("vocoder", {latent}, timings, bucket);
      timings["inference_ms"] = elapsed(inference_start) - selected_load_ms;
      timings["selected_bucket_load_ms"] = selected_load_ms;
      timings["bucket_length"] = capacity;
      timings["flow_unroll"] = unroll;
      timings["flow_invocations"] = steps / unroll;
      write_tensor(duration, out_dir / "duration.raw");
      write_tensor(embedding, out_dir / "text_emb.raw");
      write_tensor(latent, out_dir / "latent.raw");
      write_tensor(audio, out_dir / "audio.raw");
      timings["duration_seconds"] = seconds;
      timings["wav_length"] = wav_length;
      timings["latent_length"] = latent_length;
      timings["rtf"] = timings["inference_ms"].get<double>() / (seconds * 1000);
    } else {
      throw std::runtime_error("Unknown command");
    }
    timings["request_ms"] = elapsed(start);
    return {{"ok", true}, {"timings", timings}};
  }
};

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: supertonic_qnn <model-directory>\n";
    return 2;
  }
  executorch::runtime::runtime_init();
  QnnExecuTorchBackendRegister(reinterpret_cast<void*>(&executorch::runtime::register_backend));
  Runner runner(argv[1]);
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      std::cout << runner.request(json::parse(line)).dump() << std::endl;
    } catch (const std::exception& error) {
      std::cout << json({{"ok", false}, {"error", error.what()}}).dump() << std::endl;
    }
  }
}

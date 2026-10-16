cmake_minimum_required(VERSION 3.24)
project(supertonic_qnn LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(executorch CONFIG REQUIRED)
add_executable(supertonic_qnn runner.cpp)
target_include_directories(supertonic_qnn PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../.deps"
  "${CMAKE_CURRENT_SOURCE_DIR}/../.deps/executorch/third-party/json/include")
target_link_libraries(supertonic_qnn PRIVATE
  executorch extension_module extension_tensor qnn_executorch_backend)
target_compile_options(supertonic_qnn PRIVATE /EHsc)
add_custom_command(TARGET supertonic_qnn POST_BUILD COMMAND ${CMAKE_COMMAND} -E
  copy_if_different $<TARGET_FILE:qnn_executorch_backend> $<TARGET_FILE_DIR:supertonic_qnn>)

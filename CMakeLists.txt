cmake_minimum_required(VERSION 3.18)
project(example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Torch REQUIRED PATHS "/home/agustin_nahuel/projects/XNNPack/libtorch")

include_directories(${TORCH_INCLUDE_DIRS})


# Set options for executorch build.
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON) # Build with Xnnpack backend

set(TORCH_INSTALL_PREFIX ${TORCH_INSTALL_PREFIX} CACHE PATH "Path to LibTorch installation")


# Include the executorch subdirectory.
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/executorch
  ${CMAKE_BINARY_DIR}/executorch
)

add_executable(example main.cpp)
target_link_libraries(
  example
  PRIVATE executorch
          extension_module_static # Provides the Module class
          extension_tensor # Provides the TensorPtr class
          optimized_native_cpu_ops_lib # Provides baseline cross-platform
                                       # kernels
          xnnpack_backend # Provides the XNNPACK CPU acceleration backend
)

target_compile_options(example PRIVATE "-D_GLIBCXX_USE_CXX11_ABI=0")
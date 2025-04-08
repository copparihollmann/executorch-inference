# executorch-inference
Simple repository as an example for executorch inference.

This repository demonstrates a linking error with ExecuTorch's Module class. The goal is to help reproduce and identify a solution for the undefined reference error when linking against ExecuTorch libraries.

## Error to Reproduce

```
undefined reference to `executorch::extension::Module::Module(...)'
undefined reference to `executorch::extension::Module::execute(...)'
```
Full error message:
```
...
[100%] Linking CXX executable example
/usr/bin/ld: CMakeFiles/example.dir/main.cpp.o: in function `main':
main.cpp:(.text+0xa0): undefined reference to `executorch::extension::Module::Module(std::string const&, executorch::extension::Module::LoadMode, std::unique_ptr<executorch::runtime::EventTracer, std::default_delete<executorch::runtime::EventTracer> >)'
/usr/bin/ld: CMakeFiles/example.dir/main.cpp.o: in function `executorch::extension::Module::forward(std::vector<executorch::runtime::EValue, std::allocator<executorch::runtime::EValue> > const&)':
main.cpp:(.text._ZN10executorch9extension6Module7forwardERKSt6vectorINS_7runtime6EValueESaIS4_EE[_ZN10executorch9extension6Module7forwardERKSt6vectorINS_7runtime6EValueESaIS4_EE]+0x62): undefined reference to `executorch::extension::Module::execute(std::string const&, std::vector<executorch::runtime::EValue, std::allocator<executorch::runtime::EValue> > const&)'
[100%] Built target executor_runner
collect2: error: ld returned 1 exit status
gmake[2]: *** [CMakeFiles/example.dir/build.make:118: example] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:573: CMakeFiles/example.dir/all] Error 2
gmake: *** [Makefile:136: all] Error 2
```

## Setup

### Env Setup 

The repository includes an `environment.yml` file for easy setup:

```
# Create conda environment from the file
conda env create -f environment.yml

# Activate environment
conda activate executorch
```

### Init submodules

```
# Initialize submodules
git submodule update --init --recursive
```

## Generate `.pte` files 
Please run the only python script available in this repository to generate 2 models that can be used for inference.

## Building the project

```
# Create and enter build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build . -j$(nproc)
```

## Attempted Solutions

So far, the following approaches have been tried to fix the linking error:

1. **Using static vs shared Module library:**
```cmake
# Tried both options
target_link_libraries(example PRIVATE
    ${TORCH_LIBRARIES}
    executorch
    extension_module_static  # Option 1
    # extension_module       # Option 2 (shared)
    extension_tensor
    optimized_native_cpu_ops_lib
    extension_data_loader
)
```
2. **Using whole-archive linker flags:**
```cmake
target_link_libraries(example PRIVATE
    ${TORCH_LIBRARIES}
    -Wl,--whole-archive
    executorch
    extension_module
    extension_tensor
    optimized_native_cpu_ops_lib
    extension_data_loader
    -Wl,--no-whole-archive
)
```

## Current state

I tried to build using executorch as third_party since this was also done this way in examples like: [llvm_manual](https://github.com/pytorch/executorch/tree/main/examples/llm_manual) from the official executorch repository.

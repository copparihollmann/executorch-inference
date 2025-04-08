#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <cassert>
#include <iostream>

using namespace ::executorch::extension;

int main() {
    // Create a Module.
    Module module("/models/executorch/densenet169_dynamic.pte");

    // Create input tensor from a static buffer.
    float dummy[1 * 3 * 244 * 244] = {0};  
    auto tensor = from_blob(dummy, {1, 3, 244, 244});



    // Perform an inference.
    const auto result = module.forward(tensor);

    // Check for success or failure.
    if (result.ok()) {
        // Retrieve the output data.
        const auto output = result->at(0).toTensor().const_data_ptr<float>();
    }

    return 0;
}
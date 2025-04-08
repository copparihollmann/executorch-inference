import torch
import torchvision
import os
from torch.export import ExportedProgram, export, Dim
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# Define input shapes for different models
INPUT_SHAPES = {
    "densenet161": (4, 3, 224, 224),
    "resnet50": (4, 3, 224, 224),
}

# Define model list
MODEL_LIST = ["densenet161", "resnet50"]

def import_models(model_name: str, sample_inputs:tuple, use_dynamic_batch=True):
    """Import the models.

    Args:
        model_name (str): Model Name
        sample_inputs (tuple): Input Sample to export model
        use_dynamic_batch (bool): Whether to use dynamic batch size

    Returns:
        Edge: EdgeProgramManager which later can be used to compile.
    """
    print(f"Exporting {model_name} with {'dynamic' if use_dynamic_batch else 'static'} batch...")
    
    # Create sample tensor
    sample_inputs = torch.randn(sample_inputs)
    sample_args = (sample_inputs, )

    # Set up dynamic shapes
    dynamic_shapes = None
    if use_dynamic_batch:
        batch_dim = Dim("batch", min=1, max=64)  # Add min/max constraints
        
        # Use parameter name "x" which is what most torchvision models use
        dynamic_shapes = {"x": {0: batch_dim}}  # First dimension of input "x" is dynamic

    # Create the model
    if model_name == "densenet161":
        m = torchvision.models.densenet161(weights=torchvision.models.DenseNet161_Weights.DEFAULT).eval()
    elif model_name == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).eval()

    # Export model with torch.export
    exported_program: ExportedProgram = export(m, sample_args, dynamic_shapes=dynamic_shapes)
    
    # Lower to ExecuTorch with XNNPACK partitioner
    edge = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()]
    )

    # Create output directory if it doesn't exist
    os.makedirs("models/executorch", exist_ok=True)
    
    # Create executable program and save to file
    exec_prog = edge.to_executorch()
    suffix = "_dynamic" if use_dynamic_batch else ""
    output_path = f"models/executorch/{model_name}{suffix}.pte"
    
    with open(output_path, "wb") as f:
        exec_prog.write_to_file(f)
    
    print(f"Model exported to {output_path}")
    return edge

def test_compile(model_name:str, use_dynamic_batch=True) -> None:
    """Testing the compile func of executorch."""
    print(f"Compiling {model_name}...")
    import_models(
        model_name=model_name, 
        sample_inputs=INPUT_SHAPES[model_name], 
        use_dynamic_batch=use_dynamic_batch
    )

if __name__ == "__main__":
    for model_name in MODEL_LIST:
        test_compile(model_name, use_dynamic_batch=True)
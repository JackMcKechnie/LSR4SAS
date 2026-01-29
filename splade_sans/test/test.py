import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("No GPU available. Please run this script on a machine with a CUDA-capable GPU.")

print(f"Using device: {device}")

# Create a large tensor on the GPU
tensor_size = (10000, 10000)  # Large tensor size to push GPU limits
large_tensor = torch.randn(tensor_size, device=device)

# Infinite loop to perform heavy operations
while True:
    # Perform multiplication and division on the large tensor
    result = large_tensor * 2.0
    result = result / 2.0

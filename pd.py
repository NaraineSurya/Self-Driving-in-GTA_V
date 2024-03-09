import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    print(f"GPU is available with {num_gpus} {'GPU' if num_gpus == 1 else 'GPUs'}.")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available, using CPU.")

print(torch.cuda.is_available())
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Architecture List: {torch.cuda.get_arch_list()}")
# This line will run a small operation on the GPU to confirm functionality
print(torch.randn(1).cuda())

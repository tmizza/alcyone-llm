import torch
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
print("✅ Cleared GPU cache.")
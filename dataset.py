from datasets import load_dataset

# ✅ Fix: Add trust_remote_code=True to allow execution
dataset = load_dataset("openwebtext", trust_remote_code=True)

print(dataset)

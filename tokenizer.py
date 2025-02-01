from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

# ✅ Step 1: Initialize BPE Tokenizer (Same as Before)
print("Step 1: Initializing tokenizer...")
tokenizer = Tokenizer(models.BPE())

# ✅ Step 2: Set Pre-Tokenizer (Whitespace splitting, same as before)
print("Step 2: Setting up pre-tokenizer...")
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# ✅ Step 3: Load Dataset for Tokenizer Training (Same as Before)
print("Step 3: Loading dataset...")
dataset = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)

# ✅ Step 4: Process First 100K Samples for Tokenizer Training (Same as Before)
sample_count = 100_000  # Using the same sample count that worked before
texts = []
for i, sample in enumerate(dataset):
    if "text" in sample and isinstance(sample["text"], str):
        texts.append(sample["text"])
    if i >= sample_count:
        break
print(f"✅ Loaded {len(texts)} samples.")

# ✅ Step 5: Train Tokenizer with 25K Vocab (Only Change: Vocab Size)
print("Step 5: Training tokenizer...")
trainer = trainers.BpeTrainer(vocab_size=25_000, min_frequency=2)  # Only vocab size changed
tokenizer.train_from_iterator(texts, trainer)

# ✅ Step 6: Save Tokenizer (Same as Before)
tokenizer.save("bpe_tokenizer_25k.json")
print("✅ Tokenizer successfully saved as `bpe_tokenizer_25k.json`")

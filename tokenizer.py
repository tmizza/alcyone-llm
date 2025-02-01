from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

# ✅ Step 1: Initialize BPE Tokenizer
print("Step 1: Initializing tokenizer...")
tokenizer = Tokenizer(models.BPE())

# ✅ Step 2: Set Pre-Tokenizer (Whitespace splitting)
print("Step 2: Setting up pre-tokenizer...")
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# ✅ Step 3: Load Dataset for Training Tokenizer
print("Step 3: Loading dataset...")
dataset = load_dataset("openwebtext", split="train", streaming=True)

# ✅ Step 4: Process First 100K Samples for Tokenizer Training
sample_count = 100_000  # Adjust if needed
texts = []
for i, sample in enumerate(dataset):
    if "text" in sample and isinstance(sample["text"], str):
        texts.append(sample["text"])
    if i >= sample_count:
        break
print(f"✅ Loaded {len(texts)} samples.")

# ✅ Step 5: Train New Tokenizer (25K Vocab)
print("Step 5: Training tokenizer...")
trainer = trainers.BpeTrainer(vocab_size=25_000, min_frequency=2)
tokenizer.train_from_iterator(texts, trainer)

# ✅ Step 6: Save Tokenizer
tokenizer.save("bpe_tokenizer_25k.json")
print("✅ Tokenizer successfully saved as `bpe_tokenizer_25k.json`")

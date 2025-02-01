from tokenizers import Tokenizer

# Load the tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

# Check vocabulary size
vocab_size = tokenizer.get_vocab_size()
print(f"✅ Tokenizer Vocabulary Size: {vocab_size}")

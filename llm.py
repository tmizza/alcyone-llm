import torch
from model import MiniGPT  # Import your custom Transformer model
from tokenizer import tokenizer  # Import your trained tokenizer

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 10_000  # Must match training config
model = MiniGPT(vocab_size).to(device)

# Load trained weights (Assuming they are saved as 'minigpt.pth')
model.load_state_dict(torch.load("minigpt.pth", map_location=device))
model.eval()

def generate_text(model, start_text, max_length=50):
    """Generate text using trained LLM with greedy decoding."""
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(start_text).ids
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_ids)[:, -1, :]  # Get logits for last token
            next_token = torch.argmax(output, dim=-1).unsqueeze(0)  # Greedy decoding
            input_ids = torch.cat((input_ids, next_token), dim=1)  # Append next token

    return tokenizer.decode(input_ids.squeeze().tolist())

# Example usage
start_text = "The dog"
generated_text = generate_text(model, start_text)
print("\nGenerated Text:")
print(generated_text)

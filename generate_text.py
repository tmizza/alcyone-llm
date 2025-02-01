import torch
import torch.nn as nn
from tokenizers import Tokenizer
from model import MiniGPT

# ✅ Load Tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

# ✅ Load Trained Model
VOCAB_SIZE = 20_000
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
MAX_GEN_LENGTH = 100  # ✅ Max number of tokens to generate

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MiniGPT(VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS).to(device)
model.load_state_dict(torch.load("minigpt_final.pth", map_location=device))
model.eval()  # ✅ Set model to evaluation mode

# ✅ Text Generation Function with Debugging
def generate_text(prompt, max_length=MAX_GEN_LENGTH, temperature=1.2, top_k=20):
    # Tokenize input
    input_ids = tokenizer.encode(prompt).ids
    print(f"DEBUG: Tokenized input: {input_ids}")  # ✅ Print tokenized sequence

    input_tensor = torch.tensor([input_ids], device=device)
    print(f"DEBUG: Initial input tensor shape: {input_tensor.shape}")  # ✅ Print shape

    # Generate text token by token
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_tensor)
            print(f"DEBUG: Model output shape: {logits.shape}")  # ✅ Print model output shape

            logits = logits[:, -1, :]  # Get last token's logits
            logits = logits / temperature  # ✅ Adjust for randomness

            # ✅ Apply Top-K Sampling
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                probs = torch.nn.functional.softmax(values, dim=-1)
                next_token = indices[0, torch.multinomial(probs[0], num_samples=1)]  # ✅ Extract a single token
            else:
                # ✅ Greedy Decoding (Pick highest probability token)
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)  # ✅ Ensure shape is [1]

        # ✅ Debug `next_token` shape
        print(f"DEBUG: Next token shape before concatenation: {next_token.shape}")

        # ✅ Ensure `next_token` is a 2D tensor before concatenation
        next_token = next_token.unsqueeze(0)  # ✅ Fix shape to [1, 1]
        input_tensor = torch.cat([input_tensor, next_token], dim=1)

        # ✅ Fix EOS token check
        if next_token.item() == tokenizer.token_to_id("</s>"):  # ✅ Ensure single token check
            break

    # Decode generated tokens
    generated_text = tokenizer.decode(input_tensor.squeeze().tolist())
    return generated_text

# ✅ User Input for Text Generation
while True:
    prompt = input("\nEnter prompt (or type 'exit' to quit): ")
    if prompt.lower() == "exit":
        break

    generated_output = generate_text(prompt)
    print("\n📝 Generated Text:\n", generated_output)

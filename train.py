import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MiniGPT
from datasets import load_dataset
from tokenizers import Tokenizer

# ✅ Load tokenizer and dataset
tokenizer = Tokenizer.from_file("bpe_tokenizer_25k.json")
dataset = load_dataset("openwebtext", split="train", streaming=True)

# ✅ Model Hyperparameters (Optimized for 100M Model)
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 5e-4
VOCAB_SIZE = 25_000

# ✅ Model Architecture
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 12

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ✅ FIX: Update to new GradScaler syntax
scaler = torch.amp.GradScaler("cuda")

# ✅ Training Loop with Mixed Precision
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    batch_idx = 0

    for batch in DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4):
        batch_idx += 1
        input_ids = batch.to(device)
        target_ids = input_ids.clone()

        optimizer.zero_grad()

        # ✅ Use mixed precision training
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            output = model(input_ids)
            loss = criterion(output.view(-1, VOCAB_SIZE), target_ids.view(-1))

        # ✅ Apply gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/batch_idx}")

    # ✅ Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"minigpt_100m_epoch{epoch+1}.pth")
        print(f"✅ Checkpoint saved: minigpt_100m_epoch{epoch+1}.pth")

# ✅ Save Final Model
torch.save(model.state_dict(), "minigpt_100m_final.pth")
print("✅ Training complete! Final model saved.")

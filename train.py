import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from model import MiniGPT
import os

if __name__ == "__main__":
    # ✅ Load the tokenizer (Matches Your 25K Vocab Model)
    tokenizer = Tokenizer.from_file("bpe_tokenizer_25k.json")

    # ✅ Download and Cache Dataset Locally (No More Streaming)
    dataset_cache_path = "./dataset_cache"
    print("📥 Downloading dataset (only happens once)...")
    dataset = load_dataset("openwebtext", split="train", cache_dir=dataset_cache_path)
    print("✅ Dataset downloaded and cached locally.")

    # ✅ Model Hyperparameters (Same as Before)
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LR = 5e-4
    VOCAB_SIZE = 25_000
    SEQ_LEN = 256  # ✅ Enforce Fixed Sequence Length

    # ✅ Model (100M Parameter Version)
    D_MODEL = 512
    N_HEADS = 8
    N_LAYERS = 12

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ✅ Check if there's a checkpoint to resume training
    checkpoint_path = "minigpt_100m_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("🆕 No checkpoint found, starting from scratch.")

    # ✅ Convert dataset into tokenized tensors
    def tokenize_function(example):
        text = example["text"].strip()
        tokens = tokenizer.encode(text).ids[:SEQ_LEN]  # Truncate if too long
        tokens += [0] * (SEQ_LEN - len(tokens))  # Pad if too short
        return {"input_ids": torch.tensor(tokens, dtype=torch.long)}

    dataset = dataset.map(tokenize_function, remove_columns=["text"])
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # ✅ Mixed Precision Training (Reduces VRAM Usage)
    scaler = torch.amp.GradScaler("cuda")

    # ✅ Training Loop (Auto-Saves Every 5 Epochs)
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        batch_idx = 0

        for batch in data_loader:
            batch_idx += 1

            input_ids = batch["input_ids"].to(device)
            target_ids = input_ids.clone()

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model(input_ids)
                loss = criterion(output.view(-1, VOCAB_SIZE), target_ids.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/batch_idx}")

        # ✅ Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")

    # ✅ Save Final Model
    final_model_path = "minigpt_100m_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Training complete! Final model saved at {final_model_path}")

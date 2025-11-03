import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from google.colab import drive

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# ==== Settings ====
SEQ_LEN = 100  # number of time steps per sample
D_MODEL = 128
NHEAD = 8
FF_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== Positional Encoding ====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# ==== Custom Dataset ====
class BehaviorDataset(Dataset):
    def __init__(self, base_path, seq_len=SEQ_LEN):
        self.samples = []
        self.labels = []

        label_map = {
            'saline_csv': 0,
            'fentanyl_csv': 1
        }

        for label_dir, label in label_map.items():
            path = os.path.join(base_path, label_dir)
            csv_files = glob.glob(os.path.join(path, "*.csv"))

            for file in csv_files:
                df = pd.read_csv(file)
                df = df.drop(df.columns[16:20], axis=1)  # remove cols 16–19
                data = df.values.astype(np.float32)
                # Normalize
                data = (data - data.mean()) / (data.std() + 1e-6)

                # Split into sequences
                for i in range(0, len(data) - seq_len, seq_len):
                    seq = data[i:i+seq_len]
                    if seq.shape[0] == seq_len:
                        self.samples.append(seq)
                        self.labels.append(label)

        self.samples = torch.tensor(self.samples)  # (N, seq_len, features)
        self.labels = torch.tensor(self.labels).float()  # Binary

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# ==== Transformer Classifier ====
class TransformerClassifier(nn.Module):
    def __init__(self, encoder, input_dim=18, d_model=D_MODEL):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)  # 🔁 Project input to match d_model
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Binary classification
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)           # ➜ (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)           # ➜ (seq_len, batch, d_model)
        x = self.encoder(x)              # ➜ (seq_len, batch, d_model)
        pooled = x.mean(dim=0)           # ➜ (batch, d_model)
        logits = self.classifier(pooled) # ➜ (batch, 1)
        return logits.squeeze(-1)        # ➜ (batch)


# ==== Load Pretrained Encoder ====
def load_pretrained_encoder(path="/content/drive/My Drive/transformer_encoder.pt"):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL, 
        nhead=NHEAD, 
        dim_feedforward=FF_DIM, 
        dropout=0.2
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
    encoder.load_state_dict(torch.load(path, map_location=DEVICE))
    return encoder


# ==== Train Function ====
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=EPOCHS):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

# ==== Pipeline ====
# Load dataset
dataset = BehaviorDataset(base_path="/content/drive/My Drive/data")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Load pretrained encoder & build model
encoder = load_pretrained_encoder().to(DEVICE)
for param in encoder.parameters():
    param.requires_grad = True  # Fine-tune entire encoder

model = TransformerClassifier(encoder).to(DEVICE)

# Positional encoding wrapper
pos_enc = PositionalEncoding(D_MODEL).to(DEVICE)

# Wrap data to add positional encoding
def add_positional_encoding(batch):
    return [(pos_enc(x.to(DEVICE)), y.to(DEVICE)) for x, y in batch]

# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss()

# Train
train_model(model, train_loader, val_loader, optimizer, loss_fn)

# Evalutaion 
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        x = pos_enc(x.to(DEVICE))
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        print(f"Predicted: {preds.tolist()}")
        print(f"True: {y.int().tolist()}")
        break

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizer import NTLevelTokenizer

# ======== Dummy Dataset für Beispielzwecke ========
class DNADataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # Tensor of shape (N, L)
        self.labels = labels        # Tensor of shape (N,)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'input_ids': self.sequences[idx],
            'labels': self.labels[idx]
        }

# ======== Transformer-Modell ========
class SimpleDNAClassifier(nn.Module):
    def __init__(self, vocab_size=4, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2, num_classes=2, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)              # (B, L, D)
        x = self.transformer(x)                    # (B, L, D)
        x = x.mean(dim=1)                          # Pooling (mean over sequence)
        logits = self.classifier(x)                # (B, num_classes)
        return logits

# ======== Trainingsfunktion ========
def train(model, dataloader, epochs=5, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")


# Dummy-Sequenzen (Batch size 4, Länge 50, Tokenized: 0=A, 1=C, 2=G, 3=T)
'''torch.manual_seed(42)
X = torch.randint(0, 4, (100, 50))  # 100 Sequenzen, je 50 Tokens
y = torch.randint(0, 2, (100,))     # Binary labels (0/1)
print(y)
dataset = DNADataset(X, y)
print(dataset)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Modell & Training
model = SimpleDNAClassifier()
train(model, dataloader)
'''

import hydra
from omegaconf import DictConfig
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW

# 1. Custom Dataset
class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, k=6, max_len=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.k = k
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # K-mer tokenization
        kmers = [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]
        encoded = self.tokenizer(
            " ".join(kmers),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. Load tokenizer and model (use pre-trained DNA-BERT)
def load_model(model_name='zhihan1996/DNA_bert_6', num_labels=2):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

# 3. Training loop
def train_model(model, dataloader, epochs=3, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits 
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

    
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.info(f"Running experiment: {cfg.run_name}")
    logger.info(f"Training config: {cfg.train}")
    logger.info(f"Model config: {cfg.model}")
    logger.info(f"Loss function: {cfg.lossFunction}")

    # Set up output directory (Hydra does this automatically)
    output_dir = os.getcwd()
    logger.info(f"Outputs will be saved to: {output_dir}")

    # Example data
    sequences = ["ACGTACGTACGT", "CGTACGTAGCTA", "TGCATGCACTGA"]
    labels = [0, 1, 0]  # Binary labels for classification

    tokenizer, model = load_model(model_name="zhihan1996/DNA_bert_6", num_labels=2)
    dataset = DNADataset(sequences, labels, tokenizer, k=6)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    train_model(model, dataloader, epochs=3)

if __name__ == "__main__":
    main()
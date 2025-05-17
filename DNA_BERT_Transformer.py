import hydra
from omegaconf import DictConfig
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from utils import load_config  # pointing to conf/download_conf.yml
from tokenizer import NTLevelTokenizer
from dataloader import create_genomic_dataloader
from torch.utils.tensorboard import SummaryWriter

# 1. Custom Dataset
class DNABERTBatchDataset(Dataset):
    def __init__(self, tokenized_batch, labels, id_to_base, tokenizer, k=6, max_len=512):
        self.tokenized_batch = tokenized_batch
        self.labels = labels
        self.id_to_base = id_to_base
        self.tokenizer = tokenizer
        self.k = k
        self.max_len = max_len

    def _detokenize(self, seq_tensor):
        return ''.join([self.id_to_base.get(token.item(), 'N') for token in seq_tensor])

    def _kmerize(self, seq):
        return [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]

    def __len__(self):
        return len(self.tokenized_batch)

    def __getitem__(self, idx):
        seq_tensor = self.tokenized_batch[idx]
        label = self.labels[idx]

        nt_seq = self._detokenize(seq_tensor)
        kmers = self._kmerize(nt_seq)

        encoded = self.tokenizer(
            ' '.join(kmers),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': label
        }


# 2. Load tokenizer and model (use pre-trained DNA-BERT)
def load_model(model_name='zhihan1996/DNA_bert_6', num_labels=2):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

# 3. Training loop
def train_model(model, dataloader, dna_tokenizer, id_to_base, epochs=3, lr=2e-5, k=6, log_dir='runs/dnabert_experiment'):
    # create writer for tensor board
    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        counter = 0
        total_loss = 0
        for batch in dataloader:
            counter += 1
            print("Round", counter)
            if counter == 10:
                break
            sample_seqs, ref_seqs, phenotypes, exon_masks, rep_masks, positions, chroms = batch
            # sample_seqs: [batch_size, chunk_size]
            # phenotypes: [batch_size, num_phenotypes]
            # masks and positions aligned to sequences
            # Use these tensors for model training or evaluation

            labels = exon_masks[:, 0].long() if exon_masks.ndim > 1 else exon_masks.long()
            wrapped_dataset = DNABERTBatchDataset(
            tokenized_batch=sample_seqs,
            labels=labels,
            id_to_base=id_to_base,
            tokenizer=dna_tokenizer,
            k=k)

            batch_dataloader = DataLoader(wrapped_dataset, batch_size=4, shuffle=False)

            for sub_batch in batch_dataloader:
                input_ids = sub_batch['input_ids'].to(device)
                attention_mask = sub_batch['attention_mask'].to(device)
                labels = sub_batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits 
                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
        avg_loss = total_loss/len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
        # close writer
        writer.close()

def tokens_to_sequence(token_tensor, id_to_base):
    """
    Converts a 1D tensor of token ids into a nucleotide string.

    Args:
        token_tensor: torch.Tensor of shape (sequence_length,)
        id_to_base: dict mapping int token -> str base (e.g. {0:'A',1:'C',...})

    Returns:
        str: nucleotide sequence
    """
    return ''.join(id_to_base.get(token.item(), 'N') for token in token_tensor)

def batch_tokens_to_sequences(batch_tensor, id_to_base):
    """
    Converts a batch of token ids to list of nucleotide strings.

    Args:
        batch_tensor: torch.Tensor of shape (batch_size, sequence_length)
        id_to_base: dict mapping int token -> str base

    Returns:
        List[str]: list of nucleotide sequences
    """
    sequences = []
    for seq_tensor in batch_tensor:
        seq_str = tokens_to_sequence(seq_tensor, id_to_base)
        sequences.append(seq_str)
    return sequences
    
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

    # Load project configuration
    gconfig = load_config('conf/download_config.yml')
    # Initialize tokenizer and dataloader
    tokenizer = NTLevelTokenizer()
    dataloader = create_genomic_dataloader(gconfig, tokenizer)
    counter = 0
    

    # Load DNA BERT tokenizer/model once
    dna_tokenizer, model = load_model("zhihan1996/DNA_bert_6", num_labels=2)

    vocab = tokenizer.vocab
    id_to_base = {v: k for k, v in vocab.items()}

    log_dir = os.path.join(os.getcwd(), "tensorboard_logs_DNA_BERT")

    train_model(model, dataloader, dna_tokenizer, id_to_base, epochs=3, log_dir=log_dir)


if __name__ == "__main__":
    main()
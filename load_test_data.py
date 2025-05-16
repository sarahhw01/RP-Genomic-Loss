from utils import load_config  # pointing to conf/download_conf.yml
from tokenizer import NTLevelTokenizer
from dataloader import create_genomic_dataloader

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

# Load project configuration
gconfig = load_config('conf/download_config.yml')
# Initialize tokenizer and dataloader
tokenizer = NTLevelTokenizer()
dataloader = create_genomic_dataloader(gconfig, tokenizer)
counter = 0
for batch in dataloader:
    counter += 1
    sample_seqs, ref_seqs, phenotypes, exon_masks, rep_masks, positions, chroms = batch
    vocab = tokenizer.vocab
    inverted_vocab = {v: k for k, v in vocab.items()}
    print(sample_seqs)
    de_tokenized_seqs = batch_tokens_to_sequences(sample_seqs, inverted_vocab)
    print(de_tokenized_seqs)
    # sample_seqs: [batch_size, chunk_size]
    # phenotypes: [batch_size, num_phenotypes]
    # masks and positions aligned to sequences
    # Use these tensors for model training or evaluation
    print("Round ", counter)
    if counter == 2:
        break
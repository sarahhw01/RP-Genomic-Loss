from utils import load_config  # pointing to conf/download_conf.yml
from tokenizer import NTLevelTokenizer
from dataloader import create_genomic_dataloader

# Load project configuration
gconfig = load_config('conf/download_config.yml')
# Initialize tokenizer and dataloader
tokenizer = NTLevelTokenizer()
dataloader = create_genomic_dataloader(gconfig, tokenizer)

for batch in dataloader:
    print(batch)
    sample_seqs, ref_seqs, phenotypes, exon_masks, rep_masks, positions, chroms = batch
    # sample_seqs: [batch_size, chunk_size]
    # phenotypes: [batch_size, num_phenotypes]
    # masks and positions aligned to sequences
    # Use these tensors for model training or evaluation
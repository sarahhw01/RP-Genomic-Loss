# RP-Genomic-Loss
Research Project with the Topic genomic loss


## Download Scripts

The `/download` folder contains utilities to fetch and prepare reference and annotation data for the HG38 genome.

### Download Reference Genome
```bash
python download/download_reference.py --config conf/download_conf.yml
```
- Downloads `hg38.fa.gz`, decompresses and indexes to `reference/hg38.fa` and `reference/hg38.fa.fai`.

### Download and Process Annotations
```bash
python download/download_annotations.py --config conf/download_conf.yml
```
- Downloads GENCODE GTF and RepeatMasker tracks.
- Generates BED files: `reference/hg38_exons.bed` and `reference/hg38_repeats.bed`.
- Sorts, compresses (`.gz`) and indexes (`.tbi`) each BED file.


## Testing the Dataloader

A convenience script is provided to validate the genomic dataloader.

```bash
python test_dataloader.py --config conf/download_conf.yml [--num-batches N] [--sample-limit M] [--token-display T]
```

- Loads a few batches and logs sequence, masks and phenotype information.


## Using the Genomic DataLoader

Import and create a PyTorch `DataLoader` for training or analysis:

```python
from utils import load_config  # pointing to conf/download_conf.yml
from tokenizer import NTLevelTokenizer
from dataloader import create_genomic_dataloader

# Load project configuration
gconfig = load_config('conf/download_conf.yml')
# Initialize tokenizer and dataloader
tokenizer = NTLevelTokenizer()
dataloader = create_genomic_dataloader(gconfig, tokenizer)

for batch in dataloader:
    sample_seqs, phenotypes, exon_masks, rep_masks, positions, chroms = batch
    # sample_seqs: [batch_size, chunk_size]
    # phenotypes: [batch_size, num_phenotypes]
    # masks and positions aligned to sequences
    # Use these tensors for model training or evaluation
```

Refer to `dataloader.py` for advanced options (shuffling, caching, custom collate function).
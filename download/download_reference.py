"""
Download HG38 Reference Genome

This script downloads the HG38 reference genome and indexes it for use with
the genomic data processing pipeline. Configuration is loaded from a YAML file.

Usage:
    python download_reference.py --config config.yml
"""

import os
import argparse
import logging
import sys
from typing import Optional
import pysam

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current))

# Import utility functions
from utils import (
    load_config,
    setup_logging,
    download_file,
    run_command,
    decompress_gzip,
    ensure_dir,
)

# URL for HG38 reference genome
HG38_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download HG38 reference genome.")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    return parser.parse_args()


def index_fasta(fasta_file: str) -> Optional[str]:
    """
    Index a FASTA file using samtools.

    Args:
        fasta_file: Path to FASTA file

    Returns:
        Path to index file or None if indexing failed
    """
    # Check if index already exists
    index_file = fasta_file + ".fai"
    if os.path.exists(index_file) and os.path.getsize(index_file) > 0:
        logging.info(f"Index file already exists: {index_file}. Skipping indexing.")
        return index_file

    logging.info(f"Indexing {fasta_file}")
    try:
        pysam.faidx(fasta_file)
    except Exception as e:
        logging.error(f"Failed to index {fasta_file}: {e}")
        return None

    return index_file


def download_hg38(output_dir: str) -> Optional[str]:
    """
    Download, decompress, and index HG38 reference genome.

    Args:
        output_dir: Directory to store reference genome

    Returns:
        Path to indexed FASTA file or None if any step failed
    """
    # Define output filenames
    gz_file = os.path.join(output_dir, "hg38.fa.gz")
    fasta_file = os.path.join(output_dir, "hg38.fa")

    # Step 1: Download HG38 reference genome
    logging.info("Downloading HG38 reference genome")
    downloaded_file = download_file(HG38_URL, gz_file)
    if not downloaded_file:
        logging.error(f"Failed to download HG38 reference genome from {HG38_URL}")
        return None

    # Step 2: Decompress the gzipped file
    decompressed_file = decompress_gzip(gz_file, fasta_file)
    if not decompressed_file:
        logging.error(f"Failed to decompress {gz_file}")
        return None

    # Step 3: Index the FASTA file
    index_file = index_fasta(fasta_file)
    if not index_file:
        logging.error(f"Failed to index {fasta_file}")
        return None

    logging.info(f"Successfully prepared HG38 reference genome: {fasta_file}")
    return fasta_file


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup directories
    base_dir = config.get("base_dir", "")
    if not base_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Check if reference download is enabled
    if not config["reference"]["download_reference"]:
        logging.info("Reference genome download is disabled in config. Exiting.")
        sys.exit(0)

    # Setup reference directory using the path from config
    reference_dir = os.path.join(base_dir, config["paths"]["reference_dir"])
    reference_dir = ensure_dir(reference_dir)

    # Setup output directory for logs
    output_dir = os.path.join(base_dir, config["paths"]["output_dir"])
    output_dir = ensure_dir(output_dir)

    # Setup logging
    log_config = config["logging"]
    log_level = log_config["level"]
    log_file = "download_reference.log"
    if log_file:
        log_file = os.path.join(output_dir, log_file)
    setup_logging(log_level, log_file)

    # Download and prepare reference genome
    reference_path = download_hg38(reference_dir)
    if reference_path:
        logging.info("\n" + "=" * 80)
        logging.info("REFERENCE GENOME DOWNLOAD COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Reference genome path: {reference_path}")
        logging.info(f"Index path: {reference_path + '.fai'}")
        logging.info("=" * 80)
    else:
        logging.error("Failed to download and prepare reference genome")


if __name__ == "__main__":
    main()

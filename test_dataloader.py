"""
Simple test script for the genomic dataloader.
Tests basic functionality by loading a small batch of data.
"""

import os
import sys
import yaml
import logging
import argparse
import traceback

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current))

# Import the dataloader
from dataloader import create_genomic_dataloader
from tokenizer import NTLevelTokenizer

from utils import setup_logging, load_config, ensure_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test genomic dataloader functionality")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to test")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=2,
        help="Maximum number of samples to display per batch",
    )
    parser.add_argument(
        "--token-display",
        type=int,
        default=100,
        help="Number of tokens to display for each sequence",
    )
    return parser.parse_args()


def test_dataloader(config, num_batches=10, sample_limit=10, token_display=100):
    """
    Test the genomic dataloader with the given configuration.

    Args:
        config (dict): Path to the configuration YAML file
        num_batches (int): Number of batches to test
        sample_limit (int): Maximum number of samples to display per batch
        token_display (int): Number of tokens to display for each sequence

    Returns:
        bool: True if test completes successfully, False otherwise
    """
    # Create tokenizer
    try:
        tokenizer = NTLevelTokenizer()
        logging.info(f"Created tokenizer with vocabulary size: {len(tokenizer)}")
    except Exception as e:
        logging.error(f"Failed to create tokenizer: {e}")
        return False

    # Create dataloader
    logging.info("Creating dataloader...")
    try:
        dataloader = create_genomic_dataloader(config, tokenizer)
        logging.info("Dataloader created successfully")
    except Exception as e:
        logging.error(f"Failed to create dataloader: {e}")
        traceback.print_exc()
        return False

    # Test by loading batches of data
    logging.info(f"Attempting to load {num_batches} batches of data...")
    try:
        dataloader_iter = iter(dataloader)
        for batch_idx in range(num_batches):
            # Get the next batch
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                logging.warning(f"Dataloader exhausted after {batch_idx} batches")
                break

            # Unpack the batch
            (
                sample_seqs,
                phenotypes,
                exon_mask,
                repetitive_mask,
                positions,
                chrom,
            ) = batch

            # Print batch statistics
            batch_size = sample_seqs.shape[0]
            logging.info(f"Batch {batch_idx+1}: Successfully loaded {batch_size} samples")
            logging.info(f"  Sample sequence shape:    {sample_seqs.shape}")
            logging.info(f"  Phenotype vector shape:   {phenotypes.shape}")
            # Display samples from the batch
            if batch_size > 0:
                display_samples(batch, tokenizer, min(sample_limit, batch_size), token_display)

        logging.info("Dataloader test completed successfully")
        return True

    except Exception as e:
        logging.error(f"Error testing dataloader: {e}")
        traceback.print_exc()
        return False


def display_samples(batch, tokenizer, sample_limit, token_display):
    """
    Display information about samples in a batch.

    Args:
        batch (tuple): The batch data containing sequences and metadata
        tokenizer: The tokenizer for converting token IDs to sequences
        sample_limit (int): Maximum number of samples to display
        token_display (int): Number of tokens to display for each sequence
    """
    (
        sample_seqs,
        ref_seqs,
        lookahead_seqs,
        phenotypes,
        exon_mask,
        repetitive_mask,
        positions,
        chrom,
    ) = batch
    batch_size = sample_seqs.shape[0]

    for i in range(min(sample_limit, batch_size)):
        logging.info(f"Sample {i+1}:")

        # Access region information
        logging.info(f"  Region: Chromosome {chrom}, Position {positions[0][0]}-{positions[-1][0]}")

        # Show a snippet of the sequences
        token_sample_size = min(token_display, sample_seqs.shape[1])
        sample_seq_sample = sample_seqs[i, :token_sample_size].tolist()
        ref_seq_sample = ref_seqs[i, :token_sample_size].tolist()
        lookahead_seqs_sample = lookahead_seqs[i, :token_sample_size].tolist()

        # Create match mask by comparing sample and reference sequences
        match_mask = [p == r for p, r in zip(sample_seq_sample, ref_seq_sample)]
        match_binary = "".join(["1" if x else "0" for x in match_mask])

        # Get other masks
        exon_binary = "".join(
            ["1" if x else "0" for x in exon_mask[i].bool().tolist()][:token_sample_size]
        )
        repetitive_binary = "".join(
            ["1" if x else "0" for x in repetitive_mask[i].bool().tolist()][:token_sample_size]
        )

        # Get sample and reference sequences
        sample_str = tokenizer.detokenize(sample_seq_sample)
        ref_str = tokenizer.detokenize(ref_seq_sample)
        lookahead_str = tokenizer.detokenize(lookahead_seqs_sample)
        # Display all information aligned to positions
        logging.info(f"  Alignment information (first {token_sample_size} positions):")
        logging.info(f"  Sample sequence:     {sample_str}")
        logging.info(f"  Reference sequence:  {ref_str}")
        logging.info(f"  Match mask:          {match_binary}")
        logging.info(f"  Exon mask:           {exon_binary}")
        logging.info(f"  Repetitive mask:     {repetitive_binary}")
        logging.info(f"  Lookahead sequence:  {lookahead_str}")

        # Calculate match statistics
        match_count = match_mask.count(True)
        match_percentage = (match_count / len(match_mask)) * 100 if match_mask else 0
        logging.info(
            f"  Match statistics: {match_count}/{len(match_mask)} positions match ({match_percentage:.2f}%)"
        )

        # Display phenotype information
        logging.info(f"  Phenotype vector: {phenotypes[i]}")


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    # Setup base directory

    config = load_config(args.config)

    # Setup directories
    base_dir = config.get("base_dir", "")
    if not base_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    output_dir = ensure_dir(os.path.join(base_dir, config["paths"]["output_dir"]))

    # Setup logging
    log_config = config["logging"]
    log_level = log_config.get("level", "INFO")
    log_file = "test_dataloader.log"
    if log_file:
        log_file = os.path.join(output_dir, log_file)
    setup_logging(log_level, log_file)

    # Run the test
    success = test_dataloader(config, args.num_batches, args.sample_limit, args.token_display)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

"""
Precompute repetitive regions and download intron/exon annotations for HG38 genome.

This script downloads gene annotations (GTF) and repetitive region tracks (RepeatMasker)
for the HG38 reference genome and processes them into a format that can be used by the
genomic dataloader for region-aware loss weighting.

Usage:
    python precompute_genomic_features.py --config config.yml
"""

import os
import sys
import argparse
import logging
import gzip
import pandas as pd
import numpy as np
import subprocess
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import pysam

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current))

# Import utility functions
from utils import load_config, setup_logging, ensure_dir, download_file

# URLs for downloading annotation data (HG38 specific)
GENCODE_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz"
REPEATMASKER_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Precompute genomic features for HG38.")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    return parser.parse_args()


def download_annotations(config: Dict) -> Dict[str, str]:
    """
    Download gene annotations and repeat masking data for HG38.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with paths to downloaded files
    """
    # Extract configuration
    base_dir = config.get("base_dir", "")
    if not base_dir:
        base_dir = os.path.abspath(os.path.dirname(__file__))

    # Setup directories
    annotation_dir = ensure_dir(os.path.join(base_dir, config["paths"]["reference_dir"]))

    # Define output file paths
    file_paths = {
        "gtf": os.path.join(annotation_dir, "hg38_annotations.gtf.gz"),
        "repeatmasker": os.path.join(annotation_dir, "hg38_repeatmasker.txt.gz"),
        "processed_exons": os.path.join(annotation_dir, "hg38_exons.bed"),
        "processed_repeats": os.path.join(annotation_dir, "hg38_repeats.bed"),
    }

    # Download gene annotations (GTF)
    if not os.path.exists(file_paths["gtf"]) or os.path.getsize(file_paths["gtf"]) == 0:
        logging.info("Downloading gene annotations for HG38")
        download_file(GENCODE_URL, file_paths["gtf"])
    else:
        logging.info(f"Gene annotation file already exists: {file_paths['gtf']}")

    # Download RepeatMasker data
    if (
        not os.path.exists(file_paths["repeatmasker"])
        or os.path.getsize(file_paths["repeatmasker"]) == 0
    ):
        logging.info("Downloading RepeatMasker data for HG38")
        download_file(REPEATMASKER_URL, file_paths["repeatmasker"])
    else:
        logging.info(f"RepeatMasker file already exists: {file_paths['repeatmasker']}")

    return file_paths


def process_gtf_for_exons(gtf_file: str, output_bed: str):
    """
    Process GTF file to extract exon regions.

    Args:
        gtf_file: Path to GTF file
        output_bed: Path to output BED file with exon regions
    """
    logging.info(f"Processing GTF file to extract exon regions")

    # Check if output already exists
    if os.path.exists(output_bed) and os.path.getsize(output_bed) > 0:
        logging.info(f"Exon BED file already exists: {output_bed}")
        return

    # Open output file
    with open(output_bed, "w") as out_f:
        # Write header
        out_f.write("# BED file with exon regions\n")
        out_f.write("# chrom\tstart\tend\tgene_id\texon_id\tstrand\n")

        # Process GTF file line by line
        with gzip.open(gtf_file, "rt") as f:
            for line in tqdm(f, desc="Processing GTF"):
                if line.startswith("#"):
                    continue

                fields = line.strip().split("\t")

                # Check if this is an exon entry
                if len(fields) >= 8 and fields[2] == "exon":
                    chrom = fields[0]
                    start = int(fields[3]) - 1  # Convert to 0-based
                    end = int(fields[4])
                    strand = fields[6]

                    # Extract gene_id and exon_id from attributes
                    attr_dict = {}
                    for attr in fields[8].split(";"):
                        attr = attr.strip()
                        if not attr:
                            continue

                        try:
                            key, value = attr.split(" ", 1)
                            attr_dict[key] = value.strip('"')
                        except ValueError:
                            continue

                    gene_id = attr_dict.get("gene_id", "unknown")
                    exon_id = attr_dict.get("exon_id", attr_dict.get("exon_number", "unknown"))

                    # Write to BED file
                    out_f.write(f"{chrom}\t{start}\t{end}\t{gene_id}\t{exon_id}\t{strand}\n")

    logging.info(f"Completed processing GTF. Exon regions saved to: {output_bed}")


def process_repeatmasker(repeatmasker_file: str, output_bed: str):
    """
    Process RepeatMasker file to extract repetitive regions.

    Args:
        repeatmasker_file: Path to RepeatMasker file
        output_bed: Path to output BED file with repetitive regions
    """
    logging.info(f"Processing RepeatMasker file to extract repetitive regions")

    # Check if output already exists
    if os.path.exists(output_bed) and os.path.getsize(output_bed) > 0:
        logging.info(f"Repeat regions BED file already exists: {output_bed}")
        return

    # Open output file
    with open(output_bed, "w") as out_f:
        # Write header
        out_f.write("# BED file with repetitive regions\n")
        out_f.write("# chrom\tstart\tend\trepeat_class\trepeat_family\n")

        # Process RepeatMasker file
        with gzip.open(repeatmasker_file, "rt") as f:
            # Skip header line if it exists
            first_line = f.readline()
            if not first_line.startswith("#"):
                # If there's no header, reset file pointer
                f.seek(0)

            # RepeatMasker from UCSC has these columns
            # bin, swScore, milliDiv, milliDel, milliIns, genoName, genoStart, genoEnd, ...
            for line in tqdm(f, desc="Processing RepeatMasker"):
                if line.startswith("#"):
                    continue

                fields = line.strip().split("\t")

                # UCSC RepeatMasker format (rmsk.txt.gz)
                # fields[5] = chromosome
                # fields[6] = start (0-based)
                # fields[7] = end
                # fields[10] = repeat class
                # fields[11] = repeat family

                if len(fields) >= 12:
                    try:
                        chrom = fields[5]
                        start = int(fields[6])
                        end = int(fields[7])
                        repeat_class = fields[10]
                        repeat_family = fields[11]

                        # Write to BED file
                        out_f.write(f"{chrom}\t{start}\t{end}\t{repeat_class}\t{repeat_family}\n")
                    except (ValueError, IndexError) as e:
                        logging.warning(f"Error processing line: {line.strip()}, Error: {e}")

    logging.info(f"Completed processing RepeatMasker. Repetitive regions saved to: {output_bed}")


def index_bed_files(bed_files: List[str]):
    """
    Index BED files using pysam's tabix functionality for faster retrieval.

    Args:
        bed_files: List of BED files to index
    """
    for bed_file in bed_files:
        sorted_bed = bed_file + ".sorted"
        bgzipped_bed = sorted_bed + ".gz"

        # Skip if already indexed
        if os.path.exists(bgzipped_bed + ".tbi"):
            logging.info(f"BED file already indexed: {bgzipped_bed}")
            continue

        logging.info(f"Indexing BED file: {bed_file}")

        try:
            # Sort BED file using pandas
            logging.info(f"Sorting BED file: {bed_file}")

            # Skip header lines that start with #
            header_lines = []
            with open(bed_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        header_lines.append(line)
                    else:
                        break

            # Read and sort the BED file
            bed_df = pd.read_csv(
                bed_file,
                sep="\t",
                comment="#",
                header=None,
                names=["chrom", "start", "end", "name", "score", "strand"],
            )

            # Sort by chromosome and then by start position
            bed_df = bed_df.sort_values(["chrom", "start"])

            # Write sorted data to file, first the headers, then the sorted data
            with open(sorted_bed, "w") as f:
                # Write original header lines
                for line in header_lines:
                    f.write(line)

                # Write sorted data without header
                bed_df.to_csv(f, sep="\t", index=False, header=False)

            logging.info(f"Successfully sorted BED file to: {sorted_bed}")

            # Compress with bgzip using pysam
            logging.info(f"Compressing with bgzip: {sorted_bed}")
            pysam.tabix_compress(sorted_bed, bgzipped_bed, force=True)

            # Index with tabix using pysam
            logging.info(f"Indexing with tabix: {bgzipped_bed}")
            pysam.tabix_index(bgzipped_bed, preset="bed", force=True)

            # Remove temporary files
            os.remove(sorted_bed)

            logging.info(f"Successfully indexed BED file: {bgzipped_bed}")

        except Exception as e:
            logging.error(f"Error indexing BED file {bed_file}: {str(e)}")
            # Clean up any partial files if they exist
            for temp_file in [sorted_bed, bgzipped_bed]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup base directory
    base_dir = config.get("base_dir", "")
    if not base_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Check if annotation download is enabled
    if not config["annotations"]["download_annotations"]:
        logging.info("Annotations download is disabled in config. Exiting.")
        sys.exit(0)

    # Setup output directory
    output_dir = ensure_dir(os.path.join(base_dir, config["paths"]["output_dir"]))

    # Setup logging
    log_config = config["logging"]
    log_level = log_config["level"]
    log_file = os.path.join(output_dir, "annotations.log")
    setup_logging(log_level, log_file)

    logging.info("Starting precomputation of genomic features for HG38")

    # Download annotation files
    file_paths = download_annotations(config)

    # Process GTF file to extract exon regions
    process_gtf_for_exons(file_paths["gtf"], file_paths["processed_exons"])

    # Process RepeatMasker file to extract repetitive regions
    process_repeatmasker(file_paths["repeatmasker"], file_paths["processed_repeats"])

    # Index BED files for faster retrieval
    index_bed_files([file_paths["processed_exons"], file_paths["processed_repeats"]])

    logging.info("Successfully precomputed genomic features for HG38")
    logging.info(f"Exon regions: {file_paths['processed_exons']}.sorted.gz")
    logging.info(f"Repetitive regions: {file_paths['processed_repeats']}.sorted.gz")

    print("\n" + "=" * 80)
    print(" " * 20 + "GENOMIC FEATURES PRECOMPUTATION COMPLETE")
    print("=" * 80)
    print(f"\nExon regions: {file_paths['processed_exons']}.sorted.gz")
    print(f"Repetitive regions: {file_paths['processed_repeats']}.sorted.gz")


if __name__ == "__main__":
    main()

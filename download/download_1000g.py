"""
Download VCF files from the 1000 Genomes Project.

This script downloads VCF files from the 1000 Genomes Project using the IGSR
(International Genome Sample Resource) API and AWS resources where the data
is currently hosted. Configuration is loaded from a YAML file.

Usage:
    python download_1000g.py --config config.yml
"""

import os
import sys
import argparse
import pandas as pd
import concurrent.futures
import logging
from typing import List, Dict, Tuple, Optional, Set
import shutil
import pysam
import heapq
import gc
from itertools import islice

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current))

# Import utility functions
from utils import (
    load_config,
    setup_logging,
    download_file,
    ensure_dir,
    all_chromosomes,
)

# Base URL for 1000 Genomes GRCh38 data (NYGC Realignment Release)
API_BASE = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/"

# URL for sample information (Using the standard Phase 3 panel)
# The core samples are consistent.
SAMPLE_INFO_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and preprocess 1000 Genomes Project data."
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    return parser.parse_args()


def get_vcf_locations(chromosomes) -> Tuple[List[str], str]:
    """
    Get GRCh38 VCF file URLs and Panel URL from the 1000genomes.ebi.ac.uk endpoint.

    Returns:
        chromosomes: List of chromosomes
        Tuple[List[str], str]: List of VCF file URLs and panel URL
    """
    vcf_urls = {}
    for chrom in chromosomes:
        vcf_url = f"{API_BASE}1kGP_high_coverage_Illumina.chr{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
        vcf_urls[chrom] = vcf_url
    if not vcf_urls:
        raise ValueError("No VCF files found for the selected chromosomes")

    return vcf_urls


def download_chr_vcf(
    vcf_url: str, chrom: str, output_dir: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Download a chromosome VCF file and its index.

    Args:
        vcf_url: URL of the VCF file
        output_dir: Directory to save the downloaded files

    Returns:
        Tuple of paths to downloaded VCF and index files
    """
    index_url = f"{vcf_url}.tbi"

    # Determine filenames based on chromosome
    vcf_filename = f"chr{chrom}.vcf.gz"
    index_filename = f"chr{chrom}.vcf.gz.tbi"

    # Define output paths
    vcf_path = os.path.join(output_dir, vcf_filename)
    index_path = os.path.join(output_dir, index_filename)

    # Download VCF file
    logging.info(f"Downloading chromosome {chrom} VCF file")
    logging.info(f"URL: {vcf_url}")
    vcf_result = download_file(vcf_url, vcf_path)

    if not vcf_result:
        logging.error(f"Failed to download VCF file for chromosome {chrom}")
        return None, None

    # Download index file
    logging.info(f"Downloading chromosome {chrom} index file")
    logging.info(f"URL: {index_url}")
    index_result = download_file(index_url, index_path)

    if not index_result:
        logging.error(f"Failed to download index file for chromosome {chrom}")
        return vcf_result, None

    return vcf_result, index_result, chrom


def create_phenotype_panel(
    panel_url: str, output_file: str, num_samples: int, population: Optional[str] = None
) -> Tuple[str, List[str]]:
    """
    Download panel file and create phenotype CSV.

    Args:
        panel_url: URL to the panel file
        output_file: Path to save the phenotype CSV
        num_samples: Number of samples to include
        population: Population filter (e.g., EUR, AFR)

    Returns:
        Tuple of output file path and list of sample IDs
    """
    columns_to_use = ["sample", "pop", "super_pop", "gender"]

    # Download panel file to a temporary location
    temp_panel_file = output_file + ".tmp"
    panel_result = download_file(panel_url, temp_panel_file)

    if not panel_result:
        logging.error(f"Failed to download panel file from {panel_url}")
        sys.exit(1)

    # Read the panel file
    try:
        panel_df = pd.read_csv(temp_panel_file, sep="\t")
        # Clean up temporary file
        os.remove(temp_panel_file)
    except Exception as e:
        logging.error(f"Failed to parse panel file: {e}")
        sys.exit(1)

    # Ensure all required columns exist
    for col in columns_to_use:
        if col not in panel_df.columns:
            logging.error(f"Required column '{col}' not found in panel file")
            sys.exit(1)

    panel_df = panel_df[columns_to_use].copy()

    # Filter by population if specified
    if population:
        original_count = len(panel_df)
        panel_df = panel_df[panel_df["super_pop"] == population]
        filtered_count = len(panel_df)

        if filtered_count == 0:
            logging.error(f"No samples found for population '{population}'")
            sys.exit(1)

        logging.info(
            f"Filtered {original_count} samples to {filtered_count} samples with population '{population}'"
        )

    # Ensure we have enough samples after filtering
    if len(panel_df) < num_samples:
        logging.warning(
            f"Requested {num_samples} samples but only {len(panel_df)} are available. Using all available samples."
        )
        num_samples = len(panel_df)

    # Take only the first num_samples
    panel_df = panel_df.head(num_samples)

    # Convert gender to numeric as some sources are inconsistent
    gender_map = {
        "male": 1,
        "Male": 1,
        "m": 1,
        "M": 1,
        "1": 1,
        "female": 2,
        "Female": 2,
        "f": 2,
        "F": 2,
        "2": 2,
    }

    panel_df["gender_numeric"] = panel_df["gender"].map(gender_map)

    # Ensure all gender values were mapped properly
    if panel_df["gender_numeric"].isnull().any():
        unmapped_genders = panel_df.loc[panel_df["gender_numeric"].isnull(), "gender"].unique()
        logging.error(f"Unrecognized gender values: {unmapped_genders}")
        sys.exit(1)

    # Create numeric mapping for population (pop)
    pop_values = sorted(panel_df["pop"].unique())
    pop_map = {pop: idx + 1 for idx, pop in enumerate(pop_values)}
    panel_df["pop_numeric"] = panel_df["pop"].map(pop_map)

    # Log population mapping for reference
    pop_mapping_str = ", ".join([f"{pop}={idx}" for pop, idx in pop_map.items()])
    logging.info(f"Population mapping: {pop_mapping_str}")

    # Create numeric mapping for super population (super_pop)
    superpop_values = sorted(panel_df["super_pop"].unique())
    superpop_map = {superpop: idx + 1 for idx, superpop in enumerate(superpop_values)}
    panel_df["super_pop_numeric"] = panel_df["super_pop"].map(superpop_map)

    # Log super population mapping for reference
    superpop_mapping_str = ", ".join([f"{pop}={idx}" for pop, idx in superpop_map.items()])
    logging.info(f"Super population mapping: {superpop_mapping_str}")

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    panel_df.to_csv(output_file, index=False)
    logging.info(f"Created phenotype CSV file: {output_file} with {len(panel_df)} samples")

    return output_file, panel_df["sample"].tolist()


def extract_sample_vcf(
    vcf_path: str,
    sample_id: str,
    chrom: str,
    output_dir: str,
    file_type: str = "vcf",
    write_mode: str = "w",
) -> Optional[str]:
    """
    Extract a single sample's data from a VCF file using pysam optimized for high memory environments.

    Args:
        vcf_path: Path to the VCF file
        sample_id: Sample ID to extract
        chrom: Chromosome identifier
        output_dir: Directory to save the extracted VCF
        file_type: Output file type (vcf or bcf)
        write_mode

    Returns:
        Tuple containing sample_id, chrom, and path to the extracted VCF file, or None if extraction failed
    """
    # Create output file path
    output_file = os.path.join(output_dir, f"{sample_id}_{chrom}.{file_type}")

    # Skip if output file already exists
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        logging.info(f"Sample VCF already exists: {output_file}. Skipping extraction.")
        return sample_id, chrom, output_file

    try:
        with pysam.VariantFile(vcf_path) as vcf_in:
            # Check if sample exists in the VCF
            if sample_id not in vcf_in.header.samples:
                logging.error(f"Sample {sample_id} not found in {vcf_path}")
                return None

            # For high memory environments, load everything at once
            with pysam.VariantFile(output_file, write_mode, header=vcf_in.header) as vcf_out:
                # Count for logging only
                total_variants = 0

                # Process all records in one go - with high RAM we don't need batching
                for record in vcf_in:
                    if sample_id in record.samples:
                        vcf_out.write(record)
                        total_variants += 1

        # Index the extracted VCF
        pysam.tabix_index(output_file, preset=file_type, force=True)

        logging.info(
            f"Successfully extracted sample {sample_id} for {chrom} - {total_variants} variants"
        )
        return sample_id, chrom, output_file

    except Exception as e:
        logging.error(f"Failed to extract sample {sample_id} from {vcf_path} using pysam: {e}")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass
        return None


def merge_sample_vcfs(
    sample_id: str,
    vcf_paths: List[str],
    output_dir: str,
    file_type: str = "vcf",
    write_mode: str = "w",
) -> Optional[str]:
    """
    Merge multiple chromosome VCFs for a single sample using a multi-way merge.

    Args:
        sample_id: Sample ID
        vcf_paths: List of sorted VCF file paths to merge (assumed sorted by contig and pos)
        output_dir: Directory to save the merged VCF
        write_mode:

    Returns:
        Path to the merged VCF file or None if merging failed
    """
    if not vcf_paths:
        logging.error(f"No VCF files provided for sample {sample_id}")
        return None

    output_file = os.path.join(output_dir, f"{sample_id}_merged.{file_type}")
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        logging.info(f"Merged VCF already exists: {output_file}. Skipping merge.")
        return output_file

    try:
        # Open each VCF file as an iterator (they are assumed sorted)
        vcf_iters = [pysam.VariantFile(path) for path in sorted(vcf_paths)]

        # Use heapq.merge to perform a multi-way merge based on (contig, pos)
        merged_iter = heapq.merge(*vcf_iters, key=lambda r: (r.contig, r.pos))

        # Write merged records out in batches to reduce per-record overhead.
        with pysam.VariantFile(vcf_paths[0]) as first_vcf:
            # Using "w" mode (uncompressed) for speed.
            with pysam.VariantFile(output_file, write_mode, header=first_vcf.header) as vcf_out:
                BATCH_SIZE = 10000
                batch = []
                count = 0
                for record in merged_iter:
                    batch.append(record)
                    if len(batch) >= BATCH_SIZE:
                        for rec in batch:
                            vcf_out.write(rec)
                        count += len(batch)
                        batch = []
                if batch:
                    for rec in batch:
                        vcf_out.write(rec)
                    count += len(batch)

        # Close all VCF iterators
        for vf in vcf_iters:
            vf.close()

        logging.info(f"Successfully merged {count} variants for sample {sample_id}")
        pysam.tabix_index(output_file, preset=file_type, force=True)
        return output_file

    except Exception as e:
        logging.error(f"Failed to merge VCFs for sample {sample_id}: {e}")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                logging.info(f"Removed incomplete output file: {output_file}")
            except:
                pass
        return None


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup directories
    base_dir = config.get("base_dir", "")
    if not base_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Use paths from the config structure
    paths_config = config["paths"]

    data_dir = ensure_dir(os.path.join(base_dir, paths_config["data_dir"]))
    merged_dir = ensure_dir(os.path.join(data_dir, paths_config["merged_dir"]))
    output_dir = ensure_dir(os.path.join(base_dir, paths_config["output_dir"]))
    chr_dir = ensure_dir(os.path.join(data_dir, paths_config["chromosomes_dir"]))
    sample_dir = ensure_dir(os.path.join(data_dir, paths_config["samples_dir"]))

    # Setup logging
    log_config = config["logging"]
    log_level = log_config.get("level", "INFO")
    log_file = "download_1000g.log"
    if log_file:
        log_file = os.path.join(output_dir, log_file)
    setup_logging(log_level, log_file)

    # Check if we should download 1000g data
    if not config["download"]["download_1000g"]:
        logging.info("1000 Genomes download is disabled in config. Exiting.")
        sys.exit(0)

    # Parse download options
    download_config = config["download"]
    n_samples = download_config["n_samples"]
    population = download_config["population"]
    chromosomes = download_config["chromosomes"]
    max_workers = download_config["max_workers"]
    extract_samples = download_config["extract_samples"]
    merge_samples = download_config["merge_samples"]
    compress = download_config["compress"]

    # Parse chromosomes
    if chromosomes == "all":
        chromosomes = all_chromosomes
    else:
        # Validate chromosomes
        for chrom in chromosomes:
            if chrom not in all_chromosomes:
                logging.warning(f"Chromosome {chrom} is not a standard chromosome. Skipping.")
                chromosomes.remove(chrom)

        if not chromosomes:
            logging.error("No valid chromosomes specified.")
            sys.exit(1)

    write_mode = "w"
    file_type = "vcf"
    if compress:
        write_mode = "wb"
        file_type = "bcf"

    logging.info(f"Configuration loaded from {args.config}")
    logging.info(f"Downloading {n_samples} samples, population: {population or 'any'}")
    logging.info(f"Chromosomes: {', '.join(chromosomes)}")
    logging.info(f"Using 1000 Genomes Project data with GRCh38 coordinates")

    # Get VCF and panel information
    logging.info("Fetching VCF and panel information")
    vcf_urls = get_vcf_locations(chromosomes)

    logging.debug(f"Found {len(vcf_urls)} VCF files for the selected chromosomes")

    # Create phenotype CSV and get sample list
    phenotype_file = os.path.join(data_dir, paths_config["phenotype_csv"])
    logging.info(f"Creating phenotype panel from {SAMPLE_INFO_URL}")
    phenotype_file, sample_list = create_phenotype_panel(
        SAMPLE_INFO_URL, phenotype_file, n_samples, population
    )

    logging.info(
        f"Selected {len(sample_list)} samples: {', '.join(sample_list[:5])}{'...' if len(sample_list) > 5 else ''}"
    )

    # Download chromosome VCFs
    chrom_vcf_paths = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(download_chr_vcf, url, chrom, chr_dir): url
            for chrom, url in vcf_urls.items()
        }

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                vcf_path, index_path, chrom = future.result()
                if vcf_path and index_path:
                    chrom_vcf_paths[chrom] = vcf_path
            except Exception as e:
                logging.error(f"Error downloading VCF from {url}: {e}")

    if not chrom_vcf_paths:
        logging.error("Failed to download any chromosome VCFs. Exiting.")
        sys.exit(1)

    logging.info(f"Successfully downloaded {len(chrom_vcf_paths)} chromosome VCFs")

    if extract_samples:
        # Extract VCFs per sample using ProcessPoolExecutor for CPU-heavy tasks
        sample_chr_vcfs = {sample_id: {} for sample_id in sample_list}

        logging.info("Extracting sample VCFs from chromosome VCFs")

        # Process each chromosome for each sample
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for chrom, vcf_path in chrom_vcf_paths.items():
                for sample_id in sample_list:
                    futures.append(
                        executor.submit(
                            extract_sample_vcf,
                            vcf_path,
                            sample_id,
                            chrom,
                            sample_dir,
                            file_type,
                            write_mode,
                        )
                    )

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        # Extract sample_id and chrom from the filename
                        sample_id, chrom, vcf_path = result
                        sample_chr_vcfs[sample_id][chrom] = vcf_path
                except Exception as e:
                    logging.error(f"Error extracting sample VCF: {e}")

        if not sample_chr_vcfs:
            logging.error("Failed to extract any sample VCFs. Exiting.")
            sys.exit(1)

    # Merge chromosomes per sample using ProcessPoolExecutor
    if extract_samples and merge_samples:
        logging.info("Merging chromosome VCFs for each sample")
        merged_sample_vcfs = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for sample_id, chr_vcfs in sample_chr_vcfs.items():
                if chr_vcfs:  # Only process samples with at least one chromosome VCF
                    vcf_paths = list(chr_vcfs.values())
                    futures[
                        executor.submit(
                            merge_sample_vcfs,
                            sample_id,
                            vcf_paths,
                            merged_dir,
                            file_type,
                            write_mode,
                        )
                    ] = sample_id

            for future in concurrent.futures.as_completed(futures):
                sample_id = futures[future]
                try:
                    merged_vcf = future.result()
                    if merged_vcf:
                        merged_sample_vcfs[sample_id] = merged_vcf
                except Exception as e:
                    logging.error(f"Error merging VCFs for sample {sample_id}: {e}")

        if not merged_sample_vcfs:
            logging.error("Failed to merge any sample VCFs. Exiting.")
            sys.exit(1)

        logging.info(f"Successfully merged VCFs for {len(merged_sample_vcfs)} samples")

    logging.info("1000 Genomes download and processing completed.")


if __name__ == "__main__":
    main()

import os
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd
import numpy as np
import pysam
import multiprocessing
import logging
import re
import random
from functools import lru_cache
from intervaltree import IntervalTree
from typing import Tuple, List, Dict, Optional, Union, Iterator, Generator

# Import from local files
from tokenizer import NTLevelTokenizer
from utils import map_chrom_to_ref, map_ref_to_chrom, all_chromosomes
import glob  # Added for file searching


class GenomicIterableDataset(IterableDataset):
    """
    Iterable Dataset for lazily loading genomic data from 1000 Genomes VCF files.
    This version pre-computes all sampling regions and shuffles them if requested,
    then iterates through the planned regions.

    Args:
        config: Configuration dictionary with all settings
        tokenizer: Tokenizer instance that implements encode() method
    """

    def __init__(
        self,
        config: Dict,
        tokenizer=None,
    ):
        self.config = config
        self.base_dir = config["base_dir"]
        if not self.base_dir:
            # Assuming the script runs relative to the project root where config is
            self.base_dir = os.path.abspath(".")  # Fallback to current dir

        self.data_dir = os.path.join(self.base_dir, config["paths"]["data_dir"])

        # --- Determine VCF data directory based on configuration ---
        # TODO simplify, lets just input a folder and take all that matches a regex

        self.extract_samples = config["download"]["extract_samples"]
        self.merge_samples = config["download"]["merge_samples"]

        chromosomes_dir_rel = config["paths"]["chromosomes_dir"]
        samples_dir_rel = config["paths"]["samples_dir"]
        merged_dir_rel = config["paths"]["merged_dir"]

        compressed = config["download"]["compress"]
        file_type = "vcf"
        if compressed:
            file_type = "bcf"

        if self.extract_samples:
            if self.merge_samples:
                # One merged file per sample
                self.vcf_data_dir = os.path.join(self.data_dir, merged_dir_rel)
                self.file_format_pattern = "{sample_id}_merged." + file_type
            else:
                # One file per sample per chromosome
                self.vcf_data_dir = os.path.join(self.data_dir, samples_dir_rel)
                self.file_format_pattern = "{sample_id}_{chrom}." + file_type
        else:
            # One file per chromosome, all samples included
            self.vcf_data_dir = os.path.join(self.data_dir, chromosomes_dir_rel)
            self.file_format_pattern = "chr{chrom}.vcf.gz"

        logging.info(f"Using VCF data directory: {self.vcf_data_dir}")
        # --- End Directory Selection ---

        # Set up other paths based on the project structure
        self.reference_dir = os.path.join(self.data_dir, config["paths"]["reference_dir"])
        self.reference_path = os.path.join(self.reference_dir, "hg38.fa")
        self.phenotype_path = os.path.join(self.data_dir, config["paths"]["phenotype_csv"])
        self.exon_bed_path = os.path.join(self.reference_dir, "hg38_exons.bed.sorted.gz")
        self.repeat_bed_path = os.path.join(self.reference_dir, "hg38_repeats.bed.sorted.gz")

        # Extract dataloader settings
        dataloader_config = config["dataloader"]
        data_config = dataloader_config["data"]

        # Set dataset parameters
        self.tokenizer = tokenizer
        self.chunk_size = data_config["chunk_size"]
        self.chromosomes = data_config["chromosomes"]
        if self.chromosomes == "all":
            self.chromosomes = all_chromosomes
        self.sample_id_column = data_config["sample_id_column"]
        self.phenotype_columns = data_config["phenotype_columns"]
        self.samples_per_sample = dataloader_config["samples_per_sample"]
        self.shuffle = dataloader_config["shuffle"]

        self.skip_n_repeat = data_config["skip_n_repeat"]
        self.cast_to_upper = data_config["cast_to_upper"]
        self.max_indel_length = data_config["max_indel_length"]
        self.burnin_len = data_config["burnin_length"]

        self.gap_char = "D"

        # Load phenotype data
        self._load_phenotype_data()  # Sets self.phenotype_df, self.samples_with_phenotypes

        # Load reference genome
        self._load_reference_genome()  # Sets self.reference_genome, self.chromosome_lengths

        # Initialize annotation dictionaries
        self.exon_regions = {}
        self.repeat_regions = {}

        # Load exon and repeat annotations into memory
        self._load_annotation_data()

        self.sampling_plan = self._prepare_sampling_plan()
        logging.info(f"Prepared sampling plan with {len(self.sampling_plan)} total regions.")

    def _load_phenotype_data(self):
        """Load phenotype data from CSV file."""
        try:
            self.phenotype_df = pd.read_csv(self.phenotype_path)

            # Ensure sample IDs are strings for consistent comparison
            self.phenotype_df[self.sample_id_column] = self.phenotype_df[
                self.sample_id_column
            ].astype(str)

            # Determine phenotype columns if not specified
            if self.phenotype_columns is None:
                self.phenotype_columns = list(self.phenotype_df.columns)
                if self.sample_id_column in self.phenotype_columns:
                    self.phenotype_columns.remove(self.sample_id_column)

            # Create set of samples with phenotype data for quick lookup
            self.samples_with_phenotypes = set(self.phenotype_df[self.sample_id_column])

            logging.info(f"Loaded phenotype data for {len(self.samples_with_phenotypes)} samples")

        except Exception as e:
            logging.error(f"Error loading phenotype data: {e}")
            raise ValueError(f"Failed to load phenotype data from {self.phenotype_path}")

    def _load_reference_genome(self):
        """Load reference genome from FASTA file."""
        try:
            if not os.path.exists(self.reference_path):
                raise FileNotFoundError(f"Reference genome not found at {self.reference_path}")

            self.reference_genome = pysam.FastaFile(self.reference_path)

            # Map chromosomes to their lengths
            self.chromosome_lengths = {}
            for chrom in self.chromosomes:
                ref = map_chrom_to_ref(chrom)
                self.chromosome_lengths[chrom] = self.reference_genome.get_reference_length(ref)

            logging.info(f"Loaded reference genome with {len(self.chromosome_lengths)} chromosomes")

        except Exception as e:
            logging.error(f"Error loading reference genome: {e}")
            raise ValueError(f"Failed to load reference genome from {self.reference_path}")

    def _load_annotation_data(self):
        """
        Load exon and repetitive region annotations into memory data structures.
        Uses interval trees for efficient region lookups.
        """
        logging.info("Loading genomic annotations into memory...")

        # Check if annotation files exist
        if not os.path.exists(self.exon_bed_path):
            raise FileNotFoundError(f"Exon annotations not found at {self.exon_bed_path}")

        if not os.path.exists(self.repeat_bed_path):
            raise FileNotFoundError(
                f"Repetitive region annotations not found at {self.repeat_bed_path}"
            )

        # Initialize interval trees for each chromosome
        for chrom in self.chromosomes:
            self.exon_regions[chrom] = IntervalTree()
            self.repeat_regions[chrom] = IntervalTree()

        # Load exon regions
        try:
            with pysam.TabixFile(self.exon_bed_path) as exon_tabix:
                available_contigs = set(exon_tabix.contigs)
                for chrom in self.chromosomes:
                    ref_chrom = map_chrom_to_ref(chrom)
                    try:
                        for line in exon_tabix.fetch(ref_chrom):
                            fields = line.split("\t")
                            if len(fields) >= 3:
                                start, end = int(fields[1]), int(fields[2])
                                # BED is 0-based, half-open; IntervalTree is [begin, end)
                                self.exon_regions[chrom].addi(start, end + 1, True)
                    except (ValueError, OSError) as e:
                        logging.warning(f"No exon data found for chromosome {chrom}: {e}")
                        continue
        except Exception as e:
            logging.error(f"Error loading exon annotations: {e}")
            raise e

        # Load repetitive regions
        try:
            with pysam.TabixFile(self.repeat_bed_path) as repeat_tabix:
                available_contigs = set(repeat_tabix.contigs)
                for chrom in self.chromosomes:
                    ref_chrom = map_chrom_to_ref(chrom)
                    try:
                        for line in repeat_tabix.fetch(ref_chrom):
                            fields = line.split("\t")
                            if len(fields) >= 3:
                                start, end = int(fields[1]), int(fields[2])
                                self.repeat_regions[chrom].addi(start, end + 1, True)
                    except (ValueError, OSError) as e:
                        logging.warning(
                            f"Could not fetch repetitive region data for chromosome {chrom} (ref: {ref_chrom}): {e}"
                        )
                        continue
        except Exception as e:
            logging.error(f"Error loading repetitive region annotations: {e}")
            raise e

        exon_count = sum(len(tree) for tree in self.exon_regions.values())
        repeat_count = sum(len(tree) for tree in self.repeat_regions.values())
        logging.info(
            f"Successfully loaded {exon_count} exon regions and {repeat_count} repetitive regions into memory"
        )

    def _prepare_sampling_plan(self) -> List[Tuple[str, str, int, str]]:
        """
        Generates a list of all possible sampling regions across all relevant VCF files and samples.
        Each element represents one chunk: (vcf_file_path, chromosome, start_pos, sample_id).
        Applies samples_per_sample logic and shuffling if enabled.

        Regions are planned with margins to allow for random offsets during iteration.
        """
        regions_by_sample = {sample_id: [] for sample_id in self.samples_with_phenotypes}
        required_chunk_length = self.chunk_size + self.burnin_len

        if self.extract_samples:
            # Files are organized by sample (either merged or per-chromosome)
            for sample_id in self.samples_with_phenotypes:
                for chrom in self.chromosomes:
                    chrom_len = self.chromosome_lengths.get(chrom)
                    if not chrom_len:
                        continue  # Skip if chromosome length unknown

                    if self.merge_samples:
                        # Expect merged file: {sample_id}_merged.vcf.gz
                        file_pattern = os.path.join(
                            self.vcf_data_dir, self.file_format_pattern.format(sample_id=sample_id)
                        )
                    else:
                        # Expect per-chromosome file: {sample_id}_{chrom}.vcf.gz
                        file_pattern = os.path.join(
                            self.vcf_data_dir,
                            self.file_format_pattern.format(sample_id=sample_id, chrom=chrom),
                        )

                    # Use glob to find the exact file, handling potential variations
                    matching_files = glob.glob(file_pattern)
                    if not matching_files:
                        logging.warning(f"VCF file not found for pattern: {file_pattern}")
                        continue
                    elif len(matching_files) > 1:
                        logging.warning(
                            f"Multiple files matched pattern {file_pattern}, using first: {matching_files[0]}"
                        )

                    file_path = matching_files[0]  # Assume first match is correct

                    # Generate regions for this file/chromosome
                    # Leave margin for burnin, random offsets and lookahead
                    available_len = chrom_len - self.burnin_len - self.chunk_size + 1

                    num_regions = max(0, available_len // self.chunk_size)
                    if chrom_len >= required_chunk_length and num_regions == 0:
                        num_regions = 1
                    num_regions = int(num_regions)

                    # If chromosome is shorter than needed do not sample. We only sample if full length is available.
                    for i in range(num_regions):
                        start = i * self.chunk_size
                        # Ensure the region including burnin and lookahead AND potential offset fits within the chromosome
                        if start + required_chunk_length + (self.chunk_size - 1) <= chrom_len:
                            regions_by_sample[sample_id].append((file_path, chrom, start))
                        else:  # debug
                            logging.debug(
                                f"Skipping region {chrom}:{start}-{start+required_chunk_length} for sample {sample_id} (len={chrom_len})"
                            )

                if self.merge_samples and not regions_by_sample[sample_id]:
                    # Check if the merged file was missing only once
                    file_pattern = os.path.join(
                        self.vcf_data_dir, self.file_format_pattern.format(sample_id=sample_id)
                    )
                    if not glob.glob(file_pattern):
                        logging.warning(f"Merged VCF file not found for pattern: {file_pattern}")

        else:
            # Files are organized by chromosome (extract_samples=False)
            for chrom in self.chromosomes:
                chrom_len = self.chromosome_lengths.get(chrom)
                if not chrom_len:
                    logging.warning(f"Unknown chromosome: {chrom}")
                    continue

                file_pattern = os.path.join(
                    self.vcf_data_dir, self.file_format_pattern.format(chrom=chrom)
                )
                matching_files = glob.glob(file_pattern)

                if not matching_files:
                    logging.warning(f"Chromosome VCF file not found for pattern: {file_pattern}")
                    continue
                elif len(matching_files) > 1:
                    logging.warning(
                        f"Multiple files matched chromosome pattern {file_pattern}, using first: {matching_files[0]}"
                    )

                file_path = matching_files[0]

                # Generate regions for this chromosome
                # Modified: Leave margin for random offsets
                available_len = chrom_len - self.burnin_len - (self.chunk_size - 1)
                num_regions = max(0, available_len // self.chunk_size)

                if chrom_len >= required_chunk_length and num_regions == 0:
                    num_regions = 1

                num_regions = int(num_regions)
                regions_for_chrom = []
                for i in range(num_regions):
                    start = i * self.chunk_size
                    if start + required_chunk_length + (self.chunk_size - 1) <= chrom_len:
                        regions_for_chrom.append((file_path, chrom, start))

                # Assign these regions to *each* sample
                for sample_id in self.samples_with_phenotypes:
                    regions_by_sample[sample_id].extend(regions_for_chrom)

        # Apply samples_per_sample limit and flatten
        final_sampling_plan = []
        for sample_id, regions in regions_by_sample.items():
            if not regions:
                logging.warning(
                    f"No valid sampling regions found for sample {sample_id}. Skipping."
                )
                continue

            if self.shuffle:
                random.shuffle(regions)  # Shuffle regions *within* a sample

            if self.samples_per_sample > 0:  # Use 0 or negative to indicate no limit
                if self.samples_per_sample <= len(regions):
                    selected_regions = regions[: self.samples_per_sample]
                else:
                    # Repeat regions to reach the desired count
                    repeats = (self.samples_per_sample + len(regions) - 1) // len(regions)
                    # Ceiling division
                    selected_regions = (regions * repeats)[: self.samples_per_sample]
                    logging.debug(
                        f"Repeating {len(regions)} regions {repeats} times for sample {sample_id} to get {self.samples_per_sample} samples."
                    )
            else:
                selected_regions = regions  # Use the single found regions

            # Add sample_id back to each selected region tuple
            final_sampling_plan.extend([(fp, ch, st, sample_id) for fp, ch, st in selected_regions])

        # Shuffle the final combined list across all samples if requested
        if self.shuffle:
            random.shuffle(final_sampling_plan)

        logging.info(
            f"Generated {len(final_sampling_plan)} regions after applying samples_per_sample ({self.samples_per_sample})"
        )
        return final_sampling_plan

    def _get_reference_sequence(self, chrom: str, start: int, end: int) -> str:
        """Get sequence from reference genome, using an internal cache."""
        # Note: Using pysam's internal caching; decorator cache removed for simplicity
        try:
            ref_seq = self.reference_genome.fetch(map_chrom_to_ref(chrom), start, end)
            if self.cast_to_upper:
                ref_seq = ref_seq.upper()
            return ref_seq
        except Exception as e:
            logging.error(f"Error fetching reference sequence for {chrom}:{start}-{end}: {e}")
            raise e

    def _is_exon(self, chrom: str, pos: int) -> bool:
        """Check if a position is in an exon using the pre-loaded interval trees."""
        try:
            if chrom not in self.exon_regions:
                return False
            return self.exon_regions[chrom].overlaps(pos)
        except Exception as e:
            logging.debug(f"Error checking exon status for {chrom}:{pos}: {e}")
            return False

    def _is_repetitive(self, chrom: str, pos: int) -> bool:
        """Check if a position is in a repetitive region using the pre-loaded interval trees."""
        try:
            if chrom not in self.repeat_regions:
                return False
            return self.repeat_regions[chrom].overlaps(pos)
        except Exception as e:
            logging.debug(f"Error checking repetitive status for {chrom}:{pos}: {e}")
            return False

    def _is_n_repeat_sequence(self, sequence: str) -> bool:
        """Check if a sequence consists only of 'N' characters."""
        return not sequence or sequence.upper().count("N") == len(sequence)

    def _create_region_masks(
        self, chrom: str, start: int, length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create exon and repetitive region masks for a genomic region using vectorized operations.
        Returns:
            Tuple of (exon_mask, repetitive_mask) as numpy boolean arrays.
        """
        exon_mask = np.zeros(length, dtype=bool)
        repetitive_mask = np.zeros(length, dtype=bool)

        region_start = start
        region_end = start + length  # exclusive end
        try:
            # Exon mask
            for iv in self.exon_regions[chrom].overlap(region_start, region_end):
                # Calculate intersection of interval [iv.begin, iv.end) with [region_start, region_end)
                # Convert to mask indices [0, length) relative to region_start
                s = max(0, iv.begin - region_start)
                e = min(length, iv.end - region_start)
                if e > s:  # Ensure valid slice
                    exon_mask[s:e] = True

            # Repetitive mask
            for iv in self.repeat_regions[chrom].overlap(region_start, region_end):
                s = max(0, iv.begin - region_start)
                e = min(length, iv.end - region_start)
                if e > s:
                    repetitive_mask[s:e] = True
        except Exception as e:
            # Catch potential errors during interval processing
            logging.error(f"Error creating masks for {chrom}:{region_start}-{region_end}: {e}")
            # Return empty masks in case of error
            return np.zeros(length, dtype=bool), np.zeros(length, dtype=bool)

        return exon_mask, repetitive_mask

    def _get_phenotype_vector(self, sample_id: str) -> torch.Tensor:
        """Get phenotype vector for a sample."""
        # Use .loc for potentially faster lookup if index is set, but works either way
        sample_row = self.phenotype_df[self.phenotype_df[self.sample_id_column] == sample_id]
        if sample_row.empty:
            raise ValueError(f"sample {sample_id} not found in phenotype data")
        phenotype_values = sample_row[self.phenotype_columns].iloc[0].astype(float).values

        return torch.tensor(phenotype_values, dtype=torch.float32)

    def _extract_variants_from_region(
        self, vcf: pysam.VariantFile, chrom: str, start: int, end: int, sample_id=None
    ) -> List[Tuple[str, int, str, Tuple[str]]]:
        """
        Extract variants from an open VCF file for a given region using pysam.
        The VCF handle `vcf` might be pre-filtered for a specific sample if extract_samples=False.
        """
        variants = []
        ref_chrom = map_chrom_to_ref(chrom)
        try:
            # Fetch variants. If vcf was opened with samples=[sample_id],
            # fetch will only yield records where that sample has data (often non-reference GT).
            fetched_records = vcf.fetch(ref_chrom, start, end)
            for record in fetched_records:
                if sample_id is not None and sample_id not in record.samples:
                    continue
                # Basic filtering (can be expanded)
                # Check if REF and ALT alleles are valid (not None, not empty)
                if record.ref is None or not record.alts or any(alt is None for alt in record.alts):
                    continue

                # Apply length constraint (using self.max_indel_length)
                if len(record.ref) > self.max_indel_length or any(
                    len(alt) > self.max_indel_length for alt in record.alts
                ):
                    continue

                # Store the variant details
                # Ensure alts is a tuple of strings
                alts_tuple = tuple(str(alt) for alt in record.alts if alt is not None)
                if not alts_tuple:
                    continue  # Skip if no valid alts after filtering Nones

                # record.pos is 1-based
                variants.append((chrom, record.pos, str(record.ref), alts_tuple))

        except ValueError as e:
            # Catch errors like "invalid region" if chrom not in VCF header
            logging.warning(
                f"Error fetching variants for region {chrom}:{start}-{end} (ref: {ref_chrom}). Is contig in VCF? Error: {e}"
            )
        except Exception as e:
            logging.error(f"Unexpected error fetching variants for {chrom}:{start}-{end}: {e}")
        return variants

    def _apply_variants(
        self,
        reference_seq: str,
        chrom: str,
        start: int,
        variants: List[Tuple[str, int, str, Tuple[str]]],
    ) -> str:
        """Apply variants to reference sequence."""
        # Convert to mutable list for editing
        sequence = list(reference_seq)
        end = start + len(reference_seq)
        # Filter variants in this region
        if not variants:
            logging.debug(f"No variants found for {chrom}:{start}-{end}")
            return reference_seq

        variants = iter(variants)
        v_chrom, v_pos, v_ref, v_alts = next(variants)

        try:
            while v_chrom < chrom:
                v_chrom, v_pos, v_ref, v_alts = next(variants)
            while v_pos < start:
                v_pos, v_ref, v_alts = next(variants)
        except StopIteration:
            logging.debug(f"No variants found for {chrom}:{start}-{end}")
            return reference_seq

        while v_pos < end:
            # Convert from 1-based to 0-based and adjust for region start
            idx = v_pos - 1 - start

            # Skip if outside our region or at the very end
            if idx < 0 or idx >= len(sequence):
                continue

            if len(v_alts) > 1:
                # Multi-allelic variant - choose one randomly
                alt = random.choice(v_alts)
            else:
                alt = v_alts[0]

            # Replace reference with alternate allele
            ref_len = len(v_ref)
            alt_len = len(alt)

            if alt_len < ref_len:
                # Deletion - represent with "D" character (for token 5 in NTLevelTokenizer)
                sequence[idx : idx + ref_len] = list(alt) + ["D"] * (ref_len - alt_len)
            elif alt_len == ref_len:
                # Simple substitution
                sequence[idx : idx + ref_len] = list(alt)
            else:
                # Insertion - might need to handle this differently
                sequence[idx : idx + ref_len] = list(alt)

            # Move to the next variant
            try:
                v_pos, v_ref, v_alts = next(variants)
            except StopIteration:
                break

        # Join back into a string, keeping "D" characters for deletions
        sequence = "".join(sequence)
        if self.cast_to_upper:
            sequence = sequence.upper()
        return sequence

    def __len__(self) -> int:
        """Return the total number of chunks planned for the dataset across all workers."""
        return len(self.sampling_plan)

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            str,
        ]
    ]:
        """Iterate through the pre-computed and shuffled sampling plan."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process data loading
            iter_start = 0
            iter_end = len(self.sampling_plan)
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            per_worker = int(np.ceil(len(self.sampling_plan) / num_workers))
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.sampling_plan))
            # Set random seed for this worker based on base seed and worker id
            np.random.seed((torch.initial_seed() + worker_id) % (2**32))
            random.seed((torch.initial_seed() + worker_id) % (2**32))

        # Generate random offset for this epoch (same for all workers in this epoch)
        epoch_seed = torch.initial_seed() % (2**32)  # Get current epoch's seed
        rng = random.Random(epoch_seed)
        random_offset = rng.randint(0, self.chunk_size - 1)

        logging.debug(
            f"Worker {worker_id}/{num_workers} processing indices [{iter_start}, {iter_end}) with random offset {random_offset}"
        )

        # Iterate through the assigned slice of the sampling plan
        for i in range(iter_start, iter_end):
            try:
                file_path, chrom, start, sample_id = self.sampling_plan[i]

                # 1. Get Phenotype Vector
                phenotype_vector = self._get_phenotype_vector(sample_id)
                # Skip if phenotype couldn't be retrieved (e.g., returned NaN tensor)
                if torch.isnan(phenotype_vector).any():
                    logging.warning(
                        f"Skipping region {chrom}:{start} for sample {sample_id} due to missing phenotype."
                    )
                    continue

                # 2. Get Reference Sequence (including burn-in)
                fetch_start = start + random_offset
                fetch_end = fetch_start + self.chunk_size + self.burnin_len
                reference_seq_full = self._get_reference_sequence(chrom, fetch_start, fetch_end)

                if not reference_seq_full:  # Handle error from _get_reference_sequence
                    logging.warning(
                        f"Skipping region {chrom}:{start} for sample {sample_id} due to reference fetch error."
                    )
                    continue

                # 3. Skip if sequence is all 'N's (after potential casting)
                if self.skip_n_repeat and self._is_n_repeat_sequence(
                    reference_seq_full[self.burnin_len :]
                ):
                    # logging.debug(f"Skipping N-repeat sequence at {chrom}:{start + self.burnin_len}")
                    continue

                # 4. Open VCF and Extract Variants for the specific sample/region
                # Determine if sample filtering is needed when opening
                try:
                    # Use context manager for automatic closing
                    with pysam.VariantFile(file_path, mode="r") as vcf:
                        # Ensure sample exists in header if filtering was requested
                        if not self.extract_samples and sample_id not in vcf.header.samples:
                            logging.error(
                                f"Sample {sample_id} not found in header of VCF {file_path}. Skipping region."
                            )
                            continue  # Skip this region

                        # Extract variants for the full region including burn-in
                        variants = self._extract_variants_from_region(
                            vcf,
                            chrom,
                            fetch_start,
                            fetch_end,
                            sample_id,
                        )

                except FileNotFoundError:
                    logging.error(f"VCF file not found at path: {file_path}. Skipping region.")
                    continue
                except ValueError as ve:  # Catch specific pysam errors like sample not found
                    logging.error(
                        f"Error opening or processing VCF {file_path} for sample {sample_id}: {ve}. Skipping region."
                    )
                    continue
                except Exception as e:  # Catch other potential errors during VCF handling
                    logging.error(
                        f"Unexpected error with VCF {file_path} for sample {sample_id}: {e}. Skipping region."
                    )
                    continue

                # 5. Apply Variants
                sample_seq_full = self._apply_variants(
                    reference_seq_full,
                    chrom,
                    fetch_start,  # Start position of reference_seq_full
                    variants,
                )

                # 6. Remove Burn-in Region
                reference_seq = reference_seq_full[self.burnin_len :]
                sample_seq = sample_seq_full[self.burnin_len :]

                # Ensure sequences have the expected chunk_size length after burn-in removal
                if len(reference_seq) != self.chunk_size or len(sample_seq) != self.chunk_size:
                    logging.warning(
                        f"Sequence length mismatch after burn-in removal for {chrom}:{start}. Ref: {len(reference_seq)}, Sample: {len(sample_seq)}. Expected: {self.chunk_size}. Skipping."
                    )
                    # This cloud happen if the chromosome end was reached within the burn-in/chunk.
                    # Should never happen if we created the sampling plan correctly.
                    continue

                # 7. Create Annotation Masks (relative to the final chunk start)
                mask_start = start + self.burnin_len
                exon_mask, repetitive_mask = self._create_region_masks(
                    chrom, mask_start, self.chunk_size  # Length is chunk_size
                )

                # 8. Tokenize (if tokenizer provided)
                if self.tokenizer is not None:
                    try:
                        tokenized_sample = self.tokenizer.tokenize(sample_seq)
                    except Exception as e:
                        logging.error(
                            f"Tokenization failed for {chrom}:{start} sample {sample_id}: {e}"
                        )
                        continue

                # 9. Create Position Tensor (relative position within chromosome)
                # Position corresponds to the start of the chunk after burn-in
                position_indices = torch.arange(
                    start + self.burnin_len,
                    start + self.burnin_len + self.chunk_size,
                    dtype=torch.float32,
                )
                position = position_indices / self.chromosome_lengths[chrom]

                # 10. Yield Data Tuple
                yield (
                    torch.tensor(tokenized_sample),
                    phenotype_vector,
                    torch.tensor(exon_mask, dtype=torch.bool),
                    torch.tensor(repetitive_mask, dtype=torch.bool),
                    position,
                    chrom,
                )
            except Exception as e:
                # Catch unexpected errors during the processing of a single region
                logging.exception(
                    f"Worker {worker_id}: Unhandled error processing region index {i} ({self.sampling_plan[i]}): {e}"
                )
                # Continue to the next region
                continue

    def __del__(self):
        """Clean up resources when the dataset is destroyed."""
        if hasattr(self, "reference_genome") and self.reference_genome is not None:
            self.reference_genome.close()


def _collate_fn(batch):
    """
    Custom collate function for genomic data without lookahead.
    """
    (
        sample_seqs,
        phenotypes,
        exon_masks,
        repetitive_masks,
        positions,
        chrom,
    ) = zip(*batch)
    batched_sample_seqs = torch.stack(sample_seqs)
    batched_phenotypes = torch.stack(phenotypes)
    batched_exon_masks = torch.stack(exon_masks)
    batched_repetitive_masks = torch.stack(repetitive_masks)
    batched_positions = torch.stack(positions)

    return (
        batched_sample_seqs,
        batched_phenotypes,
        batched_exon_masks,
        batched_repetitive_masks,
        batched_positions,
        chrom,
    )


def create_genomic_dataloader(config, tokenizer=None):
    """
    Create a DataLoader for genomic data based on configuration settings.

    Args:
        config: Configuration dictionary with all settings
        tokenizer: Tokenizer instance (uses NTLevelTokenizer by default)

    Returns:
        DataLoader: PyTorch DataLoader
    """
    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = NTLevelTokenizer()

    dataset = GenomicIterableDataset(config, tokenizer)

    dataloader_config = config["dataloader"]
    num_workers = dataloader_config.get("num_workers")
    if num_workers is None:
        num_workers = min(8, multiprocessing.cpu_count())

    return DataLoader(
        dataset,
        batch_size=dataloader_config["batch_size"],
        num_workers=num_workers,
        pin_memory=dataloader_config["pin_memory"],
        prefetch_factor=dataloader_config["prefetch_factor"],
        persistent_workers=(num_workers > 0),
        collate_fn=_collate_fn,
    )

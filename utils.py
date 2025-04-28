"""
Utility functions for 1000 Genomes Project data processing.
"""

import os
import gzip
import shutil
import requests
import logging
import yaml
import subprocess
import numpy as np
import pandas as pd
import json
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


all_chromosomes = [str(i) for i in range(1, 23)]  # + ["X", "Y"]
chromosome_mapping = {str(i): i for i in range(1, 23)}
# chromosome_mapping.update({"X": 23, "Y": 24, "MT": 25})
nucleotides = ["A", "C", "G", "T", "N", "D"]


def map_chrom_to_ref(chrom) -> str:
    """Map chromosome name to reference format."""
    chrom = str(chrom).removeprefix("chr")
    if chrom == "MT":
        return "chrM"
    return "chr" + chrom


def map_ref_to_chrom(chrom) -> str:
    """Map chromosome name to reference format."""
    chrom = str(chrom).removeprefix("chr")
    if chrom == "M":
        return "MT"
    return chrom


def get_chrom_id(chrom) -> int:
    """Get the chromosome ID based on the chromosome name."""

    chrom = str(chrom).removeprefix("chr")
    return chromosome_mapping.get(chrom, 0)


def get_chrom_size(chrom: str) -> int:
    """
    Get the size (length) of a chromosome.

    Args:
        chrom: Chromosome name (without 'chr' prefix)

    Returns:
        Size of the chromosome in base pairs, or 0 if unknown
    """
    # Chromosome sizes based on GRCh38/hg38
    chrom_sizes = {
        "1": 248956422,
        "2": 242193529,
        "3": 198295559,
        "4": 190214555,
        "5": 181538259,
        "6": 170805979,
        "7": 159345973,
        "8": 145138636,
        "9": 138394717,
        "10": 133797422,
        "11": 135086622,
        "12": 133275309,
        "13": 114364328,
        "14": 107043718,
        "15": 101991189,
        "16": 90338345,
        "17": 83257441,
        "18": 80373285,
        "19": 58617616,
        "20": 64444167,
        "21": 46709983,
        "22": 50818468,
    }

    # Normalize chromosome name by removing 'chr' prefix if present
    chrom_str = str(chrom).removeprefix("chr")

    # Return the chromosome size or 0 if not found
    return chrom_sizes.get(chrom_str, 0)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration
    """
    if not config_path:
        raise ValueError("Config file path not provided")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError("Config file is empty")

    return config


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure logging with specified level and optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Configure handler
    handlers = []
    # Always add console handler
    console_handler = logging.StreamHandler()
    handlers.append(console_handler)

    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    # Configure logging
    logging.basicConfig(
        level=level_map.get(level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    return logging.getLogger(__name__)


def download_file(url: str, filepath: str, chunk_size: int = 8192) -> Optional[str]:
    """
    Download a file from URL to filepath with progress bar.

    Args:
        url: URL to download from
        filepath: Local path to save the file
        chunk_size: Size of chunks to download

    Returns:
        Path to downloaded file or None if download failed
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Check if file already exists and is non-empty
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logging.info(f"File already exists: {filepath}. Skipping download.")
            return filepath

        # Temporary file for download
        temp_filepath = filepath + ".tmp"

        # Set up requests with stream=True for large files
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Get file size if available
        file_size = int(response.headers.get("content-length", 0))

        # Write to file with progress bar
        with open(temp_filepath, "wb") as f, tqdm(
            desc=os.path.basename(filepath),
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        # Move temp file to final location
        shutil.move(temp_filepath, filepath)
        return filepath

    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return None


def run_command(cmd: List[str], check: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Run a shell command and return success status and error message.

    Args:
        cmd: List of command and arguments
        check: Whether to raise an exception if command fails

    Returns:
        Tuple of (success, error_message)
    """
    try:
        logging.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return True, None
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr
        logging.error(f"Command failed: {' '.join(cmd)}")
        logging.error(f"Error: {error_msg}")
        return False, error_msg
    except Exception as e:
        logging.error(f"Exception running command: {e}")
        return False, str(e)


def decompress_gzip(gz_file: str, output_file: str) -> Optional[str]:
    """
    Decompress a gzip file with progress tracking.

    Args:
        gz_file: Path to gzipped file
        output_file: Path for decompressed output

    Returns:
        Path to decompressed file or None if decompression failed
    """
    try:
        # Skip if output file already exists and is non-empty
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logging.info(
                f"Decompressed file already exists: {output_file}. Skipping decompression."
            )
            return output_file

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        logging.info(f"Decompressing {gz_file} to {output_file}")
        with gzip.open(gz_file, "rb") as f_in, open(output_file, "wb") as f_out, tqdm(
            desc=f"Decompressing {os.path.basename(gz_file)}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            # Read and write in chunks to avoid loading entire file into memory
            chunk_size = 8192
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                progress_bar.update(len(chunk))

        return output_file

    except Exception as e:
        logging.error(f"Error decompressing {gz_file}: {e}")
        # Clean up partial output file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
        return None


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists and return path.

    Args:
        path: Directory path

    Returns:
        Same path after ensuring directory exists
    """
    os.makedirs(path, exist_ok=True)
    return path

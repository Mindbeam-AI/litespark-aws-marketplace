#!/usr/bin/env python3
"""
Optimized SlimPajama Dataset Preparation Script for Litespark Training
Designed for high-performance processing on AWS g5.48xlarge instances.

This version uses a combination of batched processing and immediate disk writing
to efficiently handle large datasets while minimizing memory usage.

Usage:
    python prepare_data.py --input-dir /path/to/slimpajama --output-dir /path/to/tokenized_output --extension parquet
"""

import argparse
import glob
import json
import logging
import os
import random
import struct
import time
from typing import Dict, List, Optional, Tuple, Union
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import LlamaTokenizer
import concurrent.futures
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global tokenizer for thread safety
global_tokenizer = None
tokenizer_lock = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SlimPajama dataset for Litespark (Optimized)")
    parser.add_argument("--input-dir", type=str, required=True, 
                        help="Directory containing SlimPajama raw data files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save tokenized datasets")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b",
                        help="Tokenizer to use for tokenization")
    parser.add_argument("--seq-length", type=int, default=4096,
                        help="Sequence length for tokenized data")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of documents to process in each batch")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Maximum number of worker threads for processing")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to process (for testing)")
    parser.add_argument("--add-bos", action="store_true", default=True, 
                        help="Add BOS token to each document")
    parser.add_argument("--add-eos", action="store_true", default=True,
                        help="Add EOS token to each document")
    parser.add_argument("--extension", type=str, default="jsonl",
                        choices=["jsonl", "parquet", "txt"],
                        help="File extension of raw data files")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Column name for text in parquet/json files")
    parser.add_argument("--token-size", type=int, default=2,
                        help="Size of each token in bytes (2 for vocab<65536, 4 for larger)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--subset", type=str, default=None, choices=["train", "validation", "test"],
                        help="Process only a specific subset (train, validation, or test)")
    parser.add_argument("--skip-existing", action="store_true", 
                        help="Skip processing shards that already have output files")
    parser.add_argument("--write-interval", type=int, default=10000000,
                        help="Number of tokens to accumulate before writing to disk")
    
    return parser.parse_args()

def load_tokenizer(tokenizer_name: str):
    """Load tokenizer from HuggingFace hub or local path"""
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    
    try:
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False,
            legacy=True
        )
            
        logger.info(f"Loaded tokenizer with vocab size: {len(tokenizer)}")
        
        # Set special tokens if needed
        if tokenizer.bos_token_id is None and hasattr(tokenizer, 'bos_token'):
            logger.warning("BOS token not defined, setting to default")
            tokenizer.bos_token = '<s>'
            
        if tokenizer.eos_token_id is None and hasattr(tokenizer, 'eos_token'):
            logger.warning("EOS token not defined, setting to default")
            tokenizer.eos_token = '</s>'
            
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

def get_file_paths(input_dir: str, extension: str, max_files: Optional[int] = None, subset: Optional[str] = None) -> List[str]:
    """Get list of files to process, handling specific directory structures"""
    # Look for files with the specified extension in the directory tree
    if subset:
        pattern = os.path.join(input_dir, subset, f"**/*.{extension}")
    else:
        pattern = os.path.join(input_dir, f"**/*.{extension}")
    
    file_paths = glob.glob(pattern, recursive=True)
    
    if not file_paths:
        logger.error(f"No files found matching pattern: {pattern}")
        raise FileNotFoundError(f"No {extension} files found in {input_dir}")
    
    logger.info(f"Found {len(file_paths)} {extension} files")
    
    if max_files:
        file_paths = file_paths[:max_files]
        logger.info(f"Limited to {max_files} files")
        
    return file_paths

def process_parquet_file(file_path: str, text_column: str, batch_size: int = 1000) -> List[List[str]]:
    """Process a Parquet file and extract texts in batches"""
    try:
        logger.info(f"Reading parquet file: {file_path}")
        
        # Use PyArrow to read the parquet file efficiently
        parquet_file = pq.ParquetFile(file_path)
        
        # Get schema to check if text column exists
        schema = parquet_file.schema
        if text_column not in [field.name for field in schema]:
            # Try to find a suitable text column
            potential_text_columns = [field.name for field in schema if 'text' in field.name.lower()]
            if potential_text_columns:
                text_column = potential_text_columns[0]
                logger.warning(f"Specified text column not found, using {text_column} instead")
            else:
                logger.error(f"Text column '{text_column}' not found in parquet file columns")
                return []
        
        # Read and process in batches
        batches = []
        num_row_groups = parquet_file.num_row_groups
        total_docs = 0
        
        for i in range(num_row_groups):
            # Read one row group at a time
            table = parquet_file.read_row_group(i)
            df = table.to_pandas()
            
            # Extract text documents from this batch
            documents_in_group = []
            for text in df[text_column]:
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    documents_in_group.append(text)
            
            # Split into smaller batches if needed
            for j in range(0, len(documents_in_group), batch_size):
                batch = documents_in_group[j:j+batch_size]
                if batch:
                    batches.append(batch)
                    total_docs += len(batch)
        
        logger.info(f"Extracted {total_docs} documents in {len(batches)} batches from {file_path}")
        return batches
    
    except Exception as e:
        logger.error(f"Error processing parquet file {file_path}: {e}")
        return []

def tokenize_batch(batch: List[str], add_bos: bool = True, add_eos: bool = True) -> List[List[int]]:
    """Tokenize a batch of documents"""
    global global_tokenizer
    
    # Skip empty documents
    documents = [doc for doc in batch if doc and len(doc.strip()) > 0]
    
    # Acquire lock when using the tokenizer
    with tokenizer_lock:
        tokenized_documents = []
        for doc in documents:
            # Apply tokenizer with appropriate special tokens
            tokens = global_tokenizer.encode(doc, add_special_tokens=False)
            
            # Add special tokens if needed
            if add_bos and global_tokenizer.bos_token_id is not None:
                tokens = [global_tokenizer.bos_token_id] + tokens
            if add_eos and global_tokenizer.eos_token_id is not None:
                tokens = tokens + [global_tokenizer.eos_token_id]
                
            tokenized_documents.append(tokens)
    
    return tokenized_documents

def create_token_sequences(tokenized_documents: List[List[int]], seq_length: int) -> List[List[int]]:
    """Create sequences of tokens with specified length + 1 (for label)"""
    sequences = []
    current_seq = []
    
    # Flatten and create sequences
    for doc in tokenized_documents:
        # If current sequence + doc would be too long, finish current sequence
        if current_seq and len(current_seq) + len(doc) > seq_length + 1:
            # If we have enough tokens, save this sequence
            if len(current_seq) >= seq_length + 1:
                sequences.append(current_seq[:seq_length + 1])
                leftover = current_seq[seq_length + 1:]
                current_seq = leftover + doc
            else:
                # Not enough tokens, so add doc to current_seq
                current_seq.extend(doc)
        else:
            # Add document to current sequence
            current_seq.extend(doc)
            
        # Process any full sequences from the current accumulation
        while len(current_seq) >= seq_length + 1:
            sequences.append(current_seq[:seq_length + 1])
            current_seq = current_seq[seq_length + 1:]
    
    # Add the last incomplete sequence if it's close enough to target length
    if len(current_seq) >= (seq_length + 1) * 0.8:  # 80% of target length
        # Pad with EOS tokens if needed
        if len(current_seq) < seq_length + 1:
            padding_token = global_tokenizer.eos_token_id
            padding = [padding_token] * (seq_length + 1 - len(current_seq))
            current_seq.extend(padding)
        sequences.append(current_seq[:seq_length + 1])
    
    return sequences

def save_tokens_to_file(tokens: List[int], output_file: str, token_size: int):
    """Save tokens to .ds file in binary format"""
    format_char = 'H' if token_size == 2 else 'I'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'wb') as f:
            f.write(struct.pack(f"{len(tokens)}{format_char}", *tokens))
        logger.info(f"Saved {len(tokens)} tokens to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving tokens to {output_file}: {e}")
        return False

def process_file(file_path: str, args, shard_id: int) -> Tuple[int, int]:
    """Process a single file and return the number of sequences and tokens"""
    subset_path = None
    
    # Determine which subset this file belongs to
    for subdir in ['train', 'validation', 'test']:
        if subdir in file_path:
            subset_path = os.path.join(args.output_dir, subdir)
            break
    
    # Use output_dir if no subset was identified
    if subset_path is None:
        subset_path = args.output_dir
    
    # Create output file path
    output_file = os.path.join(subset_path, f"tokens_shard_{shard_id:06d}.ds")
    
    # Skip if file already exists and --skip-existing is set
    if args.skip_existing and os.path.exists(output_file):
        logger.info(f"Shard {shard_id} already exists at {output_file}, skipping")
        return 0, 0
    
    # Process file based on extension
    if args.extension == 'parquet':
        document_batches = process_parquet_file(file_path, args.text_column, args.batch_size)
    else:
        logger.error(f"Unsupported file extension: {args.extension}")
        return 0, 0
    
    total_sequences = 0
    total_tokens = 0
    all_tokens = []
    
    # Process each batch of documents
    for batch in document_batches:
        # Tokenize documents
        tokenized_docs = tokenize_batch(batch, add_bos=args.add_bos, add_eos=args.add_eos)
        
        # Create sequences
        sequences = create_token_sequences(tokenized_docs, args.seq_length)
        
        # Accumulate tokens
        batch_tokens = []
        for seq in sequences:
            batch_tokens.extend(seq)
        
        all_tokens.extend(batch_tokens)
        total_sequences += len(sequences)
        
        # Write to disk when we've accumulated enough tokens
        if len(all_tokens) >= args.write_interval:
            logger.info(f"Writing {len(all_tokens)} tokens from {file_path} to {output_file}")
            save_tokens_to_file(all_tokens, output_file, args.token_size)
            total_tokens += len(all_tokens)
            all_tokens = []  # Reset the accumulator
    
    # Write any remaining tokens
    if all_tokens:
        output_file_remaining = os.path.join(subset_path, f"tokens_shard_{shard_id:06d}_remainder.ds")
        logger.info(f"Writing remaining {len(all_tokens)} tokens from {file_path} to {output_file_remaining}")
        save_tokens_to_file(all_tokens, output_file_remaining, args.token_size)
        total_tokens += len(all_tokens)
    
    return total_sequences, total_tokens

def main():
    start_time = time.time()
    args = parse_args()
    random.seed(args.seed)
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    for subset in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(args.output_dir, subset), exist_ok=True)
    
    # Load tokenizer globally
    global global_tokenizer
    global_tokenizer = load_tokenizer(args.tokenizer)
    
    # Check if token size matches vocab size
    if args.token_size == 2 and len(global_tokenizer) >= 65536:
        logger.warning(f"Token size (2 bytes) may be too small for vocab size {len(global_tokenizer)}. Consider using --token-size 4")
    
    # Get list of files to process
    file_paths = get_file_paths(args.input_dir, args.extension, args.max_files, args.subset)
    
    # Group files by subset if needed
    if args.subset:
        ordered_files = file_paths
    else:
        # Group by subset
        train_files = [f for f in file_paths if 'train' in f]
        validation_files = [f for f in file_paths if 'validation' in f]
        test_files = [f for f in file_paths if 'test' in f]
        other_files = [f for f in file_paths if not any(subset in f for subset in ['train', 'validation', 'test'])]
        
        logger.info(f"Files by subset: Train={len(train_files)}, Validation={len(validation_files)}, Test={len(test_files)}, Other={len(other_files)}")
        ordered_files = train_files + validation_files + test_files + other_files
    
    # Shuffle files for better distribution
    random.shuffle(ordered_files)
    
    total_sequences = 0
    total_tokens = 0
    
    # Process files in parallel
    logger.info(f"Processing {len(ordered_files)} files using {args.max_workers} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all files for processing
        future_to_file = {
            executor.submit(process_file, file_path, args, i): (i, file_path)
            for i, file_path in enumerate(ordered_files)
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(future_to_file), desc="Processing files"):
            idx, file_path = future_to_file[future]
            try:
                sequences, tokens = future.result()
                total_sequences += sequences
                total_tokens += tokens
                
                # Log progress periodically
                if (idx + 1) % 10 == 0 or (idx + 1) == len(ordered_files):
                    logger.info(f"Processed {idx+1}/{len(ordered_files)} files, {total_sequences} sequences, {total_tokens} tokens")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
    
    # Get statistics for each subset
    subset_stats = {}
    for subset in ['train', 'validation', 'test']:
        subset_dir = os.path.join(args.output_dir, subset)
        if not os.path.exists(subset_dir):
            continue
            
        files = glob.glob(os.path.join(subset_dir, "*.ds"))
        if not files:
            continue
            
        total_size = sum(os.path.getsize(f) for f in files)
        subset_stats[subset] = {
            "files": len(files),
            "total_size_bytes": total_size,
            "total_size_human": f"{total_size / (1024**3):.2f} GB"
        }
    
    # Save metadata
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    metadata = {
        "dataset": os.path.basename(args.input_dir),
        "tokenizer": args.tokenizer,
        "vocab_size": len(global_tokenizer),
        "sequence_length": args.seq_length,
        "token_size_bytes": args.token_size,
        "total_sequences": total_sequences,
        "total_tokens": total_tokens,
        "total_files_processed": len(ordered_files),
        "subset_stats": subset_stats,
        "add_bos": args.add_bos,
        "add_eos": args.add_eos,
        "processing_time_seconds": time.time() - start_time,
        "processing_time_hours": (time.time() - start_time) / 3600,
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Completed processing {len(ordered_files)} files with {total_sequences} sequences and {total_tokens} tokens")
    logger.info(f"Total processing time: {(time.time() - start_time) / 3600:.2f} hours")
    logger.info(f"Output saved to {args.output_dir}")
    logger.info(f"Metadata saved to {metadata_file}")

if __name__ == "__main__":
    main()

import os
import boto3
import subprocess
from pathlib import Path
from typing import Optional, List, Union
import shutil
from glob import glob
import json
import pandas as pd
import io
from .preparer import DatasetPreparer

class HFDatasetPreparer(DatasetPreparer):
    def __init__(self, dataset_name: str, format_type: str = 'auto'):
        super().__init__(dataset_name)
        self.format_type = format_type
        self.hf_token = os.getenv('HF_TOKEN')
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable must be set")
            
    def setup_git_lfs(self, storage_dir: Path) -> None:
        """Configure git and git-lfs for downloading datasets"""
        try:
            # Create git-lfs storage directory
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure git
            subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], check=True)
            subprocess.run(['git', 'config', '--global', 'lfs.storage', str(storage_dir)], check=True)
            
            # Store credentials
            credentials_dir = Path.home() / '.git'
            credentials_dir.mkdir(parents=True, exist_ok=True)
            credentials_file = credentials_dir / 'credentials'
            credentials_file.write_text(f"https://{self.hf_token}@huggingface.co\n")
            credentials_file.chmod(0o600)
            
            # Initialize git-lfs
            subprocess.run(['git', 'lfs', 'install'], check=True)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to setup git-lfs: {e}")
            
    def check_dataset_access(self, dataset_name: str) -> bool:
        """Test if dataset is accessible with current credentials"""
        try:
            result = subprocess.run(
                ['git', 'ls-remote', f'https://{self.hf_token}@huggingface.co/datasets/{dataset_name}'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    def is_dataset_already_downloaded(self, download_dir: Path) -> bool:
        """Check if the dataset has already been downloaded and is valid"""
        if not download_dir.exists():
            print(f"Download directory {download_dir} does not exist")
            return False
            
        # Check if there are any dataset files present
        files = self.collect_files(download_dir)
        if not files:
            print(f"No dataset files found in {download_dir}")
            return False
            
        # If we have dataset files, consider it downloaded
        # Don't require the .git directory as it may have been downloaded via other means
        print(f"Dataset already downloaded at {download_dir} ({len(files)} files found)")
        return True
            
    def download_dataset(self, dataset_name: str, download_dir: Path) -> Path:
        """Download dataset from Hugging Face using git-lfs if not already downloaded"""
        # Check if dataset is already downloaded
        if self.is_dataset_already_downloaded(download_dir):
            print(f"Using existing dataset at {download_dir}")
            return download_dir
            
        print(f"Downloading dataset {dataset_name} to {download_dir}")
        
        # Clean any existing incomplete download
        if download_dir.exists():
            shutil.rmtree(download_dir)
            
        # Setup git-lfs
        lfs_storage = download_dir.parent / 'git-lfs-storage'
        self.setup_git_lfs(lfs_storage)
        
        # Check dataset accessibility
        if not self.check_dataset_access(dataset_name):
            raise ValueError(f"Cannot access dataset {dataset_name}. Check if it exists and you have proper permissions.")
            
        try:
            # Clone the repository
            subprocess.run([
                'git', 'clone',
                f'https://oauth2:{self.hf_token}@huggingface.co/datasets/{dataset_name}',
                str(download_dir)
            ], check=True, env={**os.environ, 'GIT_LFS_SKIP_SMUDGE': '0', 'GIT_TERMINAL_PROMPT': '0'})
            
            return download_dir
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
            
    def collect_files(self, full_source_path: Path) -> List[str]:
        """Collect all relevant files from the downloaded dataset"""
        if not isinstance(full_source_path, Path):
            full_source_path = Path(full_source_path)
            
        if not full_source_path.exists():
            print(f"Warning: Path {full_source_path} does not exist")
            return []
            
        # Check if this is a parent directory containing the actual dataset directory
        # This handles cases where the dataset is in a subdirectory like 'data/raw/SlimPajama-627B/'
        subdirs = [d for d in full_source_path.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and subdirs[0].name == self.dataset_name.split('-')[-1]:
            print(f"Found dataset in subdirectory: {subdirs[0]}")
            full_source_path = subdirs[0]
            
        # Also check for common directory structure with train/test/validation subdirs
        dataset_subdirs = ['train', 'test', 'validation', 'val']
        has_data_subdirs = any([(full_source_path / subdir).exists() for subdir in dataset_subdirs])
        
        if has_data_subdirs:
            print(f"Found dataset with train/test/validation structure")
            all_files = []
            # Search in each data subdir
            for subdir in dataset_subdirs:
                subdir_path = full_source_path / subdir
                if not subdir_path.exists():
                    continue
                    
                # Look for files with common extensions plus .zst compressed files
                if self.format_type == 'auto':
                    extensions = ['.txt', '.json', '.csv', '.parquet', '.jsonl', '.zst', '.jsonl.zst']
                    for ext in extensions:
                        files = glob(f'{subdir_path}/**/*{ext}', recursive=True)
                        if files:
                            print(f"Found {len(files)} {ext} files in {subdir}")
                            all_files.extend(files)
                else:
                    # Use specified format
                    files = glob(f'{subdir_path}/**/*.{self.format_type}', recursive=True)
                    if files:
                        print(f"Found {len(files)} {self.format_type} files in {subdir}")
                        all_files.extend(files)
                        
            return all_files
            
        # Default search for files in the current directory
        if self.format_type == 'auto':
            # Try to detect format from files
            extensions = ['.txt', '.json', '.csv', '.parquet', '.jsonl', '.zst', '.jsonl.zst']
            all_files = []
            for ext in extensions:
                files = glob(f'{full_source_path}/**/*{ext}', recursive=True)
                if files:
                    print(f"Found {len(files)} {ext} files")
                    all_files.extend(files)
            return all_files
        else:
            # Use specified format
            files = glob(f'{full_source_path}/**/*.{self.format_type}', recursive=True)
            print(f"Found {len(files)} {self.format_type} files")
            return files

    def read_file_contents(self, filepath: Union[str, Path]) -> List[str]:
        """Handle different file formats"""
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        suffix = filepath.suffix.lower()
        full_name = filepath.name.lower()
        
        try:
            # Special handling for .zst compressed files
            if suffix == '.zst' or full_name.endswith('.jsonl.zst'):
                print(f"Reading zstandard compressed file: {filepath}")
                try:
                    import zstandard as zstd
                    
                    with open(filepath, 'rb') as fh:
                        dctx = zstd.ZstdDecompressor()
                        with dctx.stream_reader(fh) as reader:
                            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                            
                            # Handle jsonl.zst files
                            if full_name.endswith('.jsonl.zst'):
                                texts = []
                                for line in text_stream:
                                    line = line.strip()
                                    if line:
                                        try:
                                            item = json.loads(line)
                                            if isinstance(item, str):
                                                texts.append(item)
                                            elif isinstance(item, dict):
                                                # Look for text fields with various common names
                                                for field in ['text', 'content', 'document', 'data']:
                                                    if field in item:
                                                        texts.append(item[field])
                                                        break
                                        except json.JSONDecodeError as e:
                                            print(f"Warning: Could not parse JSON line in {filepath}: {e}")
                                
                                print(f"Extracted {len(texts)} text entries from {filepath}")
                                if not texts:
                                    # If no items found with standard fields, print sample content for debugging
                                    print("DEBUG: No text fields found in JSONL. Printing first item structure:")
                                    fh.seek(0)
                                    with dctx.stream_reader(fh) as reader2:
                                        text_stream2 = io.TextIOWrapper(reader2, encoding='utf-8')
                                        try:
                                            first_line = next(text_stream2).strip()
                                            if first_line:
                                                sample = json.loads(first_line)
                                                print(f"Keys in first item: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
                                                if isinstance(sample, dict):
                                                    # Take the first field as fallback
                                                    first_key = next(iter(sample))
                                                    if isinstance(sample[first_key], str):
                                                        texts = [sample[first_key]]
                                                        print(f"Using '{first_key}' field as fallback")
                                        except Exception as e:
                                            print(f"Error examining file contents: {e}")
                                
                                return texts
                            
                except ImportError:
                    print("Warning: zstandard module not found. Install with 'pip install zstandard' to handle .zst files")
                except Exception as e:
                    print(f"Error decompressing zst file: {e}")
                    
                # Try fallback method using subprocess for zstd
                try:
                    print("Trying fallback decompression with zstd command line tool...")
                    import subprocess
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as temp_file:
                        temp_path = temp_file.name
                    
                    # Decompress using system zstd command
                    result = subprocess.run(
                        ['zstd', '-d', str(filepath), '-o', temp_path], 
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        texts = []
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        item = json.loads(line)
                                        if isinstance(item, str):
                                            texts.append(item)
                                        elif isinstance(item, dict) and 'text' in item:
                                            texts.append(item['text'])
                                    except json.JSONDecodeError:
                                        pass
                        
                        # Clean up temporary file
                        os.unlink(temp_path)
                        return texts
                    else:
                        print(f"zstd command failed: {result.stderr}")
                        os.unlink(temp_path)
                
                except Exception as e:
                    print(f"Fallback method failed: {e}")
                    
                return []
                
            if suffix == '.txt':
                with open(filepath, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
                    
            elif suffix == '.parquet':
                df = pd.read_parquet(filepath)
                for col in ['text', 'content', 'story', 'document', 'string']:
                    if col in df.columns:
                        return df[col].dropna().tolist()
                print(f"Available columns in parquet file: {df.columns.tolist()}")
                raise ValueError(f"No text column found in parquet file")
                
            elif suffix == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        if all(isinstance(x, str) for x in data):
                            return data
                        elif all(isinstance(x, dict) for x in data):
                            return [x['text'] for x in data if 'text' in x]
                    raise ValueError("JSON must contain array of strings or objects with 'text' field")
                    
            elif suffix == '.jsonl':
                texts = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            if isinstance(item, str):
                                texts.append(item)
                            elif isinstance(item, dict) and 'text' in item:
                                texts.append(item['text'])
                return texts
                    
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
                    
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")
            print(f"File exists: {filepath.exists()}")
            print(f"File size: {filepath.stat().st_size if filepath.exists() else 'N/A'}")
            return []

def upload_to_s3(local_dir: Path, bucket: str, prefix: str = '', 
               max_workers: int = 200, chunk_size: int = 3000) -> None:
    """
    Upload prepared dataset to S3 with parallel processing for performance.
    
    Parameters:
    -----------
    local_dir : Path
        Path to the local directory containing files to upload
    bucket : str
        Name of the S3 bucket
    prefix : str, optional
        Prefix (folder path) within the S3 bucket
    max_workers : int, optional
        Maximum number of worker threads (default 200)
    chunk_size : int, optional
        Number of files to process in a single chunk (default 3000)
    """
    import boto3
    import concurrent.futures
    import time
    from pathlib import Path
    from tqdm import tqdm
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='s3_upload.log'
    )
    
    # Initialize the S3 client with optimized settings
    s3_config = boto3.session.Config(
        max_pool_connections=max_workers * 2,  # Double the connections
        connect_timeout=5,  # Shorter connection timeout
        read_timeout=60,    # Longer read timeout for stability
        retries={'max_attempts': 10, 'mode': 'adaptive'}  # Enhanced retry
    )
    
    session = boto3.session.Session()
    s3_client = session.client('s3', config=s3_config)
    
    # Get all files upfront
    print(f"Scanning files in {local_dir}...")
    all_files = list(local_dir.rglob('*'))
    all_files = [f for f in all_files if f.is_file()]
    total_files = len(all_files)
    print(f"Found {total_files} files to upload")
    
    # Function to upload a single file
    def upload_file_to_s3(file_path):
        try:
            # Create S3 key maintaining directory structure
            relative_path = file_path.relative_to(local_dir)
            s3_key = f"{prefix}/{relative_path}" if prefix else str(relative_path)
            s3_client.upload_file(str(file_path), bucket, s3_key)
            return True
        except Exception as e:
            logging.error(f"Error uploading {file_path}: {e}")
            return False
    
    # Process files in chunks to manage memory usage
    successful_uploads = 0
    failed_uploads = 0
    
    start_time = time.time()
    
    # Create chunks to prevent memory issues with very large directories
    chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    
    for chunk_num, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_num+1}/{len(chunks)} ({len(chunk)} files)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Create a progress bar for this chunk
            with tqdm(total=len(chunk), desc=f"Chunk {chunk_num+1}", unit="file") as pbar:
                # Submit all upload tasks
                for file_path in chunk:
                    future = executor.submit(upload_file_to_s3, file_path)
                    futures.append(future)
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        successful_uploads += 1
                    else:
                        failed_uploads += 1
                    pbar.update(1)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nUpload complete!")
    print(f"Total files: {total_files}")
    print(f"Successful uploads: {successful_uploads}")
    print(f"Failed uploads: {failed_uploads} (see log for details)")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Average upload rate: {successful_uploads / duration:.2f} files/second")

def main(
    hf_dataset_name: str,
    source_path: Optional[Path] = None,
    format_type: str = 'auto',
    tokenizer_path: Optional[Path] = None,
    destination_path: Optional[Path] = None,
    chunk_size: int = 2048,
    percentage: float = 1.0,
    train_val_split_ratio: float = 0.9,
    max_cores: Optional[int] = None,
    dataset_name: str = "custom-dataset",
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    s3_max_workers: int = 200, # change to a higher value if possible 
    s3_chunk_size: int = 3000, # change to a higher value if possible 
    force_download: bool = False,
    skip_download: bool = False,
) -> None:
    # Extract the dataset name for path handling
    dataset_short_name = hf_dataset_name.split('/')[-1] if '/' in hf_dataset_name else hf_dataset_name
    
    # Create dataset name from HF dataset name
    safe_dataset_name = f"custom-{hf_dataset_name.replace('/', '-')}"
    
    # Initialize preparer with the safe dataset name
    preparer = HFDatasetPreparer(safe_dataset_name, format_type)
    
    # If source_path not provided, create default directory
    if not source_path:
        source_path = Path('data/raw')
    
    # Convert to Path if it's a string
    if isinstance(source_path, str):
        source_path = Path(source_path)
    
    # Handle the case where source_path points to the parent directory
    dataset_dir = source_path / dataset_short_name
    if dataset_dir.exists() and dataset_dir.is_dir():
        print(f"Found dataset directory at {dataset_dir}")
        source_path = dataset_dir
    
    # Handle download options
    if force_download and skip_download:
        raise ValueError("Cannot specify both force_download and skip_download")
        
    # Option to explicitly skip download
    if skip_download:
        print(f"Skipping download as requested. Using existing dataset at {source_path}")
        if not source_path.exists():
            raise ValueError(f"Source path {source_path} does not exist and skip_download is True")
    elif force_download:
        print(f"Forcing download as requested, even if dataset already exists")
        # Clean any existing download if forcing
        if source_path.exists():
            print(f"Removing existing dataset at {source_path}")
            shutil.rmtree(source_path)
        source_path = preparer.download_dataset(hf_dataset_name, source_path)
    else:
        # Normal case: download only if needed
        source_path = preparer.download_dataset(hf_dataset_name, source_path)
    
    print(f"Preparing dataset from {source_path}")
    
    # Prepare the dataset
    preparer.prepare(
        source_path,
        tokenizer_path,
        destination_path,
        chunk_size,
        percentage,
        train_val_split_ratio,
        max_cores,
    )
    
    # upload to s3 bucket
    if s3_bucket:
        upload_to_s3(destination_path, s3_bucket, s3_prefix, 
                    max_workers=s3_max_workers, chunk_size=s3_chunk_size)

if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)

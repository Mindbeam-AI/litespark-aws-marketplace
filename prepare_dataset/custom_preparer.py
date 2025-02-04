import os
import boto3
import subprocess
from pathlib import Path
from typing import Optional, List, Union
import shutil
from glob import glob
import json
import pandas as pd
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
            
    def download_dataset(self, dataset_name: str, download_dir: Path) -> Path:
        """Download dataset from Hugging Face using git-lfs"""
        # Clean any existing download
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
        if self.format_type == 'auto':
            # Try to detect format from files
            extensions = ['.txt', '.json', '.csv', '.parquet', '.jsonl']
            all_files = []
            for ext in extensions:
                files = glob(f'{full_source_path}/**/*{ext}', recursive=True)
                if files:
                    print(f"Found {len(files)} {ext} files")
                    all_files.extend(files)
            return all_files
        else:
            # Use specified format
            return glob(f'{full_source_path}/**/*.{self.format_type}', recursive=True)

    def read_file_contents(self, filepath: Union[str, Path]) -> List[str]:
        """Handle different file formats"""
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        suffix = filepath.suffix.lower()
        
        try:
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

def upload_to_s3(local_dir: Path, bucket: str, prefix: str = '') -> None:
    """Upload prepared dataset to S3"""
    s3 = boto3.client('s3')
    
    # Upload each file in the directory
    for path in local_dir.rglob('*'):
        if path.is_file():
            # Create S3 key maintaining directory structure
            relative_path = path.relative_to(local_dir)
            s3_key = f"{prefix}/{relative_path}" if prefix else str(relative_path)
            s3.upload_file(str(path), bucket, s3_key)
    print(f"Successfully uploaded {total_files} files to s3://{bucket}/{prefix if prefix else ''}")

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
) -> None:
    # Create dataset name from HF dataset name
    safe_dataset_name = f"custom-{hf_dataset_name.replace('/', '-')}"
    
    # Initialize preparer with the safe dataset name
    preparer = HFDatasetPreparer(safe_dataset_name, format_type)
    
    # If source_path not provided, create temporary directory
    if not source_path:
        source_path = Path('data/raw') / hf_dataset_name.split('/')[-1]
    
    # Download the dataset
    source_path = preparer.download_dataset(hf_dataset_name, source_path)
    
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
        upload_to_s3(destination_path, s3_bucket, s3_prefix)

if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)

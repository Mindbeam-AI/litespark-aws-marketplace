'''
Usage:
python upload_to_s3.py --folder ./train --bucket my-bucket --prefix data/SlimPajama-6B/train --workers 200 --chunk-size 1000 --start-chunk 1816
'''

import os
import boto3
import time
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import argparse

def test_credentials():
    """Test if the current credentials are valid"""
    try:
        session = boto3.Session()
        s3 = session.client('s3')
        s3.list_buckets()
        print("✅ Credentials validated successfully!")
        return True
    except Exception as e:
        logging.error(f"Credential validation failed: {e}")
        print(f"❌ Credential validation failed: {e}")
        return False

def setup_credentials_from_env():
    """Setup credentials file from environment variables"""
    try:
        # Create ~/.aws directory if it doesn't exist
        aws_dir = Path.home() / '.aws'
        aws_dir.mkdir(exist_ok=True)
        
        # Create credentials file
        credentials_file = aws_dir / 'credentials'
        with open(credentials_file, 'w') as f:
            f.write("[default]\n")
            if 'AWS_ACCESS_KEY_ID' in os.environ:
                f.write(f"aws_access_key_id = {os.environ['AWS_ACCESS_KEY_ID']}\n")
            if 'AWS_SECRET_ACCESS_KEY' in os.environ:
                f.write(f"aws_secret_access_key = {os.environ['AWS_SECRET_ACCESS_KEY']}\n")
            if 'AWS_SESSION_TOKEN' in os.environ:
                f.write(f"aws_session_token = {os.environ['AWS_SESSION_TOKEN']}\n")
        
        # Create config file
        config_file = aws_dir / 'config'
        with open(config_file, 'w') as f:
            f.write("[default]\n")
            region = os.environ.get('AWS_REGION', 'us-east-1')
            f.write(f"region = {region}\n")
        
        # Set permissions
        credentials_file.chmod(0o600)
        config_file.chmod(0o600)
        
        print(f"AWS credentials file created at {credentials_file}")
        return True
    except Exception as e:
        print(f"Error creating credentials file: {e}")
        return False

def upload_file_to_s3(s3_client, file_path, bucket, s3_key):
    """Upload a single file to S3"""
    try:
        s3_client.upload_file(str(file_path), bucket, s3_key)
        return True, s3_key
    except Exception as e:
        error_message = str(e)
        if "ExpiredToken" in error_message or "InvalidAccessKeyId" in error_message:
            raise CredentialsExpiredError("AWS credentials have expired")
        logging.error(f"Error uploading {file_path}: {e}")
        return False, s3_key

class CredentialsExpiredError(Exception):
    """Custom exception for expired credentials"""
    pass

def upload_folder_to_s3(local_dir, bucket, prefix='', max_workers=200, chunk_size=1000, 
                       start_chunk=0):
    """
    Upload a folder to S3 with resumable progress
    
    Parameters:
    -----------
    local_dir : str or Path
        Path to the local directory containing files to upload
    bucket : str
        Name of the S3 bucket
    prefix : str, optional
        Prefix (folder path) within the S3 bucket
    max_workers : int, optional
        Maximum number of worker threads (default 200)
    chunk_size : int, optional
        Number of files to process in a single chunk (default 1000)
    start_chunk : int, optional
        Chunk number to start from (0-based)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('s3_upload_resume.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Convert to Path object if string
    if isinstance(local_dir, str):
        local_dir = Path(local_dir)
    
    # Get all files upfront
    logging.info(f"Scanning files in {local_dir}...")
    all_files = list(local_dir.rglob('*'))
    all_files = [f for f in all_files if f.is_file()]
    total_files = len(all_files)
    logging.info(f"Found {total_files} files to upload")
    
    # Create chunks to prevent memory issues
    chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    total_chunks = len(chunks)
    
    # Initialize counters
    successful_uploads = 0
    failed_uploads = 0
    processed_chunks = start_chunk
    
    # Validate initial credentials
    valid_credentials = test_credentials()
    if not valid_credentials:
        print("\nINSTRUCTIONS FOR SETTING UP CREDENTIALS:")
        print("----------------------------------------")
        print("1. Open a new terminal window")
        print("2. Export your AWS credentials with these commands:")
        print("   export AWS_ACCESS_KEY_ID=your_access_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("   export AWS_SESSION_TOKEN=your_session_token  # if using temporary credentials")
        print("3. Then run this script again\n")
        
        input("Press ENTER once you've set the environment variables in another terminal...")
        
        # Test if the credentials work
        valid_credentials = test_credentials()
        if not valid_credentials:
            logging.error("Could not obtain valid credentials. Exiting.")
            return False
    
    # Setup AWS configuration
    setup_credentials_from_env()
    
    s3_config = boto3.session.Config(
        max_pool_connections=max_workers * 2,
        connect_timeout=5,
        read_timeout=60,
        retries={'max_attempts': 10, 'mode': 'adaptive'}
    )
    
    start_time = time.time()
    last_progress_save = time.time()
    
    # Create progress file to track what's been processed
    progress_file = Path('upload_progress.txt')
    
    # Process each chunk
    while processed_chunks < total_chunks:
        try:
            # Create a fresh S3 client for this chunk
            session = boto3.Session()
            s3_client = session.client('s3', config=s3_config)
            
            # Get current chunk
            chunk = chunks[processed_chunks]
            chunk_start_time = time.time()
            
            # Log progress
            logging.info(f"Processing chunk {processed_chunks+1}/{total_chunks} ({len(chunk)} files)")
            logging.info(f"Progress: {processed_chunks * chunk_size}/{total_files} files processed ({processed_chunks * 100.0 / total_chunks:.2f}%)")
            
            # Process the chunk with parallel uploads
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                # Submit all upload tasks for this chunk
                for file_path in chunk:
                    # Calculate the S3 key (path within the bucket)
                    relative_path = file_path.relative_to(local_dir)
                    s3_key = f"{prefix}/{relative_path}" if prefix else str(relative_path)
                    
                    # Fix path separator for S3
                    s3_key = s3_key.replace("\\", "/")
                    
                    # Submit the upload task
                    future = executor.submit(upload_file_to_s3, s3_client, file_path, bucket, s3_key)
                    futures.append(future)
                
                # Process results with progress bar
                with tqdm(total=len(chunk), desc=f"Chunk {processed_chunks+1}/{total_chunks}", unit="file") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            success, s3_key = future.result()
                            if success:
                                successful_uploads += 1
                            else:
                                failed_uploads += 1
                        except CredentialsExpiredError:
                            # If credentials expired, break out of the loop to refresh them
                            logging.warning("Credentials expired during upload. Pausing for refresh.")
                            # Mark all incomplete futures as cancelled
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            # Don't increment processed_chunks so we retry this chunk
                            raise CredentialsExpiredError("Credentials expired during chunk processing")
                        finally:
                            pbar.update(1)
            
            # If we get here, the chunk was processed successfully
            chunk_duration = time.time() - chunk_start_time
            logging.info(f"Chunk {processed_chunks+1} completed in {chunk_duration:.2f} seconds ({len(chunk)/chunk_duration:.2f} files/sec)")
            processed_chunks += 1
            
            # Save progress periodically
            if time.time() - last_progress_save > 60:  # Save every minute
                with open(progress_file, 'w') as f:
                    f.write(str(processed_chunks))
                last_progress_save = time.time()
            
        except CredentialsExpiredError:
            # Refresh credentials and retry the current chunk
            logging.info("AWS credentials have expired. Let's set up new credentials.")
            print("\n==============================================================")
            print("   CREDENTIALS EXPIRED - PLEASE FOLLOW THESE INSTRUCTIONS")
            print("==============================================================")
            print("1. Open a NEW terminal window")
            print("2. Get new credentials from AWS console or CLI")
            print("3. Run these commands, replacing values with your new credentials:")
            print("   export AWS_ACCESS_KEY_ID=your_access_key")
            print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
            print("   export AWS_SESSION_TOKEN=your_session_token")
            print("4. Return to THIS terminal window")
            
            input("\nPress ENTER once you've set the environment variables in another terminal...")
            
            # Test if the new credentials work
            valid_credentials = test_credentials()
            if not valid_credentials:
                retry = input("Credentials validation failed. Try again? (y/n): ")
                if retry.lower() != 'y':
                    logging.error("Upload aborted due to credential issues.")
                    return False
                continue
            
            # Update AWS configuration files
            setup_credentials_from_env()
            
        except Exception as e:
            # For other errors, log and continue to next chunk
            logging.error(f"Error processing chunk {processed_chunks+1}: {e}")
            processed_chunks += 1
            
    # Done with all chunks
    total_duration = time.time() - start_time
    
    logging.info("\nUpload process complete!")
    logging.info(f"Total files: {total_files}")
    logging.info(f"Files uploaded successfully: {successful_uploads}")
    logging.info(f"Failed uploads: {failed_uploads}")
    logging.info(f"Total time: {total_duration:.2f} seconds")
    if successful_uploads > 0:
        logging.info(f"Average upload rate: {successful_uploads / total_duration:.2f} files/second")
    
    # Save final progress
    with open(progress_file, 'w') as f:
        f.write(str(processed_chunks))
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload a folder to S3 with resume capability')
    parser.add_argument('--folder', required=True, help='Local folder path to upload')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', default='', help='S3 prefix (folder path within bucket)')
    parser.add_argument('--workers', type=int, default=200, help='Maximum number of worker threads')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Files to process in a single chunk')
    parser.add_argument('--start-chunk', type=int, default=0, help='Chunk number to start from (0-based)')
    
    args = parser.parse_args()
    
    # Upload the folder
    upload_folder_to_s3(
        args.folder, 
        args.bucket, 
        args.prefix,
        max_workers=args.workers,
        chunk_size=args.chunk_size,
        start_chunk=args.start_chunk
    )


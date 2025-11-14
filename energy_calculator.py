'''
Calculate energy in MWh (or kWh) integrating gpu power consumption from W&B dashboard over training period. 
Can be used to calculate projected energy consumption for target tokens from shorter-duration runs.

Usage example:
---------------
python energy_calculator.py --url ... --nodes 2 --seq-length 4096 --global-batch-size 256 --target-tokens 500e9
'''

import wandb
import numpy as np
import pandas as pd
import argparse
import re
from datetime import datetime
import sys

def parse_wandb_url(url):
    """Parse a WandB URL to extract entity, project, and run_id"""
    # Pattern for URLs like: https://wandb.ai/entity/project/runs/run_id
    pattern = r'https://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?]+)'
    match = re.search(pattern, url)
    
    if not match:
        raise ValueError(f"Invalid WandB URL format: {url}")
    
    entity, project, run_id = match.groups()
    return entity, project, run_id

def detect_and_exclude_postprocessing(power_kw, runtime_hours, threshold_percentile=40, 
                                     min_duration_minutes=3, look_back_points=20):
    """
    Detect and exclude post-processing (e.g. checkpoint saving) period based on power drop
    
    Args:
        power_kw: Array of power measurements in kW (zeros already filtered out)
        runtime_hours: Array of runtime in hours
        threshold_percentile: Percentile below which power is considered "low" (default 40)
        min_duration_minutes: Minimum duration to consider as post-processing (default 10)
        look_back_points: Number of points to look back for transition detection (default 20)
    
    Returns:
        Tuple of (filtered_power_kw, filtered_runtime_hours, excluded_indices)
    """
    
    if len(power_kw) < 30:
        print("Not enough data points to detect post-processing (< 30)")
        return power_kw, runtime_hours, np.array([])
    
    # Calculate threshold based on the percentile
    power_threshold = np.percentile(power_kw, threshold_percentile)
    
    # Apply light smoothing to reduce noise
    window_size = min(5, len(power_kw) // 10)
    if window_size >= 3:
        power_smooth = pd.Series(power_kw).rolling(window=window_size, center=True).mean().fillna(power_kw).values
    else:
        power_smooth = power_kw
    
    # Look at the tail to see if it's consistently low
    tail_length = min(max(look_back_points, 20), len(power_smooth) // 3)
    tail_power = power_smooth[-tail_length:]
    tail_avg = tail_power.mean()
    
    # Check if tail is significantly lower than the threshold
    if tail_avg > power_threshold:
        return power_kw, runtime_hours, np.array([])
    
    # Find where power drops and stays low
    # Work backwards to find last high-power point
    cutoff_idx = None
    consecutive_low = 0
    min_consecutive = max(5, tail_length // 4)
    
    for i in range(len(power_smooth) - 1, max(look_back_points, 10), -1):
        if power_smooth[i] < power_threshold:
            consecutive_low += 1
        else:
            if consecutive_low >= min_consecutive:
                cutoff_idx = i + 1
                break
            consecutive_low = 0
    
    if cutoff_idx is None and consecutive_low >= min_consecutive:
        cutoff_idx = max(look_back_points, 10)
    
    if cutoff_idx is None:
        print(f"No sustained low-power period found")
        return power_kw, runtime_hours, np.array([])
    
    # Check duration
    if cutoff_idx >= len(runtime_hours) - 1:
        print("Cutoff at end - nothing to exclude")
        return power_kw, runtime_hours, np.array([])
    
    excluded_duration_hours = runtime_hours[-1] - runtime_hours[cutoff_idx]
    excluded_duration_minutes = excluded_duration_hours * 60
    
    if excluded_duration_minutes < min_duration_minutes:
        print(f"Excluded period ({excluded_duration_minutes:.1f} min) shorter than minimum ({min_duration_minutes} min)")
        return power_kw, runtime_hours, np.array([])
    
    # Calculate statistics
    training_power_avg = power_kw[:cutoff_idx].mean() * 1000
    postproc_power_avg = power_kw[cutoff_idx:].mean() * 1000
    power_drop_pct = ((training_power_avg - postproc_power_avg) / training_power_avg) * 100
    
    # Calculate energy excluded
    excluded_energy = np.trapz(power_kw[cutoff_idx:], runtime_hours[cutoff_idx:])
    
    # Time when post-processing starts
    postproc_start_time_hours = runtime_hours[cutoff_idx]
    postproc_start_time_minutes = postproc_start_time_hours * 60
    
    return power_kw[:cutoff_idx], runtime_hours[:cutoff_idx], np.arange(cutoff_idx, len(power_kw))

def calculate_energy(source, nodes=4, target_steps=None, target_tokens=None,
                     seq_length_override=None, global_batch_size_override=None,
                     is_local_file=False):
    """
    Calculate energy consumption from WandB run data
    
    Args:
        source: Either a WandB URL or path to local .wandb file
        nodes: Number of compute nodes in the training cluster
        target_steps: Target number of global steps for prediction (None = no prediction)
        target_tokens: Target number of tokens for prediction (None = no prediction)
        is_local_file: If True, treat source as a local file path
    
    Returns:
        Dictionary with energy consumption metrics
    """
    
    if is_local_file:
        print(f"Loading from local file: {source}")
        
        if source.endswith('.wandb'):
            try:
                system_metrics = load_wandb_file(source)
                run_id = os.path.basename(source).replace('.wandb', '')
                project_path = "local-offline-run"
                run_config = {}  # Config not easily available from offline files
                main_history = system_metrics  # For local files, use same data
            except NotImplementedError as e:
                print(f"\n{e}")
                print("\nTo sync your offline run to W&B cloud:")
                print(f"  1. Run: wandb sync {source}")
                print(f"  2. Copy the URL from the output")
                print(f"  3. Use: python {__file__} --url <copied-url> --nodes {nodes}")
                return None
        else:
            raise ValueError(f"File must be a .wandb file, got: {source}")
        
    else:
        # Parse the URL to get project details
        entity, project, run_id = parse_wandb_url(source)
        project_path = f"{entity}/{project}"
        
        print(f"Analyzing run: {run_id} from project: {project_path}")
        
        # Initialize the API
        api = wandb.Api()
        
        # Access the specific run
        run = api.run(f"{project_path}/{run_id}")
        
        # Get run config (hyperparameters)
        run_config = run.config
        
        try:
            # Try to access system metrics first
            system_metrics = run.history(stream="system")
            
            # Also get main history for step metrics
            main_history = run.history()
            
        except Exception as e:
            print(f"Error accessing metrics: {e}")
            system_metrics = run.history()
            main_history = system_metrics
    
    try:
        # Check for GPU power metrics in different possible formats
        possible_power_formats = [
            "system.gpu.{}.powerWatts",
            "gpu.{}.power", 
            "system.gpu.{}.power",
            "gpu_{}_power_watts",
            "system.gpu.{}.powerPercent",
            "gpu.{}.powerWatts"
        ]
        
        existing_power_cols = []
        gpus_per_node = 0
        
        # Try each format until we find one that works
        for format_pattern in possible_power_formats:
            test_cols = [format_pattern.format(i) for i in range(16)]
            found_cols = [col for col in test_cols if col in system_metrics.columns]
            
            if found_cols:
                existing_power_cols = found_cols
                gpus_per_node = len(found_cols)
                break
        
        if not existing_power_cols:
            gpu_cols = [col for col in system_metrics.columns if 'gpu' in col.lower()]
            raise ValueError("No GPU power metrics found in any expected format")
        
        # Sum power across all GPUs and convert to kW
        power_watts = system_metrics[existing_power_cols].sum(axis=1)
        power_kw = power_watts / 1000
        
        # Get runtime in hours
        if '_runtime' in system_metrics.columns:
            runtime_hours = system_metrics['_runtime'].values / 3600
        elif '_timestamp' in system_metrics.columns:
            timestamps = system_metrics['_timestamp'].values
            runtime_hours = (timestamps - timestamps[0]) / 3600
        else:
            runtime_hours = np.arange(len(power_kw)) / 60
        
        # Remove any NaN values
        valid_indices = ~(np.isnan(power_kw) | np.isnan(runtime_hours))
        power_kw = power_kw[valid_indices]
        runtime_hours = runtime_hours[valid_indices]
        
        # NEW: Remove zero power values (spurious W&B logging artifacts)
        # Keep only non-zero power measurements
        nonzero_mask = power_kw > 0.01  # Small threshold to avoid floating point issues
        power_kw = power_kw[nonzero_mask]
        runtime_hours = runtime_hours[nonzero_mask]
        
        if len(power_kw) == 0:
            raise ValueError("No valid power data points found after filtering")
        
        if len(power_kw) == 0:
            raise ValueError("No valid power data points found")
        
        # Detect and exclude post-processing
        power_kw_original = power_kw.copy()
        runtime_hours_original = runtime_hours.copy()
        
        power_kw, runtime_hours, excluded_indices = detect_and_exclude_postprocessing(
            power_kw, runtime_hours, 
            threshold_percentile=40,
            min_duration_minutes=3
        )
        
        # Calculate total kWh using trapezoidal integration
        total_kwh_per_node = np.trapz(power_kw, runtime_hours)
        total_kwh_cluster = total_kwh_per_node * nodes
        
        # Calculate duration and GPU hours
        duration_hours = runtime_hours.max() - runtime_hours.min()
        total_gpu_hours = gpus_per_node * nodes * duration_hours
        
        # Get current step count - try main_history first, then system_metrics
        current_steps = None
        step_metrics = ['iteration_step', 'train/global_step', 'global_step', 'step', 'train_step', 'trainer/global_step', '_step']
        
        # First try main_history (where steps usually are)
        if not is_local_file:
            for step_metric in step_metrics:
                if step_metric in main_history.columns:
                    current_steps = main_history[step_metric].max()
                    if not pd.isna(current_steps):
                        break
                    else:
                        current_steps = None
        
        # Fall back to system_metrics if not found
        if current_steps is None:
            for step_metric in step_metrics:
                if step_metric in system_metrics.columns:
                    current_steps = system_metrics[step_metric].max()
                    if not pd.isna(current_steps):
                        break
                    else:
                        current_steps = None
        
        if current_steps is None:
            print("Warning: Could not find step count in metrics")
        
        # Try to extract training config parameters
        seq_length = None
        global_batch_size = None
        num_nodes_from_config = None
        
        if not is_local_file and run_config:
            # Try different possible config key names
            seq_length_keys = ['sequence_length', 'seq_length', 'max_seq_length', 'seq_len', 'SEQ_LEN']
            batch_size_keys = ['global_batch_size', 'batch_size', 'total_batch_size', 'GLOBAL_BATCH_SIZE']
            num_nodes_keys = ['NUM_NODES', 'num_nodes', 'n_nodes', 'world_size']
            
            for key in seq_length_keys:
                if key in run_config:
                    seq_length = int(run_config[key])
                    print(f"Found sequence length in config '{key}': {seq_length}")
                    break
            
            for key in batch_size_keys:
                if key in run_config:
                    global_batch_size = int(run_config[key])
                    print(f"Found global batch size in config '{key}': {global_batch_size}")
                    break
            
            # Try to get number of nodes from config
            for key in num_nodes_keys:
                if key in run_config:
                    num_nodes_from_config = int(run_config[key])
                    print(f"Found number of nodes in config '{key}': {num_nodes_from_config}")
                    break
        
        # Override with command-line values if provided
        if seq_length_override is not None:
            if seq_length is not None and seq_length != seq_length_override:
                print(f"WARNING: Command-line seq-length ({seq_length_override}) differs from config ({seq_length})")
                print(f"Using command-line value: {seq_length_override}")
            else:
                print(f"Using command-line sequence length: {seq_length_override}")
            seq_length = seq_length_override
        
        if global_batch_size_override is not None:
            if global_batch_size is not None and global_batch_size != global_batch_size_override:
                print(f"WARNING: Command-line global-batch-size ({global_batch_size_override}) differs from config ({global_batch_size})")
                print(f"Using command-line value: {global_batch_size_override}")
            else:
                print(f"Using command-line global batch size: {global_batch_size_override}")
            global_batch_size = global_batch_size_override
        
        # Use config value if available
        if num_nodes_from_config is not None:
            if nodes != num_nodes_from_config:
                print(f"WARNING: Command-line nodes ({nodes}) differs from config NUM_NODES ({num_nodes_from_config})")
                print(f"Using config value: {num_nodes_from_config} nodes")
            nodes = num_nodes_from_config
        
        # Calculate total tokens processed
        current_tokens = None
        if current_steps and seq_length and global_batch_size:
            current_tokens = int(current_steps) * seq_length * global_batch_size
        
        # Calculate predictions
        predicted_energy_kwh = None
        estimated_full_gpu_hours = None
        energy_per_1k_steps = None
        energy_per_billion_tokens = None
        prediction_method = None
        
        # Determine which prediction method to use
        if target_tokens is not None:
            if current_tokens:
                # Use token-based prediction
                scale_factor = target_tokens / current_tokens
                predicted_energy_kwh = total_kwh_cluster * scale_factor
                estimated_full_gpu_hours = total_gpu_hours * scale_factor
                energy_per_billion_tokens = (total_kwh_cluster / current_tokens) * 1e9
                prediction_method = "tokens"
            else:
                print("Warning: Cannot use token-based prediction - missing step count, sequence length, or batch size")
                if target_steps:
                    print("Falling back to step-based prediction")
                    target_tokens = None
        
        if target_steps is not None and target_tokens is None:
            if current_steps and current_steps > 0:
                # Use step-based prediction
                scale_factor = target_steps / current_steps
                predicted_energy_kwh = total_kwh_cluster * scale_factor
                estimated_full_gpu_hours = total_gpu_hours * scale_factor
                energy_per_1k_steps = (total_kwh_cluster / current_steps) * 1000
                prediction_method = "steps"
                
                # Also calculate energy per billion tokens if possible
                if current_tokens:
                    energy_per_billion_tokens = (total_kwh_cluster / current_tokens) * 1e9
            else:
                print("No step metrics found - predictions not available")
        
        if target_steps is None and target_tokens is None:
            print("No target steps or tokens provided - skipping prediction")
            # Still calculate energy per billion tokens if possible
            if current_tokens:
                energy_per_billion_tokens = (total_kwh_cluster / current_tokens) * 1e9
        
        # Print results
        print(f"\n=== ENERGY CONSUMPTION RESULTS ===")
        print(f"Run ID: {run_id}")
        print(f"Project: {project_path}")
        print(f"Nodes: {nodes}")
        print(f"GPUs per node: {gpus_per_node}")
        print(f"Total GPUs: {nodes * gpus_per_node}")
        print(f"Duration: {duration_hours:.2f} hours")
        print(f"Energy per node: {total_kwh_per_node:.2f} kWh")
        print(f"Total cluster energy: {total_kwh_cluster:.2f} kWh")
        print(f"GPU hours: {total_gpu_hours:.2f}")
        
        if predicted_energy_kwh:
            print(f"\n=== PREDICTIONS FOR FULL TRAINING ({prediction_method.upper()}-BASED) ===")
            
            if predicted_energy_kwh >= 1000:
                if prediction_method == "tokens" and target_tokens:
                    print(f"Predicted total energy for {target_tokens/1e9:.1f}B tokens: {predicted_energy_kwh/1000:.2f} MWh ({predicted_energy_kwh:.0f} kWh)")
                elif prediction_method == "steps" and target_steps:
                    print(f"Predicted total energy for {target_steps:,} steps: {predicted_energy_kwh/1000:.2f} MWh ({predicted_energy_kwh:.0f} kWh)")
                else:
                    print(f"Predicted total energy: {predicted_energy_kwh/1000:.2f} MWh ({predicted_energy_kwh:.0f} kWh)")
            else:
                if prediction_method == "tokens" and target_tokens:
                    print(f"Predicted total energy for {target_tokens/1e9:.1f}B tokens: {predicted_energy_kwh:.2f} kWh")
                elif prediction_method == "steps" and target_steps:
                    print(f"Predicted total energy for {target_steps:,} steps: {predicted_energy_kwh:.2f} kWh")
                else:
                    print(f"Predicted total energy: {predicted_energy_kwh:.2f} kWh")
            
            print(f"Estimated GPU hours: {estimated_full_gpu_hours:.2f}")
            if energy_per_1k_steps:
                print(f"Energy per 1K steps: {energy_per_1k_steps:.2f} kWh")
            if energy_per_billion_tokens:
                print(f"Energy per 1B tokens: {energy_per_billion_tokens:.2f} kWh")

        
        print(f"\n=== POWER STATISTICS ===")
        print(f"Average power per node: {power_kw.mean() * 1000:.0f} W")
        print(f"Peak power per node: {power_kw.max() * 1000:.0f} W")
        print(f"Average power per GPU: {(power_kw.mean() * 1000) / gpus_per_node:.0f} W")
        
        # Print efficiency metrics
        if energy_per_billion_tokens:
            print(f"\n=== EFFICIENCY METRICS ===")
            print(f"Energy per billion tokens: {energy_per_billion_tokens:.2f} kWh/B")
        if energy_per_1k_steps:
            print(f"Energy per 1K steps: {energy_per_1k_steps:.2f} kWh")
        
    except Exception as e:
        print(f"Error processing run data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return {
        "run_id": run_id,
        "total_energy_kwh": total_kwh_cluster,
        "predicted_energy_kwh": predicted_energy_kwh,
        "duration_hours": duration_hours,
        "energy_per_1k_steps": energy_per_1k_steps,
        "energy_per_billion_tokens": energy_per_billion_tokens,
        "gpu_hours": total_gpu_hours,
        "estimated_gpu_hours": estimated_full_gpu_hours,
        "gpus_per_node": gpus_per_node,
        "project_path": project_path,
        "current_tokens": current_tokens,
        "current_steps": current_steps,
        "prediction_method": prediction_method
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate energy consumption for ML training runs")
    parser.add_argument("--url", type=str,
                        help="Full WandB URL (e.g., https://wandb.ai/entity/project/runs/run_id)")
    parser.add_argument("--file", type=str,
                        help="Path to local .wandb file")
    parser.add_argument("--nodes", type=int, default=1,
                    help="Number of compute nodes (default: 1, will use W&B config if available)")
    parser.add_argument("--target-steps", type=int, default=None,
                        help="Target number of global steps for prediction (optional)")
    parser.add_argument("--target-tokens", type=float, default=None,
                        help="Target number of tokens for prediction (e.g., 1e12 for 1 trillion tokens)")
    parser.add_argument("--seq-length", type=int, default=None,
                    help="Sequence length (will use W&B config if available)")
    parser.add_argument("--global-batch-size", type=int, default=None,
                    help="Global batch size (will use W&B config if available)")
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not args.url and not args.file:
        parser.error("Either --url or --file must be provided")
    if args.url and args.file:
        parser.error("Cannot specify both --url and --file")
    if args.target_steps and args.target_tokens:
        parser.error("Cannot specify both --target-steps and --target-tokens. Choose one prediction method.")
    
    source = args.url if args.url else args.file
    is_local = bool(args.file)
    
    # Calculate energy
    results = calculate_energy(
        source=source,
        nodes=args.nodes,
        target_steps=args.target_steps,
        target_tokens=args.target_tokens,
        seq_length_override=args.seq_length,
        global_batch_size_override=args.global_batch_size,
        is_local_file=is_local
    )
    
    if results:
        # Output summary in CSV format
        print("\n" + "="*80)
        print("SUMMARY (CSV FORMAT)")
        print("="*80)
        print("run_id,project,nodes,gpus_per_node,total_gpus,energy_kwh,predicted_energy_kwh," +
              "duration_hours,gpu_hours,current_steps,current_tokens_B,energy_per_1k_steps," +
              "energy_per_B_tokens,prediction_method")
        
        # Handle None values for CSV output
        predicted_kwh = f"{results['predicted_energy_kwh']:.2f}" if results['predicted_energy_kwh'] is not None else "N/A"
        pred_method = results.get('prediction_method', 'N/A')
        current_steps = results.get('current_steps', 'N/A')
        current_tokens_b = f"{results['current_tokens']/1e9:.2f}" if results.get('current_tokens') else "N/A"
        energy_per_1k = f"{results['energy_per_1k_steps']:.2f}" if results.get('energy_per_1k_steps') else "N/A"
        energy_per_b = f"{results['energy_per_billion_tokens']:.2f}" if results.get('energy_per_billion_tokens') else "N/A"
        
        print(f"{results['run_id']},{results['project_path']},{args.nodes},{results['gpus_per_node']}," +
              f"{args.nodes * results['gpus_per_node']},{results['total_energy_kwh']:.2f}," +
              f"{predicted_kwh},{results['duration_hours']:.2f},{results['gpu_hours']:.2f}," +
              f"{current_steps},{current_tokens_b},{energy_per_1k},{energy_per_b},{pred_method}")
        
        print("="*80)
    else:
        print("\n" + "="*80)
        print("Failed to calculate energy consumption")
        print("="*80)
        sys.exit(1)

#!/usr/bin/env python3
"""
Synthetic Data Generation Pipeline

This script combines the functionality of add_noise.py and hf_dataset.py to create
a complete synthetic data generation pipeline. It takes a configuration file
as input and processes it to generate synthetic data with specified noise characteristics.
"""

import os
import json
import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def load_config(config_path):
    """
    Load and parse the JSON configuration file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Parsed configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{config_path}' contains invalid JSON.")
        sys.exit(1)


def build_add_noise_args(config):
    """
    Build command-line arguments for add_noise.py based on the configuration.
    
    Args:
        config (dict): Parsed configuration
        
    Returns:
        list: Command-line arguments for add_noise.py
    """
    args = []
    
    # Process noise types
    noise_types = config.get("noise_types", {})
    
    # White noise (always enabled in add_noise.py, but we'll respect the config)
    white = noise_types.get("white", {"enabled": True, "level": 0.01})
    if white.get("enabled", True):
        args.extend(["--white_level", str(white.get("level", 0.01))])
    
    # Pink noise
    pink = noise_types.get("pink", {"enabled": False, "level": 0.01})
    if pink.get("enabled", False):
        args.extend(["--pink", "True"])
        args.extend(["--pink_level", str(pink.get("level", 0.01))])
    
    # Brown noise
    brown = noise_types.get("brown", {"enabled": False, "level": 0.01})
    if brown.get("enabled", False):
        args.extend(["--brown", "True"])
        args.extend(["--brown_level", str(brown.get("level", 0.01))])
    
    # Beeping noise
    beeping = noise_types.get("beeping", {"enabled": False})
    if beeping.get("enabled", False):
        args.extend(["--beeping", "True"])
    
    # Talking noise
    talking = noise_types.get("talking", {"enabled": False})
    if talking.get("enabled", False):
        args.extend(["--talking", "True"])
    
    # Process noise burst parameters
    noise_bursts = config.get("noise_bursts", {"number": 20, "duration": 1.0})
    args.extend(["--noise_burst_num", str(noise_bursts.get("number", 20))])
    args.extend(["--noise_burst_duration", str(noise_bursts.get("duration", 1.0))])
    
    return args


def run_add_noise(config):
    """
    Run the add_noise.py script with the specified configuration.
    
    Args:
        config (dict): Parsed configuration
        
    Returns:
        str: Path to the output directory where noise-augmented data was saved
    """
    # Build the arguments
    args = build_add_noise_args(config)
    
    # Construct the command
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "add_noise.py")] + args
    
    print(f"Running add_noise.py with arguments: {' '.join(args)}")
    
    # Run the command
    process = subprocess.run(cmd, check=True)
    
    # Determine the output directory name based on the configuration
    noise_types = config.get("noise_types", {})
    noise_bursts = config.get("noise_bursts", {"number": 20, "duration": 1.0})
    
    white = noise_types.get("white", {"enabled": True, "level": 0.01})
    pink = noise_types.get("pink", {"enabled": False, "level": 0.01})
    brown = noise_types.get("brown", {"enabled": False, "level": 0.01})
    beeping = noise_types.get("beeping", {"enabled": False})
    talking = noise_types.get("talking", {"enabled": False})
    
    # Recreate the same directory naming logic as in add_noise.py
    output_dir = (
        "data_w_noise"
        + (f"_white{white.get('level', 0.01)}" if white.get("enabled", True) else "")
        + (f"_pink{pink.get('level', 0.01)}" if pink.get("enabled", False) else "")
        + (f"_brown{brown.get('level', 0.01)}" if brown.get("enabled", False) else "")
        + f"_nbn{noise_bursts.get('number', 20)}"
        + f"_nbd{noise_bursts.get('duration', 1.0)}"
        + ("_w-beeping" if beeping.get("enabled", False) else "_wo-beeping")
        + ("_w-talking" if talking.get("enabled", False) else "_wo-talking")
    )
    
    # Make sure the path is absolute
    output_dir = os.path.abspath(output_dir)
    
    return output_dir


def run_hf_dataset(noisy_data_dir):
    """
    Run the hf_dataset.py script to create the HuggingFace dataset.
    
    Args:
        noisy_data_dir (str): Path to the directory containing noise-augmented data
    
    Returns:
        int: Return code from the process
    """
    cough_segmentation_dir = "CoughSegmentation"
    original_data_dir = os.path.join(cough_segmentation_dir, "Data")
    
    # Save original directory if it exists
    backup_dir = None
    if os.path.exists(original_data_dir):
        backup_dir = original_data_dir + ".bak"
        print(f"Backing up original data directory to {backup_dir}")
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)  # Remove existing backup if it exists
        shutil.move(original_data_dir, backup_dir)
    
    # Create the CoughSegmentation directory if it doesn't exist
    if not os.path.exists(cough_segmentation_dir):
        os.makedirs(cough_segmentation_dir)
    
    # Create a symbolic link from the noisy data directory to the expected location
    print(f"Creating symbolic link from {noisy_data_dir} to {original_data_dir}")
    os.symlink(noisy_data_dir, original_data_dir)
    
    try:
        # Run the HuggingFace dataset script
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "hf_dataset.py")]
        print("Running hf_dataset.py to create the HuggingFace dataset...")
        process = subprocess.run(cmd, check=True)
        return_code = process.returncode
    finally:
        # Clean up: remove the symbolic link
        print("Cleaning up symbolic link")
        os.unlink(original_data_dir)
        
        # Restore the original data directory if it existed
        if backup_dir and os.path.exists(backup_dir):
            print(f"Restoring original data directory from {backup_dir}")
            shutil.move(backup_dir, original_data_dir)
    
    return return_code


def main():
    """
    Main function to run the synthetic data generation pipeline.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Pipeline")
    parser.add_argument("config", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load the configuration
    config = load_config(args.config)
    
    # Print configuration summary
    print("=== Synthetic Data Generation Configuration ===")
    noise_types = config.get("noise_types", {})
    if noise_types.get("white", {}).get("enabled", True):
        print(f"- White noise: Enabled (level: {noise_types.get('white', {}).get('level', 0.01)})")
    else:
        print("- White noise: Disabled")
        
    if noise_types.get("pink", {}).get("enabled", False):
        print(f"- Pink noise: Enabled (level: {noise_types.get('pink', {}).get('level', 0.01)})")
    else:
        print("- Pink noise: Disabled")
        
    if noise_types.get("brown", {}).get("enabled", False):
        print(f"- Brown noise: Enabled (level: {noise_types.get('brown', {}).get('level', 0.01)})")
    else:
        print("- Brown noise: Disabled")
        
    if noise_types.get("beeping", {}).get("enabled", False):
        print("- Beeping noise: Enabled")
    else:
        print("- Beeping noise: Disabled")
        
    if noise_types.get("talking", {}).get("enabled", False):
        print("- Talking noise: Enabled")
    else:
        print("- Talking noise: Disabled")
    
    noise_bursts = config.get("noise_bursts", {"number": 20, "duration": 1.0})
    print(f"- Noise bursts: {noise_bursts.get('number', 20)} bursts, {noise_bursts.get('duration', 1.0)}s each")
    print("==============================================")
    
    # Run the pipeline
    print("\nStep 1: Generating noisy data...")
    noisy_data_dir = run_add_noise(config)
    print(f"Noisy data generated in: {noisy_data_dir}")
    
    print("\nStep 2: Creating HuggingFace dataset...")
    run_hf_dataset(noisy_data_dir)
    
    print("\nSynthetic data generation complete!")
    print("The generated HuggingFace dataset is available in 'data/dataset_hf'.")


if __name__ == "__main__":
    main()

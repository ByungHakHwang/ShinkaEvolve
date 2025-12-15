"""
Script to prepare and validate test data for qt-LR coefficient rule discovery.
This helps convert your 100,000 data points into the required format.
"""

import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set


def convert_your_data_format(your_data: Dict) -> Dict:
    """
    Convert your data format to the format expected by evaluate.py.
    
    MODIFY THIS FUNCTION according to your actual data format.
    
    Args:
        your_data: Your original data point
        
    Returns:
        Dictionary in the required format
    """
    # Example conversion - ADJUST THIS to match your data structure
    return {
        'lambda': your_data['lambda'],  # Should be list of ints
        'mu': your_data['mu'],          # Should be list of ints
        'nu': your_data['nu'],          # Should be list of ints
        'A': your_data['A'],            # Should be list of [row, col] pairs
        'B': your_data['B'],            # Should be list of [row, col] pairs
        'C': your_data['C'],            # Should be list of [row, col] pairs
        'metadata': {
            # NOTE: is_pieri is no longer used for weighting
            # but kept for analysis purposes
            'is_pieri': len(your_data['mu']) == 1 or (len(your_data['mu']) > 1 and all(x == 0 for x in your_data['mu'][1:])),
            'source': your_data.get('source', 'unknown'),
        }
    }


def validate_test_case(case: Dict) -> Tuple[bool, str]:
    """
    Validate that a test case has the correct format.
    
    Returns:
        (is_valid, error_message)
    """
    required_keys = ['lambda', 'mu', 'nu', 'A', 'B', 'C']
    
    # Check all required keys present
    for key in required_keys:
        if key not in case:
            return False, f"Missing required key: {key}"
    
    # Check partitions are lists of ints
    for key in ['lambda', 'mu', 'nu']:
        if not isinstance(case[key], list):
            return False, f"{key} must be a list"
        if not all(isinstance(x, int) for x in case[key]):
            return False, f"{key} must contain only integers"
        # Check non-increasing
        for i in range(len(case[key]) - 1):
            if case[key][i] < case[key][i+1]:
                return False, f"{key} must be non-increasing"
    
    # Check A, B, C are lists of cells
    for key in ['A', 'B', 'C']:
        if not isinstance(case[key], list):
            return False, f"{key} must be a list"
        for cell in case[key]:
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                return False, f"{key} must contain [row, col] pairs"
            if not all(isinstance(x, int) and x >= 0 for x in cell):
                return False, f"{key} cells must be non-negative integers"
    
    return True, ""


def stratified_sample(
    all_data: List[Dict],
    total_samples: int,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    pieri_ratio: float = 0.3
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create stratified train/val/test splits.
    
    Args:
        all_data: All available data points
        total_samples: Total number to use from all_data
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        pieri_ratio: Target ratio of Pieri cases to include
        
    Returns:
        (train_data, val_data, test_data)
    """
    random.seed(42)  # Reproducibility
    
    # Separate Pieri and non-Pieri cases
    pieri_cases = [d for d in all_data if d.get('metadata', {}).get('is_pieri', False)]
    general_cases = [d for d in all_data if not d.get('metadata', {}).get('is_pieri', False)]
    
    print(f"Total data: {len(all_data)}")
    print(f"  - Pieri cases: {len(pieri_cases)}")
    print(f"  - General cases: {len(general_cases)}")
    
    # Sample from each category
    num_pieri = int(total_samples * pieri_ratio)
    num_general = total_samples - num_pieri
    
    sampled_pieri = random.sample(pieri_cases, min(num_pieri, len(pieri_cases)))
    sampled_general = random.sample(general_cases, min(num_general, len(general_cases)))
    
    sampled_data = sampled_pieri + sampled_general
    random.shuffle(sampled_data)
    
    # Split into train/val/test
    train_data = sampled_data[:train_samples]
    val_data = sampled_data[train_samples:train_samples + val_samples]
    test_data = sampled_data[train_samples + val_samples:train_samples + val_samples + test_samples]
    
    print(f"\nSplit sizes:")
    print(f"  - Train: {len(train_data)}")
    print(f"  - Validation: {len(val_data)}")
    print(f"  - Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def prepare_data_from_your_source(
    input_path: str,
    output_dir: str,
    train_size: int = 5000,
    val_size: int = 1000,
    test_size: int = 2000
):
    """
    Main function to prepare data from your source.
    
    Args:
        input_path: Path to your original data file
        output_dir: Directory to save processed data
        train_size: Number of training examples
        val_size: Number of validation examples
        test_size: Number of test examples
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading your data...")
    with open(input_path, 'rb') as f:
        your_data = pickle.load(f)
    
    print(f"Loaded {len(your_data)} data points")
    
    # Convert to required format
    print("\nConverting to required format...")
    converted_data = []
    for i, item in enumerate(your_data):
        try:
            converted = convert_your_data_format(item)
            
            # Validate
            is_valid, error = validate_test_case(converted)
            if not is_valid:
                print(f"Warning: Skipping invalid case {i}: {error}")
                continue
            
            converted_data.append(converted)
        except Exception as e:
            print(f"Warning: Error converting case {i}: {e}")
            continue
    
    print(f"Successfully converted {len(converted_data)} cases")
    
    # Create stratified splits
    total_needed = train_size + val_size + test_size
    train_data, val_data, test_data = stratified_sample(
        converted_data,
        total_samples=min(total_needed, len(converted_data)),
        train_samples=train_size,
        val_samples=val_size,
        test_samples=test_size
    )
    
    # Save splits
    print("\nSaving data splits...")
    
    with open(output_path / "test_data_qt_LR.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  - Training data: {output_path / 'test_data_qt_LR.json'}")
    
    with open(output_path / "validation_data_qt_LR.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"  - Validation data: {output_path / 'validation_data_qt_LR.json'}")
    
    with open(output_path / "test_data_qt_LR_final.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"  - Test data: {output_path / 'test_data_qt_LR_final.json'}")
    
    # Save statistics
    stats = {
        'total_original': len(your_data),
        'total_converted': len(converted_data),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'train_pieri_ratio': sum(1 for d in train_data if d.get('metadata', {}).get('is_pieri', False)) / len(train_data),
    }
    
    with open(output_path / "data_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Data preparation complete!")
    print(f"Statistics saved to: {output_path / 'data_stats.json'}")


def create_sample_data(output_path: str, num_samples: int = 100):
    """
    Create sample/dummy data for testing the pipeline.
    USE THIS for initial testing before using real data.
    """
    print(f"Creating {num_samples} sample data points...")
    
    sample_data = []
    
    for i in range(num_samples):
        # Generate random partitions
        lambda_size = random.randint(2, 5)
        lambda_part = sorted([random.randint(1, 6) for _ in range(lambda_size)], reverse=True)
        
        mu_size = random.randint(1, 3)
        mu_part = sorted([random.randint(1, 4) for _ in range(mu_size)], reverse=True)
        
        # Nu should be "larger" than lambda and mu
        nu_size = max(lambda_size, mu_size) + random.randint(0, 2)
        nu_part = sorted([random.randint(1, 7) for _ in range(nu_size)], reverse=True)
        
        # Generate random A, B, C sets (not necessarily correct, just for testing)
        A = [[random.randint(0, nu_size-1), random.randint(0, max(nu_part)-1)] for _ in range(random.randint(1, 5))]
        B = [[random.randint(0, lambda_size-1), random.randint(0, max(lambda_part)-1)] for _ in range(random.randint(1, 3))]
        C = [[random.randint(0, mu_size-1), random.randint(0, max(mu_part)-1)] for _ in range(random.randint(1, 2))]
        
        sample_data.append({
            'lambda': lambda_part,
            'mu': mu_part,
            'nu': nu_part,
            'A': A,
            'B': B,
            'C': C,
            'metadata': {
                'is_pieri': len(mu_part) == 1,
                'source': 'sample_generated'
            }
        })
    
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✓ Sample data saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare test data for qt-LR evolution")
    parser.add_argument("--mode", choices=['sample', 'real'], default='sample',
                        help="'sample' for dummy data, 'real' for actual data conversion")
    parser.add_argument("--input", type=str, help="Input file (for 'real' mode)")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory")
    parser.add_argument("--train-size", type=int, default=7500)
    parser.add_argument("--val-size", type=int, default=1500)
    parser.add_argument("--test-size", type=int, default=2500)
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        # Create sample data for testing
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        create_sample_data(str(output_path / "test_data_qt_LR.json"), num_samples=100)
    else:
        # Process real data
        if not args.input:
            print("Error: --input required for 'real' mode")
            exit(1)
        prepare_data_from_your_source(
            args.input,
            args.output_dir,
            args.train_size,
            args.val_size,
            args.test_size
        )
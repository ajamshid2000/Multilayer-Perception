"""
Split Wisconsin breast cancer dataset into training and validation CSV files.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader


def main():
    try:
        with open("../config.json") as f:
            conf = json.load(f)
    except:
        print("config.json does not exist, initializing using default value")
        conf = 0
        
    parser = argparse.ArgumentParser(
        description='Split breast cancer dataset into training and validation CSV files.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=conf["data"]["data_dir"] 
        if conf 
        and "data" in conf 
        and "data_dir" in conf["data"] 
        else '../data',
        help='Directory for dataset files (default: data)'
    )
    parser.add_argument(
        '--source-file',
        type=str,
        default=conf["data"]["dataset"] 
        if conf 
        and "data" in conf 
        and "dataset" in conf["data"] 
        else 'data.csv',
        help='Source CSV file name for breast cancer dataset'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=conf["training"]["validation_split"] 
        if conf 
        and "training" in conf 
        and "validation_split" in conf["training"] 
        else 0.2,
        help='Validation split ratio (default: 0.2)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=conf["training"]["random_seed"] 
        if conf 
        and "training" in conf 
        and "random_seed" in conf["training"] 
        else 42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    raw_path = os.path.join(args.data_dir, args.source_file)
    if not os.path.exists(raw_path):
        print(f"Raw dataset not found at {raw_path}. Generating a CSV from scikit-learn...")
    
    X, y = DataLoader.load_breast_cancer_csv(args.data_dir, args.source_file)
    train_path, valid_path = DataLoader.save_breast_cancer_split(
        X,
        y,
        output_dir=args.data_dir,
        train_filename='data_train.csv',
        valid_filename='data_valid.csv',
        random_state=args.random_seed,
        test_size=args.validation_split
    )
    print(f"Breast cancer dataset split complete.")
    print(f"  Training file: {train_path}")
    print(f"  Validation file: {valid_path}")
    print(f"  Training examples: {len(X) - int(len(X) * args.validation_split)}")
    print(f"  Validation examples: {int(len(X) * args.validation_split)}")


if __name__ == '__main__':
    main()

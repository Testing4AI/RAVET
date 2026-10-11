#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O utilities for file operations and JSON handling.
"""

import os
import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_results(results, dir, file_name="debug"):
    """Save results to a JSON file in the query_results directory"""
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'results/query_results/{dir}'):
        os.makedirs(f'results/query_results/{dir}', exist_ok=True)
    with open(os.path.join(f'results/query_results/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)


def load_results(file_name):
    """Load results from the results directory"""
    with open(os.path.join('results', file_name)) as file:
        results = json.load(file)
    return results


def save_json(results, file_path="debug.json", indent=4):
    """Save data to a JSON file, creating directories if needed."""
    print(file_path)
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=NpEncoder, indent=indent)


def load_json(file_path):
    """Load data from a JSON file"""
    with open(file_path) as file:
        results = json.load(file)
    return results


def create_file_if_not_exists(file_path):
    """Create a file if it doesn't exist"""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write('')
        print(f"File '{file_path}' created.")
    else:
        print(f"File '{file_path}' already exists.")


def file_exist(file_path):
    """Ensure file and its directory exist"""
    print(file_path)
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write("This is a newly created file.\n")
    else:
        print(f"The file '{file_path}' already exists.")


def find_latest_injection_result(basepath: str, dataset: str, knum: int) -> str:
    """
    Find the latest injection result for a given dataset and k-number.
    
    Args:
        basepath: Base output path (e.g. './output')
        dataset: Dataset name (e.g. 'nfcorpus')
        knum: Number of KOs (e.g. 3)
        
    Returns:
        Path to the injection_result.json file
    
    Raises:
        FileNotFoundError: If no matching injection result is found
    """
    # Construct search path matching structure: .../watermark_injections/{dataset}/k{knum}/
    k_dir = f"k{knum}"
    search_dir = os.path.join(basepath, "watermark_injections", dataset, k_dir)
    
    if not os.path.exists(search_dir):
        # Try legacy structure if new structure doesn't exist? 
        # For now, let's stick to the new structure we just defined.
        raise FileNotFoundError(f"No injections found for dataset '{dataset}' with k={knum} at {search_dir}")
        
    # List all timestamp directories
    try:
        subdirs = [os.path.join(search_dir, d) for d in os.listdir(search_dir) 
                   if os.path.isdir(os.path.join(search_dir, d))]
    except OSError as e:
        raise FileNotFoundError(f"Error accessing directory {search_dir}: {e}")

    if not subdirs:
        raise FileNotFoundError(f"No result folders found in {search_dir}")
        
    # Sort by name (timestamp) descending
    subdirs.sort(key=lambda x: os.path.basename(x), reverse=True)
    
    # Check for injection_result.json in the latest folder
    for latest_dir in subdirs:
        result_path = os.path.join(latest_dir, "injection_result.json")
        if os.path.exists(result_path):
            print(f"Auto-selected latest injection result: {result_path}")
            return result_path
            
    raise FileNotFoundError(f"No 'injection_result.json' found in any subdirectories of {search_dir}")

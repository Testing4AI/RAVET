#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic sampling utilities for watermark generation.
"""

import json
import hashlib
import random
import heapq

import numpy as np
import torch


def setup_seeds(seed):
    """Setup random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def deterministic_sample(items, n, secret_key, salt=''):
    """
    Deterministic sampling based on secret key.
    
    Args:
        items: List of items to sample from
        n: Number of items to sample
        secret_key: Secret key for deterministic sampling
        salt: Salt value for additional randomization
        
    Returns:
        list: Sampled items
    """
    if n > len(items):
        raise ValueError("Sample size cannot exceed total items")
    
    indexed_items = list(enumerate(items))
    
    def sort_key(item):
        index, _ = item
        h = hashlib.sha256((secret_key + salt + str(index)).encode()).hexdigest()
        return h

    smallest = heapq.nsmallest(n, indexed_items, key=sort_key)
    return [item for _, item in smallest]


def deterministic_select_kos(ko_pool, n, secret_key, salt='ko-selection'):
    """
    Deterministically select KOs from pool based on secret key.
    
    Args:
        ko_pool: List of KO objects
        n: Number of KOs to select
        secret_key: Secret key
        salt: Salt value
        
    Returns:
        list: Selected KOs
    """
    if n > len(ko_pool):
        raise ValueError(f"Selection count {n} exceeds KO pool size {len(ko_pool)}")
    
    # Calculate hash for each KO based on content and key
    kos_with_hash = []
    for ko in ko_pool:
        ko_str = json.dumps(ko, sort_keys=True, ensure_ascii=False)
        hash_val = hashlib.sha256((secret_key + salt + ko_str).encode()).hexdigest()
        kos_with_hash.append((hash_val, ko))
    
    # Sort by hash and select top n
    kos_with_hash.sort(key=lambda x: x[0])
    selected_kos = [ko for _, ko in kos_with_hash[:n]]
    
    return selected_kos

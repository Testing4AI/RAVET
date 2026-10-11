#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset loading utilities for BEIR and other datasets.
"""

import os
import random

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from datasets import load_dataset as hf_load_dataset

from sentinelrag.utils.paths import default_datasets_dir


def load_beir_datasets(dataset_name, split):
    """
    Load BEIR datasets.
    
    Args:
        dataset_name: One of 'nq', 'msmarco', 'hotpotqa', 'nfcorpus', 'trec-covid'
        split: 'train' or 'test'
        
    Returns:
        tuple: (corpus, queries, qrels)
    """
    
    if dataset_name in ['msmarco', 'msmarco_200k']:
        split = 'train'
    
    # For local datasets, skip download
    if dataset_name == 'multihoprag':
        out_dir = str(default_datasets_dir())
        data_path = os.path.join(out_dir, dataset_name)
    else:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        out_dir = str(default_datasets_dir())
        data_path = os.path.join(out_dir, dataset_name)
        
        if not os.path.exists(data_path):
            data_path = util.download_and_unzip(url, out_dir)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        split = 'train'
    
    # If there is only a single tsv file in qrels/, load that tsv regardless of split
    qrels_path = os.path.join(data_path, "qrels")
    if os.path.isdir(qrels_path):
        qrels_files = [f for f in os.listdir(qrels_path) if f.endswith(".tsv")]
        if len(qrels_files) == 1:
            split = qrels_files[0][:-4]  # Remove .tsv extension

    corpus, queries, qrels = data.load(split=split)

    return corpus, queries, qrels


def data_prepare(dataname, dataset):
    """
    Prepare dataset for watermark generation.
    
    Args:
        dataname: Dataset name
        dataset: Raw dataset
        
    Returns:
        dict: Preprocessed dataset
    """
    ndataset = {}

    if dataname == 'closed_qa':
        for i, item in enumerate(dataset):
            ndataset[str(i)] = f"{item['instruction']}. {item['context']}"
    else:
        for key, value in dataset.items():
            ndataset[key] = value['text']

    return ndataset


def load_dataset_for_watermark_generation(eval_dataset, split, sample_size):
    """
    Load dataset for watermark generation.
    
    Args:
        eval_dataset: Dataset name
        split: Dataset split
        sample_size: Sample size for watermark generation (0 = full dataset)
    
    Returns:
        tuple: (sampled_dataset, full_dataset, raw_dataset_dict)
    """
    if eval_dataset == 'closed_qa':
        train_dataset = hf_load_dataset("databricks/databricks-dolly-15k", split='train')
        closed_qa_dataset = train_dataset.filter(lambda example: example['category'] == 'closed_qa')
        ndataset = data_prepare('closed_qa', closed_qa_dataset)
    else:
        corpus, queries, qrels = load_beir_datasets(eval_dataset, split)
        ndataset = data_prepare(eval_dataset, corpus)
    
    full_dataset = list(ndataset.values())
    
    if sample_size == 0:
        sampled_dataset = full_dataset
    else:
        all_keys = list(ndataset.keys())
        if sample_size > len(all_keys):
            sampled_dataset = full_dataset
        else:
            sampled_keys = random.sample(all_keys, sample_size)
            sampled_dataset = [ndataset[key] for key in sampled_keys]
    
    return sampled_dataset, full_dataset, ndataset

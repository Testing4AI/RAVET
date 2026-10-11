#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Generation for BEIR Datasets

This module generates embeddings for all items in BEIR datasets and stores them
in Parquet format for efficient storage and retrieval.
"""

import os
import time
import argparse
import json
from pathlib import Path

import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from sentinelrag.utils import load_beir_datasets, load_models


class EmbeddingGenerator:
    """Generate and store embeddings for datasets in Parquet format"""
    
    def __init__(self, embedding_model, tokenizer, get_emb, device, use_multi_gpu=False):
        """
        Initialize the embedding generator.
        
        Args:
            embedding_model: The embedding model (e.g., Contriever)
            tokenizer: Tokenizer for the embedding model
            get_emb: Function to get embeddings from model
            device: torch device ('cuda' or 'cpu')
            use_multi_gpu: Whether to use multiple GPUs via DataParallel
        """
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.get_emb = get_emb
        self.device = device
        self.use_multi_gpu = use_multi_gpu
        
        # Setup multi-GPU if requested and available
        if self.use_multi_gpu and torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            if self.gpu_count > 1:
                print(f"Using {self.gpu_count} GPUs for parallel embedding generation")
                # Create separate model replicas for each GPU
                self.gpu_models = []
                for gpu_id in range(self.gpu_count):
                    model_replica = type(embedding_model).from_pretrained(
                        embedding_model.config._name_or_path
                    )
                    model_replica.eval()
                    model_replica.to(f'cuda:{gpu_id}')
                    self.gpu_models.append(model_replica)
                print(f"Created {len(self.gpu_models)} model replicas across GPUs")
            else:
                print(f"Only 1 GPU available, using single GPU mode")
                self.use_multi_gpu = False
                self.embedding_model.eval()
                self.embedding_model.to(self.device)
        else:
            self.embedding_model.eval()
            self.embedding_model.to(self.device)
    
    def get_embeddings_batch(self, texts):
        """Get embeddings for a batch of texts"""
        if self.use_multi_gpu:
            # Split batch across GPUs manually
            return self._get_embeddings_multi_gpu(texts)
        else:
            # Single GPU/CPU processing
            text_inputs = self.tokenizer(
                texts, padding=True, truncation=True, 
                return_tensors="pt", max_length=512
            )
            text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
            
            with torch.no_grad():
                embeddings = self.get_emb(self.embedding_model, text_inputs)
                
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)
                text_embs = embeddings.cpu().tolist()
            return text_embs

    @staticmethod
    def _normalize_to_text(value):
        """Convert arbitrary dataset fields into tokenizer-safe text."""
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return " ".join(EmbeddingGenerator._normalize_to_text(v) for v in value)
        if isinstance(value, dict):
            # Prefer common textual keys when present.
            for key in ("text", "contents", "content", "body", "title"):
                if key in value:
                    return EmbeddingGenerator._normalize_to_text(value[key])
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    
    def _get_embeddings_multi_gpu(self, texts):
        """Process batch across multiple GPUs in parallel using threading"""
        import threading
        
        # Split texts into sub-batches for each GPU
        batch_size = len(texts)
        sub_batch_size = (batch_size + self.gpu_count - 1) // self.gpu_count
        
        results = [None] * self.gpu_count
        threads = []
        
        def process_on_gpu(gpu_id, sub_texts, result_idx):
            """Process a sub-batch on a specific GPU"""
            if len(sub_texts) == 0:
                results[result_idx] = []
                return
                
            device = f'cuda:{gpu_id}'
            model = self.gpu_models[gpu_id]
            
            text_inputs = self.tokenizer(
                sub_texts, padding=True, truncation=True,
                return_tensors="pt", max_length=512
            )
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            
            with torch.no_grad():
                embeddings = self.get_emb(model, text_inputs)
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)
                # Move to CPU and explicitly delete GPU tensors
                cpu_embeddings = embeddings.cpu()
                del embeddings, text_inputs
                results[result_idx] = cpu_embeddings
        
        # Create and start threads for each GPU
        for gpu_id in range(self.gpu_count):
            start_idx = gpu_id * sub_batch_size
            end_idx = min(start_idx + sub_batch_size, batch_size)
            sub_texts = texts[start_idx:end_idx]
            
            thread = threading.Thread(
                target=process_on_gpu,
                args=(gpu_id, sub_texts, gpu_id)
            )
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Combine results from all GPUs
        all_embeddings = [emb for result in results if result is not None and len(result) > 0 for emb in result]
        return [emb.tolist() for emb in all_embeddings]
    
    def generate_and_save(
        self,
        dataset,
        output_dir,
        save_interval,
        resume=False,
        batch_size=None
    ):
        """
        Generate embeddings for all items in dataset and save to multiple Parquet files.
        
        Args:
            dataset: Dataset dict with document texts
            output_dir: Directory to save the Parquet shard files
            batch_size: Batch size for processing (auto-set if None)
            resume: Whether to resume from existing files
            save_interval: Save progress every N batches 
        """
        # Auto-set batch size based on device configuration
        if batch_size is None:
            if self.use_multi_gpu:
                batch_size = 512 * self.gpu_count  # Scale batch size with GPU count
                print(f"Auto-setting batch size to {batch_size} ({self.gpu_count} GPUs × 512)")
            elif self.device == 'cuda':
                batch_size = 256
                print(f"Auto-setting batch size to {batch_size} (single GPU)")
            else:
                batch_size = 64
                print(f"Auto-setting batch size to {batch_size} (CPU)")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def shard_path(shard_index: int) -> Path:
            return output_dir / f"embeddings_{shard_index:05d}.parquet"
        
        # Check if we can resume from existing files
        start_idx = 0
        file_counter = 0
        if resume:
            # Find all existing parquet files in the directory
            existing_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.parquet')])
            if existing_files:
                # Count total documents in existing files
                for file in existing_files:
                    try:
                        print(f"Reading existing file: {file}")
                        pf = pq.ParquetFile(output_dir / file)
                        start_idx += pf.metadata.num_rows
                        file_counter += 1
                    except Exception as e:
                        print(f"Could not read {file}: {e}")
                        break
                print(f"Resuming from existing files. Already have {start_idx} documents in {file_counter} files.")
            else:
                print("No existing files found. Starting fresh.")
        
        sorted_keys = sorted(dataset.keys())
        total = len(sorted_keys)
        
        if start_idx > 0:
            sorted_keys = sorted_keys[start_idx:]
        
        print(f"Processing {len(sorted_keys)} documents (total: {total})")
        print(f"Saving progress every {save_interval} batches")
        print(f"Output directory: {output_dir}")
        
        # Prepare data storage for the next shard file
        all_embeddings = []
        all_documents = []
        all_ids = []
        all_metadatas = []
        
        start_time = time.time()
        last_report_time = time.time()
        last_report_count = 0
        batch_counter = 0
        
        # Process in batches
        for i in range(0, len(sorted_keys), batch_size):
            batch_keys = sorted_keys[i:i+batch_size]
            batch_texts = []
            
            for j, key in enumerate(batch_keys):
                count = start_idx + i + j
                value = dataset[key]
                if isinstance(value, dict):
                    text_value = self._normalize_to_text(value.get('text', ''))
                    title_value = self._normalize_to_text(value.get('title', ''))
                else:
                    text_value = self._normalize_to_text(value)
                    title_value = ''
                
                batch_texts.append(text_value)
                all_ids.append(f'id_{count}')
                all_documents.append(text_value)
                all_metadatas.append({
                    'title': title_value,
                    'id': key, 
                    'change': False
                })
            
            # Generate embeddings for batch
            batch_embeddings = self.get_embeddings_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            batch_counter += 1
            
            # Save progress every save_interval batches (write a NEW shard file each time)
            if batch_counter >= save_interval or i + batch_size >= len(sorted_keys):
                processed_total = start_idx + i + len(batch_keys)
                out_path = shard_path(file_counter)
                print(f"\n💾 Writing {out_path.name} at {processed_total} documents...")

                # Create DataFrame from accumulated data
                df_chunk = pd.DataFrame(
                    {
                        'embeddings': all_embeddings,
                        'documents': all_documents,
                        'ids': all_ids,
                        'metadatas': all_metadatas,
                    }
                )

                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                pq.write_table(table, out_path, compression='snappy')

                print(
                    f"✓ Saved {len(df_chunk)} documents to {out_path.name} "
                    f"(total: {processed_total})"
                )

                # Clear accumulated data for the next shard
                all_embeddings = []
                all_documents = []
                all_ids = []
                all_metadatas = []
                batch_counter = 0
                file_counter += 1
            
            # Periodically clear GPU memory cache to prevent fragmentation
            if self.use_multi_gpu and (i + batch_size) % 10000 < batch_size:
                torch.cuda.empty_cache()
            
            # Progress reporting
            if (i + batch_size) % 500 < batch_size or i + batch_size >= len(sorted_keys):
                processed = min(i + batch_size, len(sorted_keys))
                
                # Overall average speed
                elapsed = time.time() - start_time
                avg_speed = processed / elapsed if elapsed > 0 else 0
                
                # Current batch speed (since last report)
                time_since_last = time.time() - last_report_time
                docs_since_last = processed - last_report_count
                current_speed = docs_since_last / time_since_last if time_since_last > 0 else 0
                
                # Update for next report
                last_report_time = time.time()
                last_report_count = processed
                
                eta = (len(sorted_keys) - processed) / avg_speed if avg_speed > 0 else 0
                print(f"Processed: {start_idx + processed}/{total} "
                      f"({100*(start_idx + processed)/total:.1f}%) | "
                      f"Avg: {avg_speed:.1f} docs/s | Current: {current_speed:.1f} docs/s | "
                      f"ETA: {eta/60:.1f} min")
        
        total_time = time.time() - start_time
        avg_speed = len(sorted_keys) / total_time if total_time > 0 else 0
        
        # Final summary
        total_rows = 0
        shard_files = sorted(output_dir.glob("*.parquet"))
        for shard in shard_files:
            try:
                total_rows += len(pd.read_parquet(shard))
            except Exception:
                pass

        print(f"\n✓ Done! Saved shards in {output_dir}")
        print(f"Shard files: {len(shard_files)}")
        print(f"Total documents: {total_rows}")
        print(f"Total time: {total_time/60:.2f} min | Avg speed: {avg_speed:.1f} docs/s")
        total_size = sum(s.stat().st_size for s in shard_files if s.exists())
        print(f"Total size: {total_size / (1024**2):.2f} MB")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate embeddings and save to Parquet')
    parser.add_argument("--eval_model_code", type=str, default="contriever",
                      choices=["contriever", "contriever-msmarco", "ance"])
    parser.add_argument('--eval_dataset', type=str, default='nfcorpusward')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--output_dir', type=str, default='./embeddings',
                      help='Directory to save Parquet files')
    parser.add_argument('--gpu_id', type=int, default=1, 
                      help='GPU ID to use (ignored if --multi_gpu is set)')
    parser.add_argument('--multi_gpu', action='store_true',
                      help='Use all available GPUs via DataParallel')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size for embedding generation (auto-set if None)')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from existing Parquet file if available')
    parser.add_argument('--save_interval', type=int, default=100,
                      help='Write a new Parquet shard every N batches')
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Device setup
    if args.multi_gpu and torch.cuda.is_available():
        device = 'cuda'
        gpu_count = torch.cuda.device_count()
        print(f"Multi-GPU mode enabled: {gpu_count} GPUs available")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.cuda.set_device(args.gpu_id)
            print(f"Single GPU mode: using GPU {args.gpu_id}")
        else:
            print("CPU mode")
    
    # Load model and dataset
    print(f"\nLoading model: {args.eval_model_code}")
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    
    print(f"Loading dataset: {args.eval_dataset} ({args.split})")
    if args.eval_dataset == 'msmarco' or args.eval_dataset == 'msmarco_200k':
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, 'train')
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
    
    print(f"Dataset size: {len(corpus)} documents")
    
    # Create a per-run output directory that will contain shard files.
    output_dir = Path(args.output_dir)
    run_output_dir = output_dir / f"{args.eval_dataset}_{args.eval_model_code}_{args.split}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_output_dir}")
    
    # Initialize generator
    generator = EmbeddingGenerator(
        model, tokenizer, get_emb, device, 
        use_multi_gpu=args.multi_gpu
    )
    
    # Generate and save embeddings
    generator.generate_and_save(
        corpus, 
        str(run_output_dir),
        batch_size=args.batch_size,
        resume=args.resume,
        save_interval=args.save_interval,
    )


if __name__ == '__main__':
    main()

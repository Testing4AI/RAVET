#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaDB Vector Store for RAG watermark detection

This module provides vector storage and retrieval capabilities using ChromaDB.
"""

import os
import re
import time

import chromadb
import torch

from sentinelrag.utils.paths import default_chroma_dir


ChromadbPath = str(default_chroma_dir())


class VectorStore:
    """Vector store backed by ChromaDB for document embedding and retrieval"""
    
    def __init__(self, embedding_model, tokenizer, get_emb, dataset, device, 
                 collection_name, use_local, distance='cosine', chroma_path=None):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: The embedding model (e.g., Contriever)
            tokenizer: Tokenizer for the embedding model
            get_emb: Function to get embeddings from model
            dataset: Dataset dict with document texts
            device: torch device ('cuda' or 'cpu')
            collection_name: Name for the ChromaDB collection
            use_local: Whether to use existing local collection
            distance: Distance metric ('cosine', 'l2', 'ip')
            chroma_path: Optional override for the ChromaDB persistence directory.
        """
        self.chroma_path = str(chroma_path or ChromadbPath)
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        collections = self.chroma_client.list_collections()
        # print(collections)
        
        collection_exists = any(col.name == collection_name for col in collections)
        
        if collection_exists and use_local:
            print(f"Using existing local chromadb named {collection_name}")
            self.collection = self.chroma_client.get_collection(name=collection_name)
        else:
            if collection_exists:
                print(f"Deleting existing local chromadb named {collection_name}")
                self.chroma_client.delete_collection(name=collection_name)
            print(f"Creating new chromadb collection named {collection_name}")
            self.collection = self.chroma_client.create_collection(
                name=collection_name, 
                metadata={"hnsw:space": distance}
            )
        
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.get_emb = get_emb
        self.device = device
        self.count = None
        self.embedding_model.eval()
        self.embedding_model.to(self.device)

    def get_embedding(self, text):
        """Get embedding for a single text"""
        text_input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        text_input = {key: value.to(self.device) for key, value in text_input.items()}

        with torch.no_grad():
            text_emb = self.get_emb(self.embedding_model, text_input).squeeze().tolist()
        return text_emb
    
    def get_embeddings_batch(self, texts):
        """Get embeddings for a batch of texts"""
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

    def populate_vectors(self, batch_size=256):
        """
        Populate the vector store with embeddings from the dataset.
        
        Args:
            batch_size: Batch size for processing (adjust based on GPU memory)
        """
        start_time = time.time()
        current_key = self.collection.count()
        sorted_keys = sorted(self.dataset.keys())
        
        if current_key > 0:
            print(f"Already have {current_key} documents, continuing from there...")
            sorted_keys = sorted_keys[current_key:]
            start_idx = current_key
        else:
            start_idx = 0
        
        total = len(sorted_keys)
        print(f"Processing {total} documents, batch size: {batch_size}")
        
        batch_start_time = time.time()
        for i in range(0, total, batch_size):
            batch_keys = sorted_keys[i:i+batch_size]
            batch_texts = []
            batch_ids = []
            batch_metadatas = []
            
            for j, key in enumerate(batch_keys):
                count = start_idx + i + j
                value = self.dataset[key]
                batch_texts.append(value['text'])
                batch_ids.append(f'id_{count}')
                batch_metadatas.append({
                    'title': value['title'], 
                    'id': key, 
                    'change': False
                })
            
            batch_embeddings = self.get_embeddings_batch(batch_texts)
            
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                ids=batch_ids,
                metadatas=batch_metadatas
            )
            
            if (i + batch_size) % 500 < batch_size or i + batch_size >= total:
                processed = min(i + batch_size, total)
                elapsed = time.time() - batch_start_time
                speed = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / speed if speed > 0 else 0
                print(f"Processed: {start_idx + processed}/{start_idx + total} "
                      f"({100*processed/total:.1f}%) | "
                      f"Speed: {speed:.1f} docs/s | "
                      f"ETA: {eta/60:.1f} min")
        
        total_time = time.time() - start_time
        avg_speed = total / total_time if total_time > 0 else 0
        print(f"\n✓ Done! Total time: {total_time/60:.2f} min | Avg speed: {avg_speed:.1f} docs/s")

    def search_context(self, query, n_results):
        """Search for relevant documents given a query"""
        text_emb = self.get_embedding(query)
        return self.collection.query(
            query_embeddings=text_emb, 
            n_results=n_results, 
            include=['documents', 'metadatas', 'distances']
        )

    def update_context(self, id, str_add='', pos='end'):
        """Update a document's context in the collection"""
        id_data = self.get_id(id)
        
        if id_data['metadatas'][0]['change'] == True and str_add == '':
            # For clean_collection
            context = self.dataset[id_data['metadatas'][0]['id']]['text']
            flag = False
        elif id_data['metadatas'][0]['change'] == False and str_add == '':
            return True
        else:
            if pos == 'end':
                context = id_data['documents'][0] + ' ' + str_add
            elif pos == 'front':
                context = str_add + ' ' + id_data['documents'][0]
            else:
                context = id_data['documents'][0] + ' ' + str_add
            flag = True

        embeddings = self.get_embedding(context)
        metadata = {
            'title': self.dataset[id_data['metadatas'][0]['id']]['title'],
            'id': id_data['metadatas'][0]['id'],
            'change': flag
        }

        self.collection.update(
            ids=[id], 
            embeddings=[embeddings], 
            documents=[context], 
            metadatas=metadata
        )

    def clean_collect(self):
        """Clean the collection by reverting or deleting modified documents"""
        print('Starting collection cleanup...')
        results = self.collection.get(where={'change': True})
        
        if len(results['ids']) == 0:
            print('No data needs cleaning')
            return True
        
        print(f'Found {len(results["ids"])} documents to process')
        ids_to_delete = []
        ids_to_update = []
        
        for i in range(len(results['ids'])):
            doc_id = results['ids'][i]
            count = doc_id.split('_')[1]
            
            if int(count) >= len(self.dataset) or results['metadatas'][i]['id'] == ' ':
                ids_to_delete.append(doc_id)
            else:
                ids_to_update.append(doc_id)
        
        if ids_to_delete:
            print(f'Deleting {len(ids_to_delete)} invalid documents...')
            for doc_id in ids_to_delete:
                self.collection.delete(doc_id)
        
        if ids_to_update:
            print(f'Updating {len(ids_to_update)} documents...')
            for doc_id in ids_to_update:
                self.update_context(doc_id)
        
        print('Cleanup complete')
        return True

    def show_context(self):
        """Display collection info"""
        id = 'doc0'
        get_id = self.collection.get(ids=[id])
        print(get_id)

    def get_id(self, id):
        """Get a document by ID"""
        return self.collection.get(ids=[id])
    
    def inject_direct(self, text):
        """Directly inject a watermark document"""
        if self.count is None:
            self.count = self.collection.count()

        current_id = self.count

        self.collection.add(
            embeddings=[self.get_embedding(text)],
            documents=[text],
            ids=[f'id_{current_id}'],
            metadatas={'title': ' ', 'id': ' ', 'change': True}
        )
        self.count += 1

        return f'id_{current_id}'

    # def clean_vectors(self, batch_size=100):
    #     """Batch clean vector data"""
    #     total_count = self.collection.count()
    #     print(f'Starting vector cleanup, total {total_count} documents...')
        
    #     for i in range(0, total_count, batch_size):
    #         batch_end = min(i + batch_size, total_count)
            
    #         for count in range(i, batch_end):
    #             ids = f'id_{count}'
                
    #             if int(count) >= len(self.dataset):
    #                 self.collection.delete(ids)
    #             else:
    #                 self.update_context(ids)
            
    #         if (i + batch_size) % 500 < batch_size:
    #             print(f'Cleaned: {batch_end}/{total_count} ({100*batch_end/total_count:.1f}%)')

def check_collection_exists(collection_name, chroma_path=None):
    """Check if a collection exists"""
    chroma_client = chromadb.PersistentClient(path=str(chroma_path or ChromadbPath))
    collections = chroma_client.list_collections()
    return any(col.name == collection_name for col in collections)


def check_and_clean_existing_watermarks(vectorstore, expected_size, logger=None):
    """Check and clean existing watermark documents from the database."""
    def log(msg, level='info'):
        """Helper to log message using logger if available, otherwise print"""
        if logger:
            if level == 'warning':
                logger.warning(msg)
            elif level == 'error':
                logger.error(msg)
            else:
                logger.info(msg)
        else:
            print(msg)
    
    collection = vectorstore.collection

    try:
        all_data = collection.get()
        all_ids = all_data.get('ids', [])
    except Exception as e:
        log(f"Warning: Could not retrieve all IDs for watermark check: {e}", 'warning')
        log(f"Checking collection size...")
        vectorstore.count = collection.count()
        return vectorstore.count
    
    watermark_ids = []
    standard_id_pattern = re.compile(r'^id_(\d+)$')
    
    for doc_id in all_ids:
        match = standard_id_pattern.match(doc_id)
        if match:
            idx = int(match.group(1))
            if idx >= expected_size:
                watermark_ids.append(doc_id)
        else:
            watermark_ids.append(doc_id)
    
    if not watermark_ids:
        return expected_size

    collection.delete(ids=watermark_ids)
    log(f"Removed {len(watermark_ids)} existing watermark documents")

    # Finally check if size matches expected
    log(f"Checking collection size...")
    vectorstore.count = collection.count()
    if vectorstore.count != expected_size:
        raise ValueError(
            f"Vector database has insufficient records ({vectorstore.count} < {expected_size}) after cleaning. "
        )
    return vectorstore.count


def check_collection(collection_name, count=True, chroma_path=None):
    """Check if a collection exists and return its size"""
    chroma_client = chromadb.PersistentClient(path=str(chroma_path or ChromadbPath))
    collections = chroma_client.list_collections()
    print(collections)
    
    collection_exists = any(col.name == collection_name for col in collections)
    total_items = 0
    
    if collection_exists:
        collection = chroma_client.get_collection(name=collection_name)
        total_items = collection.count() if count else -1
        print(f"Collection exists, total: {total_items}")
        return True, total_items
    else:
        print(f"Collection not found: {collection_name}")
        return False, total_items


# # CLI entry point for standalone usage
# if __name__ == '__main__':
#     import argparse
#     from sentinelrag.utils import load_beir_datasets, load_models, load_json

#     def parse_arguments():
#         parser = argparse.ArgumentParser(description='Vector store management')
#         parser.add_argument('--access', type=int, default=0, help='access ids')
#         parser.add_argument('--ids', type=str, default='id_1434', help='access ids')
#         parser.add_argument("--eval_model_code", type=str, default="contriever",
#                           choices=["contriever", "contriever-msmarco", "ance"])
#         parser.add_argument('--eval_dataset', type=str, default='msmarco',
#                           choices=['trec-covid', 'nfcorpus', 'nq', 'msmarco', 'msmarco_200k', 'hotpotqa'])
#         parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
#         parser.add_argument('--score_function', type=str, default='cosine',
#                           choices=['cosine', 'l2', 'ip'])
#         parser.add_argument('--gpu_id', type=int, default=1, choices=[0, 1, 2, 3])
#         return parser.parse_args()

#     args = parse_arguments()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     if device == 'cuda':
#         torch.cuda.set_device(args.gpu_id)

#     collection_name = f"{args.eval_dataset}_{args.eval_model_code}_{args.score_function}"
#     print(collection_name)
    
#     model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    
#     if args.eval_dataset in ['msmarco', 'msmarco_200k']:
#         corpus, queries, qrels = load_beir_datasets(args.eval_dataset, 'train')
#     else:
#         corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

#     datalen = len(corpus)
#     collection_exist, collection_len = check_collection(collection_name)
#     print(collection_exist, datalen, collection_len)
    
#     if collection_exist and datalen == collection_len:
#         use_local = True
#         vectorstore = VectorStore(model, tokenizer, get_emb, corpus, device, 
#                                  collection_name, use_local)
#     else:
#         use_local = True
#         vectorstore = VectorStore(model, tokenizer, get_emb, corpus, device, 
#                                  collection_name, use_local)
#         vectorstore.populate_vectors()
    
#     # Test update and clean
#     vectorstore.update_context('id_3', 'ok,ok,ok')
#     result = vectorstore.get_id('id_3')
#     print(result)
#     vectorstore.clean_collect()
#     result = vectorstore.get_id('id_3')
#     print(result)

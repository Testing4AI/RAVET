#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core business logic for Watermark Interference Evaluation (CDPA/CIRA).
"""

import os
import random

import openai
import torch
from datasets import load_dataset
from tqdm import tqdm

from sentinelrag.utils import (
    load_beir_datasets,
    load_models,
    LLMClient,
)
from sentinelrag.rag import VectorStore, check_collection_exists, SimpleRAGVisitor, check_and_clean_existing_watermarks
from sentinelrag.prompts import PromptTemplates


class InterferenceEvaluator:
    """Evaluates watermark interference using retrieval and answer interference metrics."""

    def __init__(self, rag_llm_client: LLMClient, eval_llm_client: LLMClient, vectorstore: VectorStore, top_k: int = 5, 
                 collection_len: int = None, rag_llm_kwargs: dict = None, eval_llm_kwargs: dict = None):
        """
        Initialize the interference evaluator.

        Args:
            rag_llm_client: LLM client for generating answers from RAG.
            eval_llm_client: LLM client for semantic comparison and interference evaluation.
            vectorstore: Vector store for document retrieval.
            top_k: Number of documents to retrieve per query.
            collection_len: Length of the collection for dynamic retrieval limit.
            rag_llm_kwargs: Additional kwargs to pass to rag_llm_client.ask_llm().
            eval_llm_kwargs: Additional kwargs to pass to eval_llm_client.ask_llm().
        """
        self.rag_llm_client = rag_llm_client
        self.eval_llm_client = eval_llm_client
        self.vectorstore = vectorstore
        self.top_k = top_k
        # Dynamic retrieval limit: use collection_len if < 10000, else use 10000
        if collection_len is not None:
            self.retrieval_limit = min(collection_len, 10000)
            print(f"Retrieval limit set to {self.retrieval_limit} (collection_len={collection_len})")
        else:
            self.retrieval_limit = 1000  # Fallback to 1000 if collection_len not provided
            print(f"Retrieval limit set to {self.retrieval_limit} (fallback, collection_len not provided)")
        self.rag_llm_kwargs = rag_llm_kwargs or {}
        self.eval_llm_kwargs = eval_llm_kwargs or {}

    def retrieve_only(self, question: str) -> dict:
        """
        Retrieve documents without generating an answer.
        
        First retrieves documents up to retrieval_limit (min(collection_len, 10000)), 
        then takes top_k from those results.

        Args:
            question: The question to answer.

        Returns:
            Dictionary containing question, doc_ids, distances, rag_document.
            Returns None if retrieval fails.
        """
        try:
            # First retrieve documents using dynamic limit
            visitor = SimpleRAGVisitor(vectorstore=self.vectorstore, top_k=self.retrieval_limit)
            visitor.wm_unit = [question, "", ""]
            rag_document_1000, doc_ids_1000, distances_1000 = visitor.ask_wm()
            retrieved_docs_1000 = getattr(visitor, "last_documents", [])
            
            # Then slice to top_k
            doc_ids = doc_ids_1000[:self.top_k]
            distances = distances_1000[:self.top_k]
            retrieved_docs = retrieved_docs_1000[:self.top_k]
            
            # Rebuild rag_document from top_k documents
            rag_document = "\n".join(retrieved_docs) if retrieved_docs else ""

            return {
                "question": question,
                "doc_ids": doc_ids,
                "distances": distances,
                "doc_ids_1000": doc_ids_1000,  # Full 1000 docs for watermark tracking
                "distances_1000": distances_1000,
                "rag_document": rag_document,
                "answer": None,  # No answer generated
                "retrieved_documents": retrieved_docs,
            }
        except Exception as e:
            print(f"\nError retrieving for question '{question[:50]}...': {str(e)}")
            print("Skipping this question and continuing...")
            return None

    def retrieve_and_answer(self, question: str) -> dict:
        """
        Retrieve documents and generate answer for a question.
        
        First retrieves documents up to retrieval_limit (min(collection_len, 10000)),
        then takes top_k from those results.

        Args:
            question: The question to answer.

        Returns:
            Dictionary containing question, doc_ids, distances, rag_document, and answer.
            Returns None if API call fails.
        """
        try:
            # First retrieve documents using dynamic limit
            visitor = SimpleRAGVisitor(vectorstore=self.vectorstore, top_k=self.retrieval_limit)
            visitor.wm_unit = [question, "", ""]
            rag_document_1000, doc_ids_1000, distances_1000 = visitor.ask_wm()
            retrieved_docs_1000 = getattr(visitor, "last_documents", [])
            
            # Then slice to top_k
            doc_ids = doc_ids_1000[:self.top_k]
            distances = distances_1000[:self.top_k]
            retrieved_docs = retrieved_docs_1000[:self.top_k]
            
            # Rebuild rag_document from top_k documents
            rag_document = "\n".join(retrieved_docs) if retrieved_docs else ""

            prompt = PromptTemplates.answer_with_rag(rag_document, question)
            answer = self.rag_llm_client.ask_llm(prompt, **self.rag_llm_kwargs)

            return {
                "question": question,
                "doc_ids": doc_ids,
                "distances": distances,
                "doc_ids_1000": doc_ids_1000,  # Full 1000 docs for watermark tracking
                "distances_1000": distances_1000,
                "rag_document": rag_document,
                "answer": answer,
                "retrieved_documents": retrieved_docs,
            }
        except Exception as e:
            print(f"\nError processing question '{question[:50]}...': {str(e)}")
            print("Skipping this question and continuing...")
            return None

    def semantic_equivalence(self, answer_a: str, answer_b: str) -> bool:
        """
        Check if two answers are semantically equivalent.

        Args:
            answer_a: First answer.
            answer_b: Second answer.

        Returns:
            True if answers are semantically equivalent, False otherwise.
        """
        answer_a = answer_a or ""
        answer_b = answer_b or ""
        a_norm = answer_a.strip().lower()
        b_norm = answer_b.strip().lower()

        if not a_norm or not b_norm:
            return False
        if a_norm == b_norm:
            return True
        if "cannot answer" in a_norm and "cannot answer" in b_norm:
            return True

        prompt = (
            "You are a strict semantic judge. Compare the meaning of Answer A and Answer B.\n"
            f"Answer A: {answer_a}\n"
            f"Answer B: {answer_b}\n"
            "\n"
            "Two answers count as having the SAME meaning if:\n"
            "- Their core factual claims match, even if the wording differs.\n"
            "- One answer adds minor details that do not contradict or change the main meaning.\n"
            "- They are paraphrases that express the same idea.\n"
            "\n"
            "They count as DIFFERENT if:\n"
            "- Any key fact, claim, or implication differs.\n"
            "- One answer contradicts or reverses the meaning of the other.\n"
            "- One introduces a significant new idea that changes the meaning.\n"
            "\n"
            "Your output MUST BE EXACTLY one token: 'yes' or 'no'.\n"
        )
        try:
            result = self.eval_llm_client.ask_llm(prompt, **self.eval_llm_kwargs).strip().lower()
            return result.startswith("yes")
        except Exception as e:
            print(f"\nError in semantic equivalence check: {str(e)}")
            print("Assuming answers are different due to evaluation failure...")
            return False

    def retrieve_batch(self, questions: list, desc: str = "Retrieving documents") -> list:
        """
        Batch retrieve documents for all questions without generating answers.

        Args:
            questions: List of (question_id, question_text) tuples.
            desc: Description for progress bar.

        Returns:
            List of retrieval result dictionaries (answer field is None).
        """
        runs = []
        skipped_count = 0
        for qid, question in tqdm(questions, desc=desc, total=len(questions)):
            result = self.retrieve_only(question)
            if result is None:
                skipped_count += 1
                continue
            result["question_id"] = qid
            runs.append(result)
        if skipped_count > 0:
            print(f"\nSkipped {skipped_count} questions during retrieval")
        return runs

    def generate_answers_batch(self, retrieval_runs: list, desc: str = "Generating answers") -> list:
        """
        Generate answers for a batch of retrieval results.

        Args:
            retrieval_runs: List of retrieval results with rag_document but no answer.
            desc: Description for progress bar.

        Returns:
            Updated list with answers generated.
        """
        completed_runs = []
        skipped_count = 0
        
        for run in tqdm(retrieval_runs, desc=desc, total=len(retrieval_runs)):
            try:
                question = run.get("question")
                rag_document = run.get("rag_document")
                prompt = PromptTemplates.answer_with_rag(rag_document, question)
                answer = self.rag_llm_client.ask_llm(prompt, **self.rag_llm_kwargs)
                run["answer"] = answer
                completed_runs.append(run)
            except Exception as e:
                print(f"\nError generating answer for question ID {run.get('question_id')}: {str(e)}")
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            print(f"\nSkipped {skipped_count} questions during answer generation")
        return completed_runs

    def evaluate_questions(self, questions: list) -> list:
        """
        Evaluate questions and collect retrieval/answer results.

        Args:
            questions: List of (question_id, question_text) tuples.

        Returns:
            List of result dictionaries for each question (skips failed questions).
        """
        runs = []
        skipped_count = 0
        for qid, question in tqdm(questions, desc="Evaluating questions", total=len(questions)):
            result = self.retrieve_and_answer(question)
            if result is None:
                skipped_count += 1
                continue
            result["question_id"] = qid
            runs.append(result)
        if skipped_count > 0:
            print(f"\nSkipped {skipped_count} questions due to API failures")
        return runs

    def evaluate_fully_optimized(self, questions: list, watermark_ids: list = None) -> tuple:
        """
        Fully optimized evaluation strategy:
        1. Retrieve all clean documents (no answers yet)
        2. Inject watermarks happens externally
        3. Retrieve all watermarked documents (no answers yet)
        4. Compare retrievals
        5. Generate answers only for questions that need them:
           - Identical retrieval + no watermarks: generate once for both
           - Different retrieval or has watermarks: generate separately

        Args:
            questions: List of (question_id, question_text) tuples.
            watermark_ids: List of injected watermark IDs.

        Returns:
            Tuple of (clean_runs, watermarked_runs, stats).
        """
        watermark_ids_set = set(watermark_ids) if watermark_ids else set()
        
        # Phase 1: Retrieve all clean documents
        print("\nPhase 1: Clean retrieval...")
        clean_retrievals = self.retrieve_batch(questions, "Retrieving (clean)")
        
        # Create lookup
        clean_lookup = {run.get("question_id"): run for run in clean_retrievals}
        
        # Note: Watermark injection happens externally here
        
        return clean_retrievals, clean_lookup
    
    def complete_evaluation_after_injection(self, questions: list, clean_retrievals: list, 
                                           watermark_ids: list = None) -> tuple:
        """
        Complete evaluation after watermarks have been injected.
        
        Args:
            questions: List of (question_id, question_text) tuples.
            clean_retrievals: Results from clean retrieval phase.
            watermark_ids: List of injected watermark IDs.
            
        Returns:
            Tuple of (clean_runs, watermarked_runs).
        """
        watermark_ids_set = set(watermark_ids) if watermark_ids else set()
        clean_lookup = {run.get("question_id"): run for run in clean_retrievals}
        
        # Phase 2: Retrieve all watermarked documents
        print("\nPhase 2: Watermarked retrieval...")
        watermarked_retrievals = self.retrieve_batch(questions, "Retrieving (watermarked)")
        
        # Track watermark positions in top 1000 docs
        print("\nTracking watermark positions in top 1000 docs...")
        for wm_run in watermarked_retrievals:
            doc_ids_1000 = wm_run.get("doc_ids_1000", [])
            distances_1000 = wm_run.get("distances_1000", [])
            
            # Find first watermark in top 1000
            watermark_rank = None
            watermark_distance = None
            
            if watermark_ids_set and doc_ids_1000:
                for idx, doc_id in enumerate(doc_ids_1000):
                    if doc_id in watermark_ids_set:
                        watermark_rank = idx + 1  # 1-indexed rank
                        watermark_distance = distances_1000[idx] if idx < len(distances_1000) else None
                        break
            
            # Add to the run
            wm_run["watermark_rank"] = watermark_rank
            wm_run["watermark_distance"] = watermark_distance
        
        # Phase 3: Compare retrievals and categorize questions
        print("\nPhase 3: Comparing retrievals...")
        identical_no_wm = []  # Identical retrieval, no watermarks
        needs_separate_generation = []  # Different retrieval or has watermarks
        
        for wm_run in watermarked_retrievals:
            qid = wm_run.get("question_id")
            clean_run = clean_lookup.get(qid)
            
            if clean_run:
                clean_ids = clean_run.get("doc_ids") or []
                wm_ids = wm_run.get("doc_ids") or []
                same_docs = list(clean_ids) == list(wm_ids)
                has_watermark = bool(watermark_ids_set.intersection(wm_ids)) if watermark_ids_set else False
                
                if same_docs and not has_watermark:
                    identical_no_wm.append((clean_run, wm_run))
                else:
                    needs_separate_generation.append((clean_run, wm_run))
            else:
                needs_separate_generation.append((None, wm_run))
        
        print(f"Identical retrieval (no watermarks): {len(identical_no_wm)}")
        print(f"Needs separate generation: {len(needs_separate_generation)}")
        
        # Phase 4: Generate answers strategically
        print("\nPhase 4: Generating answers...")
        
        # For identical retrievals with no watermarks, skip answer generation entirely
        # Mark them as auto-preserved (will be treated as no interference in metrics)
        identical_clean_runs = []
        for clean_run, wm_run in identical_no_wm:
            clean_run["answer"] = None
            clean_run["auto_preserved"] = True
            wm_run["answer"] = None
            wm_run["auto_preserved"] = True
            wm_run["answer_reused"] = True
            identical_clean_runs.append(clean_run)
        
        print(f"Skipped answer generation for {len(identical_no_wm)} identical retrievals (auto-preserved)")
        
        # For different retrievals, generate separately
        clean_to_generate = [clean_run for clean_run, _ in needs_separate_generation if clean_run]
        wm_to_generate = [wm_run for _, wm_run in needs_separate_generation]
        
        print(f"Generating clean answers: {len(clean_to_generate)}")
        separate_clean_runs = self.generate_answers_batch(clean_to_generate, "Generating (clean)")
        
        print(f"Generating watermarked answers: {len(wm_to_generate)}")
        separate_wm_runs = self.generate_answers_batch(wm_to_generate, "Generating (watermarked)")
        
        # Mark watermarked runs that were generated separately
        for wm_run in separate_wm_runs:
            wm_run["answer_reused"] = False
        
        # Combine all results
        all_clean_runs = identical_clean_runs + separate_clean_runs
        all_wm_runs = [wm_run for _, wm_run in identical_no_wm] + separate_wm_runs
        
        return all_clean_runs, all_wm_runs
    
    def evaluate_watermarked_optimized(self, questions: list, clean_runs: list, watermark_ids: list = None) -> tuple:
        """
        Optimized watermarked evaluation that reuses clean answers when possible.
        
        Only generates new answers when:
        - Retrieval differs from clean run, OR
        - Watermark documents are in retrieved results

        Args:
            questions: List of (question_id, question_text) tuples.
            clean_runs: Results from clean evaluation.
            watermark_ids: List of injected watermark IDs.

        Returns:
            Tuple of (watermarked_runs, stats) where stats contains optimization metrics.
        """
        watermark_ids_set = set(watermark_ids) if watermark_ids else set()
        watermarked_runs = []
        skipped_count = 0
        reused_answers = 0
        generated_answers = 0
        
        # Create lookup for clean runs
        clean_lookup = {run.get("question_id"): run for run in clean_runs}
        
        for qid, question in tqdm(questions, desc="Evaluating watermarked (optimized)", total=len(questions)):
            # First, only retrieve documents
            result = self.retrieve_only(question)
            if result is None:
                skipped_count += 1
                continue
                
            result["question_id"] = qid
            
            # Check if we can reuse the clean answer
            clean_run = clean_lookup.get(qid)
            if clean_run:
                clean_ids = clean_run.get("doc_ids") or []
                wm_ids = result.get("doc_ids") or []
                same_docs = list(clean_ids) == list(wm_ids)
                has_watermark = bool(watermark_ids_set.intersection(wm_ids)) if watermark_ids_set else False
                
                # Reuse clean answer if retrieval identical and no watermarks
                if same_docs and not has_watermark:
                    result["answer"] = clean_run.get("answer")
                    result["answer_reused"] = True
                    reused_answers += 1
                    watermarked_runs.append(result)
                    continue
            
            # Need to generate new answer
            try:
                prompt = PromptTemplates.answer_with_rag(result["rag_document"], question)
                answer = self.rag_llm_client.ask_llm(prompt, **self.rag_llm_kwargs)
                result["answer"] = answer
                result["answer_reused"] = False
                generated_answers += 1
            except Exception as e:
                print(f"\nError generating answer for '{question[:50]}...': {str(e)}")
                skipped_count += 1
                continue
                
            watermarked_runs.append(result)
        
        if skipped_count > 0:
            print(f"\nSkipped {skipped_count} questions due to failures")
        
        return watermarked_runs, stats

    def compute_interference(self, clean_runs: list, watermarked_runs: list, watermark_ids: list = None) -> tuple:
        """
        Compute retrieval and answer interference metrics.

        Args:
            clean_runs: List of results from clean vectorstore.
            watermarked_runs: List of results from watermarked vectorstore.
            watermark_ids: List of injected watermark IDs (optional).

        Returns:
            Tuple of (answer_interference, retrieval_interference, details) where details is a list of per-question results.
        """
        if len(clean_runs) != len(watermarked_runs):
            raise ValueError("Clean and watermarked runs must have the same number of questions")

        watermark_ids_set = set(watermark_ids) if watermark_ids else set()
        total = len(clean_runs)
        consistent_answers = 0
        consistent_retrievals = 0
        details = []

        for clean, wm in zip(clean_runs, watermarked_runs):
            clean_ids = clean.get("doc_ids") or []
            wm_ids = wm.get("doc_ids") or []
            same_docs = list(clean_ids) == list(wm_ids)
            consistent_retrievals += int(same_docs)

            # Check if retrieved docs contain watermarks (always compute this)
            has_watermark = bool(watermark_ids_set.intersection(wm_ids)) if watermark_ids_set else False

            # Check if this was auto-preserved (identical retrieval, no watermarks, no answer generation)
            if clean.get("auto_preserved") or wm.get("auto_preserved"):
                answers_consistent = True
                answer_check_skipped = True
            else:
                # If retrieval is consistent and no watermarks in retrieved docs, skip answer check
                if same_docs and not has_watermark:
                    answers_consistent = True
                    answer_check_skipped = True
                else:
                    answers_consistent = self.semantic_equivalence(
                        clean.get("answer", ""), wm.get("answer", "")
                    )
                    answer_check_skipped = False
            
            consistent_answers += int(answers_consistent)

            details.append(
                {
                    "question_id": clean.get("question_id"),
                    "question": clean.get("question"),
                    "retrieval_interfered": not same_docs,
                    "answer_interfered": not answers_consistent,
                    "answer_check_skipped": answer_check_skipped,
                    "has_watermark_in_retrieval": has_watermark,
                    "clean": {
                        "doc_ids": clean.get("doc_ids", []),
                        "answer": clean.get("answer", ""),
                    },
                    "watermarked": {
                        "doc_ids": wm.get("doc_ids", []),
                        "answer": wm.get("answer", ""),
                    },
                }
            )

        answer_consistency = consistent_answers / total if total else 0.0
        retrieval_consistency = consistent_retrievals / total if total else 0.0
        answer_interference = 1 - answer_consistency
        retrieval_interference = 1 - retrieval_consistency
        return answer_interference, retrieval_interference, details

    def inject_watermarks(self, watermark_texts: list) -> list:
        """
        Inject watermark documents into vector store.

        Args:
            watermark_texts: List of watermark document texts.

        Returns:
            List of injected watermark IDs.
        """
        watermark_ids = []
        print("Injecting watermark documents...", end=" ")
        for text in watermark_texts:
            wid = self.vectorstore.inject_direct(text)
            watermark_ids.append(wid)
        print(f"completed ({len(watermark_ids)})")
        return watermark_ids

    def cleanup_watermarks(self, watermark_ids: list) -> bool:
        """
        Remove injected watermarks from vector store.

        Args:
            watermark_ids: List of watermark IDs to remove.

        Returns:
            True if cleanup was successful, False otherwise.
        """
        if not watermark_ids:
            return True

        print("Cleaning up injected watermarks...")
        try:
            self.vectorstore.collection.delete(ids=watermark_ids)
            print("Watermark cleanup completed")
            return True
        except Exception as exc:
            print(f"Failed to clean up watermarks: {exc}")
            return False


def setup_llm_client(api_key: str, llm_url: str, model: str) -> LLMClient:
    """
    Create LLM client.

    Args:
        api_key: API key for OpenAI-compatible API.
        llm_url: Base URL for the API.
        model: Model name to use.

    Returns:
        Configured LLMClient instance.
    """
    client = openai.OpenAI(api_key=api_key, base_url=llm_url)
    return LLMClient(client, model)


def setup_vectorstore(
    eval_dataset: str,
    eval_model_code: str,
    score_function: str = "cosine",
    split: str = "test",
    gpu_id: int = 0,
    device: str = None,
) -> tuple:
    """
    Setup vector store for interference evaluation.

    Args:
        eval_dataset: Dataset name.
        eval_model_code: Retrieval model code.
        score_function: Similarity function.
        split: Dataset split.
        gpu_id: GPU device ID.
        device: Device string (overrides gpu_id if provided).

    Returns:
        Tuple of (vectorstore, collection_name, collection_len).
    """
    collection_name = f"{eval_dataset}_{eval_model_code}_{score_function}"

    model, _, tokenizer, get_emb = load_models(eval_model_code)
    collection_exist = check_collection_exists(collection_name)

    if not collection_exist:
        raise ValueError(
            f"Vector database does not exist. Please build it first:\n"
            f"sentinelrag-build-chroma --eval_dataset {eval_dataset} --eval_model_code {eval_model_code}"
        )

    corpus, _, _ = load_beir_datasets(eval_dataset, split)
    datalen = len(corpus)
    corpus_for_vectorstore = corpus  # Store for later use

    if device is None:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu"

    vectorstore = VectorStore(
        model, tokenizer, get_emb, corpus_for_vectorstore, device, collection_name, use_local=True
    )
    

    collection_len = check_and_clean_existing_watermarks(vectorstore, datalen)
    
    return vectorstore, collection_name, collection_len


def load_main_questions(
    eval_dataset: str, split: str, max_questions: int, seed: int
) -> list:
    """
    Load main-task questions from dataset.

    Args:
        eval_dataset: Dataset name.
        split: Dataset split. Use "full" to combine train/dev/test queries.
        max_questions: Maximum number of questions to load (0 for all).
        seed: Random seed for shuffling.

    Returns:
        List of (question_id, question_text) tuples.
    """
    rng = random.Random(seed)
    if eval_dataset == "closed_qa":
        train_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        closed_qa_dataset = train_dataset.filter(lambda example: example["category"] == "closed_qa")
        pairs = [(str(i), item["instruction"]) for i, item in enumerate(closed_qa_dataset)]
    else:
        if split == "full":
            seen = set()
            pairs = []
            for current_split in ("train", "dev", "test"):
                try:
                    _, queries, _ = load_beir_datasets(eval_dataset, current_split)
                except Exception:
                    continue

                for qid, qtext in queries.items():
                    key = (qid, qtext)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append((qid, qtext))

            if not pairs:
                raise ValueError(
                    f"No queries could be loaded for dataset '{eval_dataset}' with split='full'."
                )
        else:
            _, queries, _ = load_beir_datasets(eval_dataset, split)
            pairs = list(queries.items())

    # rng.shuffle(pairs)
    if max_questions > 0:
        pairs = pairs[:max_questions]
    return pairs

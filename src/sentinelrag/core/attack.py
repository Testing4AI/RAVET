#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack module for testing watermark robustness.

Implements two attack types:
1. Unrelated Content Removal - LLM-based filtering of retrieved docs
2. Knowledge Expansion Attack - Test detection with varying k values
"""

import os
import random
from datetime import datetime

from sentinelrag.utils import save_json, binomial_test_greater
from sentinelrag.core.detector import (
    RAGWatermarkDetector,
    inject_watermarks,
    cleanup_watermarks,
    generate_all_questions_async,
    calculate_overall_results,
    _is_answer_unanswerable,
    answer_contains_entity_pair,
)


# Prompt for unrelated content removal attack
UNRELATED_CONTENT_REMOVAL_PROMPT = """You are a quality assurance filter. The text below may contain "hallucinated" or injected noise sentences that do not fit the linguistic style or logical flow of the document.

Text: {text}

Task: Rewrite the text to improve flow. Aggressively remove any sentences that seem stylistically inconsistent, logically disjointed, or appear to be random insertions. Keep the text professional and coherent."""


async def remove_unrelated_content(text: str, llm_client, llm_kwargs: dict = None) -> str:
    """
    Filter out unrelated/injected content using LLM.
    
    Args:
        text: The text to filter
        llm_client: LLM client for filtering
        llm_kwargs: Additional kwargs for LLM client
        
    Returns:
        Filtered text with unrelated content removed
    """
    if llm_kwargs is None:
        llm_kwargs = {}
    
    prompt = UNRELATED_CONTENT_REMOVAL_PROMPT.format(text=text)
    
    try:
        filtered_text = await llm_client.ask_llm_async(prompt, **llm_kwargs)
        return filtered_text
    except Exception as e:
        # If filtering fails, return original text
        print(f"Error filtering content: {e}")
        return text


def remove_unrelated_content_sync(text: str, llm_client, llm_kwargs: dict = None) -> str:
    """
    Synchronous version of remove_unrelated_content.
    """
    if llm_kwargs is None:
        llm_kwargs = {}
    
    prompt = UNRELATED_CONTENT_REMOVAL_PROMPT.format(text=text)
    
    try:
        filtered_text = llm_client.ask_llm(prompt, **llm_kwargs)
        return filtered_text
    except Exception as e:
        print(f"Error filtering content: {e}")
        return text


async def perform_unrelated_content_attack(
    full_dataset, vectorstore, llm_client, rllm_client, injection_result,
    top_k, questions_per_ko, question_generation_mode,
    p0, alpha, basepath, logger, llm_kwargs: dict = None, rllm_kwargs: dict = None,
    watermark_query_count=None
):
    """
    Perform unrelated content removal attack.
    
    Attack flow:
    1. Inject watermarks to database
    2. For each question, retrieve docs and filter through LLM
    3. Perform watermark detection on filtered content
    
    Args:
        full_dataset: The full dataset
        vectorstore: Vector store instance
        llm_client: LLM client for question generation and filtering
        rllm_client: LLM client for RAG answer generation
        injection_result: Injection result dictionary
        top_k: Number of top-k documents to retrieve
        questions_per_ko: Number of questions per KO
        question_generation_mode: Question generation mode
        p0: Null hypothesis probability
        alpha: Significance level
        basepath: Base path for saving results
        logger: Logger instance
        llm_kwargs: Additional kwargs for LLM client
        rllm_kwargs: Additional kwargs for RLLM client
        watermark_query_count: Optional number of KOs to sample
        
    Returns:
        Dictionary with attack results
    """
    selected_kos = injection_result['selected_kos']
    watermark_texts = injection_result['watermark_texts']
    
    # Step 1: Inject watermarks
    watermark_ids = inject_watermarks(vectorstore, watermark_texts, logger)
    logger.info(f"Injected {len(watermark_ids)} watermark documents for attack")
    
    # Step 2: Sample KOs if requested
    if watermark_query_count and watermark_query_count < len(selected_kos):
        logger.info(f"Sampling {watermark_query_count} KOs from {len(selected_kos)} for attack...")
        indices = random.sample(range(len(selected_kos)), watermark_query_count)
        sampled_kos = [selected_kos[i] for i in indices]
        sampled_texts = [watermark_texts[i] for i in indices]
    else:
        sampled_kos = selected_kos
        sampled_texts = watermark_texts
    
    # Step 3: Generate questions
    detector = RAGWatermarkDetector(
        full_dataset, vectorstore, llm_client, rllm_client,
        top_k=top_k, questions_per_ko=questions_per_ko,
        llm_kwargs=llm_kwargs, rllm_kwargs=rllm_kwargs
    )
    
    logger.info("Generating questions for attack...")
    all_questions = await generate_all_questions_async(
        detector, sampled_kos, sampled_texts, questions_per_ko,
        question_generation_mode, logger
    )
    
    if not all_questions:
        raise ValueError("Failed to generate any questions for attack")
    
    # Step 4: Run attack - retrieve docs, filter, then detect
    per_ko_results = []
    attack_records = []
    
    for idx, ko in enumerate(sampled_kos):
        start_idx = idx * questions_per_ko
        end_idx = min(start_idx + questions_per_ko, len(all_questions))
        ko_questions = all_questions[start_idx:end_idx]
        
        correct_count = 0
        
        for q_idx, question in enumerate(ko_questions):
            # Get RAG answer (this also retrieves documents internally)
            original_answer = detector._answer_question_with_rag(question, verbose=False)
            
            # Get the retrieved document from the last retrieval
            if detector.retrieved_documents:
                last_retrieved = detector.retrieved_documents[-1]
                retrieved_doc = last_retrieved.get("retrieved_document", "")
            else:
                retrieved_doc = ""
            
            # Filter the retrieved document through LLM
            filtered_doc = remove_unrelated_content_sync(retrieved_doc, llm_client, llm_kwargs)
            
            logger.info(f"Attack - KO #{idx+1} Q{q_idx+1}: {question}")
            logger.info(f"Attack - Original doc length: {len(retrieved_doc)}, Filtered doc length: {len(filtered_doc)}")
            
            # Generate answer from filtered document
            from sentinelrag.prompts import PromptTemplates
            prompt = PromptTemplates.answer_with_rag(filtered_doc, question)
            filtered_answer = rllm_client.ask_llm(prompt, **(rllm_kwargs or {}))
            
            logger.info(f"Attack - Original answer: {original_answer[:100]}..." if len(original_answer) > 100 else f"Attack - Original answer: {original_answer}")
            logger.info(f"Attack - Filtered answer: {filtered_answer[:100]}..." if len(filtered_answer) > 100 else f"Attack - Filtered answer: {filtered_answer}")
            
            # Verify answer
            if _is_answer_unanswerable(filtered_answer):
                is_correct = False
            else:
                is_correct = detector._verify_answer(question, filtered_answer, ko, verbose=False)
            
            logger.info(f"Attack - Verification: {'Correct' if is_correct else 'Incorrect'}")
            
            if is_correct:
                correct_count += 1
            
            attack_records.append({
                "ko_index": idx,
                "question_index": q_idx,
                "question": question,
                "original_doc_length": len(retrieved_doc),
                "filtered_doc_length": len(filtered_doc),
                "original_answer": original_answer,
                "filtered_answer": filtered_answer,
                "is_correct": is_correct
            })
        
        rate = correct_count / max(1, len(ko_questions))
        p_val = binomial_test_greater(correct_count, len(ko_questions), p0=p0)
        
        per_ko_results.append({
            "num_questions": len(ko_questions),
            "num_correct": correct_count,
            "detection_rate": rate,
            "p_value": p_val,
            "questions": ko_questions
        })
    
    # Calculate overall results
    overall = calculate_overall_results(per_ko_results, p0, alpha)
    
    # Clean up watermarks
    cleanup_watermarks(vectorstore, watermark_ids, True, logger)
    
    result = {
        "attack_type": "unrelated_content_removal",
        "attack_metadata": {
            "timestamp": datetime.now().isoformat(),
            "top_k": top_k,
            "questions_per_ko": questions_per_ko,
            "question_generation_mode": question_generation_mode,
            "num_kos_tested": len(sampled_kos),
            "total_questions": len(all_questions),
            "p0": p0,
            "alpha": alpha
        },
        "per_ko": per_ko_results,
        "overall": overall,
        "attack_records": attack_records
    }
    
    # Save results
    if basepath:
        result_path = os.path.join(basepath, 'unrelated_content_attack_results.json')
        save_json(result, result_path)
        logger.info(f"Attack results saved to: {result_path}")
    
    return result


async def perform_knowledge_expansion_attack(
    full_dataset, vectorstore, llm_client, rllm_client, injection_result,
    max_k=50, k_increments=None,
    questions_per_ko=1, question_generation_mode="watermark_text_based",
    p0=0.01, alpha=0.05, basepath=None, logger=None,
    llm_kwargs: dict = None, rllm_kwargs: dict = None,
    watermark_query_count=None
):
    """
    Perform knowledge expansion attack.
    
    Attack flow:
    1. Inject watermarks to database
    2. Retrieve with high k (max_k)
    3. For each k in k_increments, use only top-k docs for detection
    4. Return detection results for each k value
    
    Args:
        full_dataset: The full dataset
        vectorstore: Vector store instance
        llm_client: LLM client for question generation
        rllm_client: LLM client for RAG answer generation
        injection_result: Injection result dictionary
        max_k: Maximum k for retrieval (default: 50)
        k_increments: List of k values to test (default: [10, 20, 30, 40, 50])
        questions_per_ko: Number of questions per KO
        question_generation_mode: Question generation mode
        p0: Null hypothesis probability
        alpha: Significance level
        basepath: Base path for saving results
        logger: Logger instance
        llm_kwargs: Additional kwargs for LLM client
        rllm_kwargs: Additional kwargs for RLLM client
        watermark_query_count: Optional number of KOs to sample
        
    Returns:
        Dictionary with attack results for each k value
    """
    if k_increments is None:
        k_increments = [10, 20, 30, 40, 50]
    
    selected_kos = injection_result['selected_kos']
    watermark_texts = injection_result['watermark_texts']
    
    # Step 1: Inject watermarks
    watermark_ids = inject_watermarks(vectorstore, watermark_texts, logger)
    logger.info(f"Injected {len(watermark_ids)} watermark documents for knowledge expansion attack")
    
    # Step 2: Sample KOs if requested
    if watermark_query_count and watermark_query_count < len(selected_kos):
        logger.info(f"Sampling {watermark_query_count} KOs from {len(selected_kos)} for attack...")
        indices = random.sample(range(len(selected_kos)), watermark_query_count)
        sampled_kos = [selected_kos[i] for i in indices]
        sampled_texts = [watermark_texts[i] for i in indices]
    else:
        sampled_kos = selected_kos
        sampled_texts = watermark_texts
    
    # Step 3: Generate questions (once, using max_k for retrieval)
    detector = RAGWatermarkDetector(
        full_dataset, vectorstore, llm_client, rllm_client,
        top_k=max_k, questions_per_ko=questions_per_ko,
        llm_kwargs=llm_kwargs, rllm_kwargs=rllm_kwargs
    )
    
    logger.info(f"Generating questions with max_k={max_k}...")
    all_questions = await generate_all_questions_async(
        detector, sampled_kos, sampled_texts, questions_per_ko,
        question_generation_mode, logger
    )
    
    if not all_questions:
        raise ValueError("Failed to generate any questions for knowledge expansion attack")
    
    # Step 4: For each question, retrieve with max_k and store all docs
    logger.info(f"Retrieving documents with max_k={max_k}...")
    all_retrievals = []
    
    for q_idx, question in enumerate(all_questions):
        # Use the visitor directly to get full retrieval info
        detector.visitor.wm_unit = [question, "", ""]
        rag_result = detector.visitor.ask_wm()
        
        if len(rag_result) == 3:
            rag_document, db_ids, distances = rag_result
        else:
            rag_document, db_ids = rag_result
            distances = []
        
        all_retrievals.append({
            "question": question,
            "full_document": rag_document,
            "document_ids": db_ids,
            "distances": distances
        })
    
    # Step 5: Test detection for each k value
    results_by_k = {}
    
    from sentinelrag.prompts import PromptTemplates
    
    for k in k_increments:
        logger.info(f"Testing detection with k={k}...")
        
        per_ko_results = []
        k_records = []
        
        for idx, ko in enumerate(sampled_kos):
            start_idx = idx * questions_per_ko
            end_idx = min(start_idx + questions_per_ko, len(all_questions))
            
            correct_count = 0
            
            for q_offset in range(end_idx - start_idx):
                global_q_idx = start_idx + q_offset
                question = all_questions[global_q_idx]
                retrieval = all_retrievals[global_q_idx]
                
                # Use only top-k documents
                # The document is concatenated, so we need to re-retrieve with k limit
                # For simplicity, we'll create a truncated context
                full_doc = retrieval["full_document"]
                doc_ids = retrieval["document_ids"][:k] if len(retrieval["document_ids"]) > k else retrieval["document_ids"]
                
                # Re-generate answer with potentially fewer docs
                # Create a detector with k as top_k
                k_detector = RAGWatermarkDetector(
                    full_dataset, vectorstore, llm_client, rllm_client,
                    top_k=k, questions_per_ko=questions_per_ko,
                    llm_kwargs=llm_kwargs, rllm_kwargs=rllm_kwargs
                )
                
                answer = k_detector._answer_question_with_rag(question, verbose=False)
                
                logger.info(f"K={k} - KO #{idx+1} Q{q_offset+1}: {question}")
                logger.info(f"K={k} - Answer: {answer[:100]}..." if len(answer) > 100 else f"K={k} - Answer: {answer}")
                
                # Verify answer
                if _is_answer_unanswerable(answer):
                    is_correct = False
                else:
                    is_correct = k_detector._verify_answer(question, answer, ko, verbose=False)
                
                logger.info(f"K={k} - Verification: {'Correct' if is_correct else 'Incorrect'}")
                
                if is_correct:
                    correct_count += 1
                
                k_records.append({
                    "ko_index": idx,
                    "question_index": q_offset,
                    "question": question,
                    "answer": answer,
                    "is_correct": is_correct,
                    "k": k
                })
            
            rate = correct_count / max(1, end_idx - start_idx)
            p_val = binomial_test_greater(correct_count, end_idx - start_idx, p0=p0)
            
            per_ko_results.append({
                "num_questions": end_idx - start_idx,
                "num_correct": correct_count,
                "detection_rate": rate,
                "p_value": p_val
            })
        
        overall = calculate_overall_results(per_ko_results, p0, alpha)
        
        results_by_k[k] = {
            "k": k,
            "per_ko": per_ko_results,
            "overall": overall,
            "records": k_records
        }
    
    # Clean up watermarks
    cleanup_watermarks(vectorstore, watermark_ids, True, logger)
    
    result = {
        "attack_type": "knowledge_expansion",
        "attack_metadata": {
            "timestamp": datetime.now().isoformat(),
            "max_k": max_k,
            "k_increments": k_increments,
            "questions_per_ko": questions_per_ko,
            "question_generation_mode": question_generation_mode,
            "num_kos_tested": len(sampled_kos),
            "total_questions": len(all_questions),
            "p0": p0,
            "alpha": alpha
        },
        "results_by_k": results_by_k
    }
    
    # Save results
    if basepath:
        result_path = os.path.join(basepath, 'knowledge_expansion_attack_results.json')
        save_json(result, result_path)
        logger.info(f"Knowledge expansion attack results saved to: {result_path}")
    
    return result

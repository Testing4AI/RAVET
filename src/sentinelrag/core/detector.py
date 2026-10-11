#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG-based watermark detection module
"""

import json
import os
import re
import time
from datetime import datetime

from sentinelrag.prompts import PromptTemplates
from sentinelrag.rag import SimpleRAGVisitor
from sentinelrag.utils import binomial_test_greater, save_json
from sentinelrag.utils.stats import calculate_neg_log_p
import math


def _normalize_entity(entity):
    """Lowercase helper that also converts underscores to spaces."""
    return str(entity).replace("_", " ").lower()


def answer_contains_entity_pair(answer, ko):
    """Check whether the answer mentions any (entity1, entity2) pair from the KO."""
    answer_l = answer.lower()
    triplets = ko.get("triplets", [])
    for triplet in triplets:
        entity1 = triplet.get("subject") or triplet.get("entity1")
        entity2 = triplet.get("object") or triplet.get("entity2")
        if not entity1 or not entity2:
            continue
        raw1 = str(entity1).lower()
        raw2 = str(entity2).lower()
        ent1_norm = _normalize_entity(entity1)
        ent2_norm = _normalize_entity(entity2)
        hit_raw = raw1 in answer_l or raw2 in answer_l
        hit_norm = ent1_norm and ent2_norm and (ent1_norm in answer_l or ent2_norm in answer_l)
        if hit_raw or hit_norm:
            return True
    return False


def _is_answer_unanswerable(answer):
    """Check if the answer indicates the question cannot be answered."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in [
        "cannot answer", "i cannot answer", "cannot be answered"
    ])


def generate_all_questions(detector, selected_kos, watermark_texts, questions_per_ko, 
                           question_generation_mode, logger):
    """Generate questions for all KOs based on the specified generation mode."""
    all_questions = []
    
    for idx, ko in enumerate(selected_kos):
        if question_generation_mode == "watermark_text_based":
            if idx < len(watermark_texts):
                questions = detector._generate_questions_from_watermark(watermark_texts[idx], verbose=False)
            else:
                logger.warning(f"No watermark text available for KO #{idx+1}")
                questions = []
        else:
            questions = detector._generate_questions(ko, verbose=False)
        
        if questions:
            questions = questions[:questions_per_ko]
            all_questions.extend(questions)
            logger.info(f"KO #{idx+1} generated questions: {questions}")
        else:
            logger.error(f"Failed to generate questions for KO #{idx+1}")
    
    return all_questions


def test_questions_on_database(detector, selected_kos, all_questions, questions_per_ko, p0, 
                               logger, db_label="Original"):
    """Test questions on a database and collect results.
    
    Args:
        detector: RAGWatermarkDetector instance
        selected_kos: List of selected KOs
        all_questions: List of all generated questions
        questions_per_ko: Number of questions per KO
        p0: Null hypothesis probability
        logger: Logger instance
        db_label: Label for logging
    """
    per_ko_results = []

    for idx, ko in enumerate(selected_kos):
        start_idx = idx * questions_per_ko
        end_idx = min(start_idx + questions_per_ko, len(all_questions))
        ko_questions = all_questions[start_idx:end_idx]
        
        correct_count = 0
        
        for q_idx, question in enumerate(ko_questions):
            answer = detector._answer_question_with_rag(question, verbose=False)
            
            logger.info(f"{db_label} database - KO #{idx+1} Question {q_idx+1}: {question}")
            logger.info(f"{db_label} database - RAG answer: {answer[:200]}..." if len(answer) > 200 else f"{db_label} database - RAG answer: {answer}")
            
            if _is_answer_unanswerable(answer):
                is_correct = False
            else:
                is_correct = detector._verify_answer(question, answer, ko, verbose=False)
            
            logger.info(f"{db_label} database - Verification: {'Correct' if is_correct else 'Incorrect'}")
            
            if is_correct:
                correct_count += 1
        
        rate = correct_count / max(1, len(ko_questions))
        p_val = binomial_test_greater(correct_count, len(ko_questions), p0=p0)
        
        result = {
            "num_questions": len(ko_questions),
            "num_correct": correct_count,
            "detection_rate": rate,
            "p_value": p_val,
            "questions": ko_questions,
        }
        
        per_ko_results.append(result)
    
    return per_ko_results


async def test_questions_on_database_async(detector, selected_kos, all_questions, questions_per_ko, p0, 
                               logger, batch_size=10, db_label="Original",
                               watermark_texts=None, watermark_ids=None,
                               all_watermark_ids=None, correct_answers=None, disable_generation=False,
                               verification_mode="correct_answer_based"):
    """Async version of test_questions_on_database
    
    Args:
        watermark_texts: List of watermark texts corresponding to each sampled KO
        watermark_ids: List of watermark IDs corresponding to each sampled KO (for watermark_id field)
        all_watermark_ids: List of all watermark IDs in the database (for is_watermark_retrieved check)
        correct_answers: List of correct answers corresponding to each question
        disable_generation: If True, skip RAG answer generation and verification (retrieval only)
        verification_mode: Answer verification mode - "ko_based" (use KO facts) or "correct_answer_based" (use generated correct_answer)
    """
    import asyncio
    
    semaphore = asyncio.Semaphore(batch_size)
    per_ko_results = []
    watermark_texts = watermark_texts or []
    watermark_ids = watermark_ids or []
    all_watermark_ids = all_watermark_ids or watermark_ids  # fallback to watermark_ids if not provided
    correct_answers = correct_answers or []
    
    # Structure data for concurrent processing
    # We will flat map KOs -> Questions to a list of tasks
    
    tasks = []

    # Each task returns (ko_idx, q_idx, is_correct)
    async def process_question_task(idx, q_idx, question, ko, global_q_idx):
        async with semaphore:
            # Get watermark text and id for this question (from sampled lists)
            wm_text = watermark_texts[idx] if idx < len(watermark_texts) else ""
            wm_id = watermark_ids[idx] if idx < len(watermark_ids) else None
            correct_answer = correct_answers[global_q_idx] if global_q_idx < len(correct_answers) else ""
            
            logger.info(f"{db_label} database - KO #{idx+1} Question {q_idx+1}: {question}")
            answer = await detector._answer_question_with_rag_async(
                question, verbose=False,
                watermark_text=wm_text, watermark_id=wm_id, watermark_ids=all_watermark_ids,
                correct_answer=correct_answer
            )
            
            # Skip verification if disable_generation is True
            if disable_generation:
                logger.info(f"{db_label} database - [Generation disabled - skipping verification]")
                # Update the last retrieved_info with is_correct=None
                if detector.retrieved_documents:
                    for doc_info in reversed(detector.retrieved_documents):
                        if doc_info.get("question") == question:
                            doc_info["is_correct"] = None
                            break
                return (idx, q_idx, None)
            
            logger.info(f"{db_label} database - RAG answer: {answer[:200]}..." if len(answer) > 200 else f"{db_label} database - RAG answer: {answer}")
            
            if _is_answer_unanswerable(answer):
                is_correct = False
            else:
                # Choose verification method based on verification_mode
                if verification_mode == "correct_answer_based" and correct_answer:
                    is_correct = await detector._verify_answer_with_correct_answer_async(question, answer, correct_answer, verbose=False)
                else:
                    # Fallback to ko_based verification
                    is_correct = await detector._verify_answer_async(question, answer, ko, verbose=False)
            
            # Update the last retrieved_info with is_correct
            if detector.retrieved_documents:
                # Find the entry for this question and update is_correct
                for doc_info in reversed(detector.retrieved_documents):
                    if doc_info.get("question") == question:
                        doc_info["is_correct"] = is_correct
                        break
            
            logger.info(f"{db_label} database - Verification: {'Correct' if is_correct else 'Incorrect'}")
            
            return (idx, q_idx, is_correct)

    for idx, ko in enumerate(selected_kos):
        start_idx = idx * questions_per_ko
        end_idx = min(start_idx + questions_per_ko, len(all_questions))
        ko_questions = all_questions[start_idx:end_idx]
        
        for q_idx, question in enumerate(ko_questions):
            global_q_idx = start_idx + q_idx
            tasks.append(process_question_task(idx, q_idx, question, ko, global_q_idx))
            
    results = await asyncio.gather(*tasks)
    
    # Aggregate results back into per-KO structure
    # Initialize per-KO counters
    ko_stats = {} # ko_idx -> {correct_count: 0, total: 0, verified: 0}
    
    for r in results:
        ko_idx, q_idx, is_correct = r
        if ko_idx not in ko_stats:
            ko_stats[ko_idx] = {"correct_count": 0, "total": 0, "verified": 0}
        
        ko_stats[ko_idx]["total"] += 1
        # Only count correct/verified if is_correct is not None (i.e., verification was performed)
        if is_correct is not None:
            ko_stats[ko_idx]["correct_count"] += 1 if is_correct else 0
            ko_stats[ko_idx]["verified"] += 1
                 
    # Reconstruct per_ko_results
    for idx, ko in enumerate(selected_kos):
        start_idx = idx * questions_per_ko
        end_idx = min(start_idx + questions_per_ko, len(all_questions))
        ko_questions = all_questions[start_idx:end_idx]
        
        stats = ko_stats.get(idx, {"correct_count": 0, "total": 0, "verified": 0})
        correct_count = stats["correct_count"]
        verified_count = stats["verified"]
        
        # When generation is disabled, verified_count is 0
        if verified_count > 0:
            rate = correct_count / max(1, verified_count)
            p_val = binomial_test_greater(correct_count, verified_count, p0=p0)
        else:
            rate = None
            p_val = None
        
        result = {
            "num_questions": len(ko_questions),
            "num_correct": correct_count if verified_count > 0 else None,
            "detection_rate": rate,
            "p_value": p_val,
            "questions": ko_questions,
            "generation_disabled": verified_count == 0,
        }
            
        per_ko_results.append(result)
        
    return per_ko_results


async def test_questions_multi_ratio_async(detector, selected_kos, all_questions, questions_per_ko, p0, 
                                           logger, batch_size=10, db_label="Contaminated",
                                           watermark_texts=None, watermark_ids=None,
                                           all_watermark_ids=None):
    """Test questions for multiple theft ratios in a single retrieval pass.
    
    This function uses the detector's allowed_ids_dict to perform one retrieval per question
    and filter for each ratio, producing results for all ratios efficiently.
    
    Args:
        detector: RAGWatermarkDetector instance with allowed_ids_dict set
        selected_kos: List of selected KOs
        all_questions: List of all generated questions
        questions_per_ko: Number of questions per KO
        p0: Null hypothesis probability
        logger: Logger instance
        batch_size: Number of concurrent async requests
        db_label: Label for logging
        watermark_texts: List of watermark texts corresponding to each sampled KO
        watermark_ids: List of watermark IDs corresponding to each sampled KO
        all_watermark_ids: List of all watermark IDs in the database
        
    Returns:
        dict: Mapping ratio (float) -> per_ko_results list
    """
    import asyncio
    
    semaphore = asyncio.Semaphore(batch_size)
    watermark_texts = watermark_texts or []
    watermark_ids = watermark_ids or []
    all_watermark_ids = all_watermark_ids or watermark_ids
    
    if not detector.allowed_ids_dict:
        raise ValueError("detector.allowed_ids_dict must be set for multi-ratio testing")
    
    ratios = list(detector.allowed_ids_dict.keys())
    logger.info(f"{db_label} database - Testing {len(all_questions)} questions across {len(ratios)} theft ratios: {ratios}")
    
    # Initialize per-ratio results storage
    for ratio in ratios:
        detector.retrieved_documents_multi[ratio] = []
    
    # Process each question
    async def retrieve_question_task(idx, q_idx, question, global_q_idx):
        async with semaphore:
            wm_text = watermark_texts[idx] if idx < len(watermark_texts) else ""
            wm_id = watermark_ids[idx] if idx < len(watermark_ids) else None
            
            logger.info(f"{db_label} database - KO #{idx+1} Question {q_idx+1}: {question}")
            
            # Perform multi-ratio retrieval
            ratio_results = await detector._retrieve_multi_ratio_async(
                question, verbose=False,
                watermark_text=wm_text, watermark_id=wm_id, watermark_ids=all_watermark_ids
            )
            
            return (idx, q_idx, ratio_results)
    
    tasks = []
    for idx, ko in enumerate(selected_kos):
        start_idx = idx * questions_per_ko
        end_idx = min(start_idx + questions_per_ko, len(all_questions))
        ko_questions = all_questions[start_idx:end_idx]
        
        for q_idx, question in enumerate(ko_questions):
            global_q_idx = start_idx + q_idx
            tasks.append(retrieve_question_task(idx, q_idx, question, global_q_idx))
    
    results = await asyncio.gather(*tasks)
    
    # Aggregate results per ratio
    multi_ratio_results = {}
    
    for ratio in ratios:
        # Initialize per-KO stats
        ko_stats = {}
        
        for r in results:
            idx, q_idx, ratio_results = r
            if idx not in ko_stats:
                ko_stats[idx] = {"retrieved_count": 0, "total": 0}
            
            ko_stats[idx]["total"] += 1
            # Check if watermark was retrieved for this ratio
            if ratio in ratio_results:
                if ratio_results[ratio].get("is_watermark_retrieved", False):
                    ko_stats[idx]["retrieved_count"] += 1
        
        # Build per_ko_results for this ratio
        per_ko_results = []
        for idx, ko in enumerate(selected_kos):
            start_idx = idx * questions_per_ko
            end_idx = min(start_idx + questions_per_ko, len(all_questions))
            ko_questions = all_questions[start_idx:end_idx]
            
            stats = ko_stats.get(idx, {"retrieved_count": 0, "total": 0})
            retrieved_count = stats["retrieved_count"]
            total = stats["total"]
            
            retrieval_rate = retrieved_count / max(1, total)
            
            result = {
                "num_questions": len(ko_questions),
                "num_watermark_retrieved": retrieved_count,
                "watermark_retrieval_rate": retrieval_rate,
                "questions": ko_questions,
                "theft_ratio": ratio,
                "generation_disabled": True,  # Multi-ratio mode is retrieval-only
            }
            
            per_ko_results.append(result)
        
        multi_ratio_results[ratio] = per_ko_results
        logger.info(f"{db_label} database - Ratio {ratio}: "
                   f"Total watermark retrievals = {sum(s['retrieved_count'] for s in ko_stats.values())} / "
                   f"{sum(s['total'] for s in ko_stats.values())}")
    
    return multi_ratio_results


def calculate_overall_results(per_ko_results, p0, alpha):
    """Calculate overall statistics from per-KO results.
    
    For generation-disabled mode, returns retrieval-only stats.
    """
    # Check if generation was disabled
    if per_ko_results and per_ko_results[0].get("generation_disabled", False):
        total_questions = sum(r["num_questions"] for r in per_ko_results)
        return {
            "total_questions": total_questions,
            "total_correct": None,
            "overall_detection_rate": None,
            "overall_p_value": None,
            "overall_significant": None,
            "generation_disabled": True,
        }
    
    total_questions = sum(r["num_questions"] for r in per_ko_results)
    total_correct = sum(r["num_correct"] for r in per_ko_results if r["num_correct"] is not None)
    overall_rate = total_correct / max(1, total_questions)
    overall_p = binomial_test_greater(total_correct, total_questions, p0=p0)
    
    # Calculate -logp: use calculate_neg_log_p when p_value is 0 to avoid log(0)
    if overall_p == 0:
        neg_log_p = calculate_neg_log_p(total_questions, total_correct, p0)
    else:
        neg_log_p = -math.log10(overall_p)
    
    overall = {
        "total_questions": total_questions,
        "total_correct": total_correct,
        "overall_detection_rate": overall_rate,
        "overall_p_value": overall_p,
        "-logp": neg_log_p,
        "overall_significant": overall_p <= alpha
    }
    
    return overall


def inject_watermarks(vectorstore, watermark_texts, logger):
    """Inject watermark documents into the vector store."""
    watermark_ids = []
    
    for i, watermark_text in enumerate(watermark_texts):
        wid = vectorstore.inject_direct(watermark_text)
        watermark_ids.append(wid)
        logger.info(f"Watermark document #{i+1} added, ID: {wid}")
    
    return watermark_ids


def cleanup_watermarks(vectorstore, watermark_ids, cleanup_requested, logger):
    """Clean up watermark documents from the vector store if requested."""
    if not watermark_ids:
        return
    
    if cleanup_requested:
        try:
            vectorstore.collection.delete(ids=watermark_ids)
            logger.info(f'Cleaned up watermark documents {watermark_ids}')
        except Exception as e:
            logger.error(f"Error cleaning up watermark documents: {e}")


def paraphrase_text(text: str, llm_client, llm_kwargs: dict = None):
    prompt = PromptTemplates.paraphrase_document(text)
    return llm_client.ask_llm(prompt, **llm_kwargs)


async def paraphrase_text_async(text: str, llm_client, llm_kwargs: dict = None):
    """Async version of paraphrase_text for parallel processing."""
    prompt = PromptTemplates.paraphrase_document(text)
    return await llm_client.ask_llm_async(prompt, **llm_kwargs)


async def translate_text_async(text: str, translator, dest: str = "zh-cn"):
    """Async version of translation for parallel processing.
    
    Args:
        text: Text to translate
        translator: googletrans Translator instance
        dest: Target language (default: zh-cn for Chinese)
    
    Returns:
        Translated text
    """
    import asyncio
    import inspect
    try:
        if inspect.iscoroutinefunction(translator.translate):
            result = await translator.translate(text, dest=dest)
        else:
            result = await asyncio.to_thread(translator.translate, text, dest=dest)
        return getattr(result, "text", str(result))
    except Exception as e:
        print(f"Error translating text: {e}")
        return text  # Return original if translation fails


async def run_translate_attack(detector, selected_kos, all_questions, 
                          questions_per_ko, p0, alpha, logger, 
                          batch_size=10, dest_lang="zh-cn"):
    """Perform translation attack on contaminated answers.
    
    The attack is performed in phases:
    1. Retrieve documents for all questions (sequential)
    2. Translate all documents to target language in async with batch_size
    3. Translate back to English
    4. Generate RAG answers with back-translated docs in async with batch_size
    
    Args:
        detector: RAGWatermarkDetector instance
        selected_kos: List of selected KOs
        all_questions: List of all generated questions
        questions_per_ko: Number of questions per KO
        p0: Null hypothesis probability
        alpha: Significance level
        logger: Logger instance
        batch_size: Batch size for async operations (translation and RAG generation)
        dest_lang: Target language for translation (default: zh-cn for Chinese)
    """
    import asyncio
    from googletrans import Translator
    from sentinelrag.prompts import PromptTemplates
    
    translator = Translator()
    
    # ========== PHASE 1: Retrieve documents for ALL questions ==========
    logger.info(f"Translation attack - Phase 1: Retrieving documents for {len(all_questions)} questions...")
    
    # Store retrieval results: list of (ko_idx, q_idx, question, separate_docs, db_ids, distances)
    retrieval_results = []
    
    for ko_idx, ko in enumerate(selected_kos):
        start_idx = ko_idx * questions_per_ko
        end_idx = min(start_idx + questions_per_ko, len(all_questions))
        ko_questions = all_questions[start_idx:end_idx]
        
        for q_idx, question in enumerate(ko_questions):
            # Retrieve documents for this question
            separate_docs, db_ids, distances = detector.visitor.ask_wm_separate(query=question, allowed_ids=detector.allowed_ids, search_k=detector.search_k)
            
            retrieval_results.append({
                "ko_idx": ko_idx,
                "q_idx": q_idx,
                "question": question,
                "separate_docs": separate_docs,
                "db_ids": db_ids,
                "distances": distances,
                "ko": ko,
            })
    
    logger.info(f"Translation attack - Phase 1 complete: Retrieved documents for {len(retrieval_results)} questions")
    
    # ========== PHASE 2: Translate all documents to target language in async with batch_size ==========
    logger.info(f"Translation attack - Phase 2: Translating documents to {dest_lang} (batch_size={batch_size})...")
    
    # Flatten all documents that need translating
    translate_tasks_info = []  # List of (result_idx, doc_idx, doc)
    
    for result_idx, result in enumerate(retrieval_results):
        for doc_idx, doc in enumerate(result["separate_docs"]):
            translate_tasks_info.append((result_idx, doc_idx, doc))
    
    # Create async translate tasks with semaphore
    semaphore = asyncio.Semaphore(batch_size)
    
    async def translate_forward_task(task_info):
        result_idx, doc_idx, doc = task_info
        async with semaphore:
            translated = await translate_text_async(doc, translator, dest=dest_lang)
            return (result_idx, doc_idx, translated)
    
    # Run all forward translation tasks in parallel (with semaphore limiting concurrency)
    if translate_tasks_info:
        forward_results = await asyncio.gather(*[translate_forward_task(info) for info in translate_tasks_info])
        
        # Organize translated docs back to their questions
        for result in retrieval_results:
            result["translated_docs"] = [""] * len(result["separate_docs"])
        
        for result_idx, doc_idx, translated in forward_results:
            retrieval_results[result_idx]["translated_docs"][doc_idx] = translated
    else:
        for result in retrieval_results:
            result["translated_docs"] = []
    
    logger.info(f"Translation attack - Phase 2 complete: Translated {len(translate_tasks_info)} documents to {dest_lang}")
    
    # ========== PHASE 3: Translate back to English in async with batch_size ==========
    logger.info(f"Translation attack - Phase 3: Translating documents back to English (batch_size={batch_size})...")
    
    # Flatten translated documents for back-translation
    back_translate_tasks_info = []
    for result_idx, result in enumerate(retrieval_results):
        for doc_idx, doc in enumerate(result.get("translated_docs", [])):
            if doc:
                back_translate_tasks_info.append((result_idx, doc_idx, doc))
    
    async def translate_back_task(task_info):
        result_idx, doc_idx, doc = task_info
        async with semaphore:
            back_translated = await translate_text_async(doc, translator, dest="en")
            return (result_idx, doc_idx, back_translated)
    
    # Run all back translation tasks in parallel
    if back_translate_tasks_info:
        back_results = await asyncio.gather(*[translate_back_task(info) for info in back_translate_tasks_info])
        
        # Organize back-translated docs
        for result in retrieval_results:
            result["back_translated_docs"] = [""] * len(result["separate_docs"])
        
        for result_idx, doc_idx, back_translated in back_results:
            retrieval_results[result_idx]["back_translated_docs"][doc_idx] = back_translated
        
        # Create combined back-translated document for each question
        for result in retrieval_results:
            result["back_translated_combined"] = "\n".join(result.get("back_translated_docs", []))
    else:
        for result in retrieval_results:
            result["back_translated_docs"] = []
            result["back_translated_combined"] = ""
    
    logger.info(f"Translation attack - Phase 3 complete: Back-translated {len(back_translate_tasks_info)} documents to English")
    
    # ========== PHASE 4: Generate RAG answers in async with batch_size ==========
    logger.info(f"Translation attack - Phase 4: Generating RAG answers (batch_size={batch_size})...")
    
    async def rag_generation_task(result):
        async with semaphore:
            if not result.get("back_translated_combined"):
                return (result, "")
            
            prompt = PromptTemplates.answer_with_rag(
                result["back_translated_combined"],
                result["question"],
                hard_mode=detector.hard_mode,
                xhard_mode=detector.xhard_mode,
            )
            rag_answer = await detector.rllm_client.ask_llm_async(prompt, **detector.rllm_kwargs)
            return (result, rag_answer)
    
    # Run all RAG generation tasks in parallel
    rag_results = await asyncio.gather(*[rag_generation_task(result) for result in retrieval_results])
    
    # Store RAG answers back to results
    for result, rag_answer in rag_results:
        result["rag_answer"] = rag_answer
    
    logger.info(f"Translation attack - Phase 4 complete: Generated {len(rag_results)} RAG answers")
    
    # ========== PHASE 5: Process results ==========
    
    # LLM verification flow - verify answers in async with batch_size
    logger.info(f"Translation attack - Phase 5: Verifying answers (batch_size={batch_size})...")
    
    async def verify_task(result):
        async with semaphore:
            if not result.get("back_translated_combined") or not result["rag_answer"]:
                return (result, False)
            
            rag_answer = result["rag_answer"]
            if _is_answer_unanswerable(rag_answer):
                return (result, False)
            
            is_correct = await detector._verify_answer_async(
                result["question"], rag_answer, result["ko"], verbose=False
            )
            return (result, is_correct)
    
    # Run all verification tasks in parallel
    verification_results = await asyncio.gather(*[verify_task(result) for result in retrieval_results])
    
    # Store verification results
    for result, is_correct in verification_results:
        result["is_correct"] = is_correct
    
    # Build per_question_results and per_ko_results
    per_question_results = []
    per_ko_results = []
    total_correct = 0
    total_questions = 0
    
    # Group results by ko_idx
    ko_results_map = {}
    for result in retrieval_results:
        ko_idx = result["ko_idx"]
        if ko_idx not in ko_results_map:
            ko_results_map[ko_idx] = []
        ko_results_map[ko_idx].append(result)
    
    for ko_idx, ko in enumerate(selected_kos):
        start_idx = ko_idx * questions_per_ko
        end_idx = min(start_idx + questions_per_ko, len(all_questions))
        ko_questions = all_questions[start_idx:end_idx]
        
        ko_correct = 0
        ko_question_results = ko_results_map.get(ko_idx, [])
        
        for result in ko_question_results:
            is_correct = result.get("is_correct", False)
            ko_correct += int(is_correct)
            total_correct += int(is_correct)
            total_questions += 1
            
            orig_combined = "\n".join(result["separate_docs"]) if result["separate_docs"] else ""
            
            per_question_results.append({
                "ko_index": ko_idx,
                "question_index": result["q_idx"],
                "question": result["question"],
                "original_document": orig_combined,
                "translated_document": "\n".join(result.get("translated_docs", [])),
                "back_translated_document": result.get("back_translated_combined", ""),
                "rag_answer": result.get("rag_answer", ""),
                "is_correct": is_correct
            })
        
        p_val = binomial_test_greater(ko_correct, max(1, len(ko_questions)), p0=p0)
        per_ko_results.append({
            "ko_index": ko_idx,
            "num_questions": len(ko_questions),
            "num_correct": ko_correct,
            "detection_rate": ko_correct / max(1, len(ko_questions)),
            "p_value": p_val,
            "questions": ko_questions
        })
    
    overall_rate = total_correct / max(1, total_questions)
    overall_p_value = binomial_test_greater(total_correct, total_questions, p0=p0)
    
    logger.info(f"Translation attack - Complete: {total_correct}/{total_questions} correct, rate={overall_rate:.4f}")
    
    return {
        "per_question": per_question_results,
        "per_ko": per_ko_results,
        "overall": {
            "total_questions": total_questions,
            "total_correct": total_correct,
            "overall_detection_rate": overall_rate,
            "overall_p_value": overall_p_value,
            "overall_significant": overall_p_value <= alpha
        },
        "attack_metadata": {
            "dest_lang": dest_lang,
            "attack_type": "translation"
        }
    }


async def run_paraphrase_attack(detector, selected_kos, all_questions, llm_client, 
                          questions_per_ko, p0, alpha, logger, llm_kwargs: dict = None,
                          batch_size=10):
    """Perform paraphrase attack on contaminated answers.
    
    The attack is performed in three phases:
    1. Retrieve documents for all questions (sequential)
    2. Paraphrase all documents in async with batch_size
    3. Generate RAG answers with paraphrased docs in async with batch_size
    
    Args:
        detector: RAGWatermarkDetector instance
        selected_kos: List of selected KOs
        all_questions: List of all generated questions
        llm_client: LLM client for paraphrasing
        questions_per_ko: Number of questions per KO
        p0: Null hypothesis probability
        alpha: Significance level
        logger: Logger instance
        llm_kwargs: Additional kwargs for LLM client
        batch_size: Batch size for async operations (paraphrasing and RAG generation)
    """
    import asyncio
    from sentinelrag.prompts import PromptTemplates
    
    llm_kwargs = llm_kwargs or {}
    
    # ========== PHASE 1: Retrieve documents for ALL questions ==========
    logger.info(f"Paraphrase attack - Phase 1: Retrieving documents for {len(all_questions)} questions...")
    
    # Store retrieval results: list of (ko_idx, q_idx, question, separate_docs, db_ids, distances)
    retrieval_results = []
    
    for ko_idx, ko in enumerate(selected_kos):
        start_idx = ko_idx * questions_per_ko
        end_idx = min(start_idx + questions_per_ko, len(all_questions))
        ko_questions = all_questions[start_idx:end_idx]
        
        for q_idx, question in enumerate(ko_questions):
            # Retrieve documents for this question
            separate_docs, db_ids, distances = detector.visitor.ask_wm_separate(query=question, allowed_ids=detector.allowed_ids, search_k=detector.search_k)
            
            retrieval_results.append({
                "ko_idx": ko_idx,
                "q_idx": q_idx,
                "question": question,
                "separate_docs": separate_docs,
                "db_ids": db_ids,
                "distances": distances,
                "ko": ko,
            })
            
            logger.info(f"Paraphrase attack - Retrieved {len(separate_docs)} docs for KO #{ko_idx+1} Q{q_idx+1}")
    
    logger.info(f"Paraphrase attack - Phase 1 complete: Retrieved documents for {len(retrieval_results)} questions")
    
    # ========== PHASE 2: Paraphrase all documents in async with batch_size ==========
    logger.info(f"Paraphrase attack - Phase 2: Paraphrasing documents (batch_size={batch_size})...")
    
    # Flatten all documents that need paraphrasing
    # We need to track which question each document belongs to
    paraphrase_tasks_info = []  # List of (result_idx, doc_idx, doc)
    
    for result_idx, result in enumerate(retrieval_results):
        for doc_idx, doc in enumerate(result["separate_docs"]):
            paraphrase_tasks_info.append((result_idx, doc_idx, doc))
    
    # Create async paraphrase tasks with semaphore
    semaphore = asyncio.Semaphore(batch_size)
    
    async def paraphrase_task(task_info):
        result_idx, doc_idx, doc = task_info
        async with semaphore:
            paraphrased = await paraphrase_text_async(doc, llm_client, llm_kwargs=llm_kwargs)
            return (result_idx, doc_idx, paraphrased)
    
    # Run all paraphrase tasks in parallel (with semaphore limiting concurrency)
    if paraphrase_tasks_info:
        paraphrase_results = await asyncio.gather(*[paraphrase_task(info) for info in paraphrase_tasks_info])
        
        # Organize paraphrased docs back to their questions
        # Initialize paraphrased_docs list for each result
        for result in retrieval_results:
            result["paraphrased_docs"] = [""] * len(result["separate_docs"])
        
        for result_idx, doc_idx, paraphrased in paraphrase_results:
            retrieval_results[result_idx]["paraphrased_docs"][doc_idx] = paraphrased
        
        # Create combined paraphrased document for each question
        for result in retrieval_results:
            result["paraphrased_combined"] = "\n".join(result["paraphrased_docs"])
    else:
        for result in retrieval_results:
            result["paraphrased_docs"] = []
            result["paraphrased_combined"] = ""
    
    logger.info(f"Paraphrase attack - Phase 2 complete: Paraphrased {len(paraphrase_tasks_info)} documents")
    
    # ========== PHASE 3: Generate RAG answers in async with batch_size ==========
    logger.info(f"Paraphrase attack - Phase 3: Generating RAG answers (batch_size={batch_size})...")
    
    async def rag_generation_task(result):
        async with semaphore:
            if not result["paraphrased_combined"]:
                return (result, "")
            
            prompt = PromptTemplates.answer_with_rag(
                result["paraphrased_combined"],
                result["question"],
                hard_mode=detector.hard_mode,
                xhard_mode=detector.xhard_mode,
            )
            rag_answer = await detector.rllm_client.ask_llm_async(prompt, **detector.rllm_kwargs)
            return (result, rag_answer)
    
    # Run all RAG generation tasks in parallel
    rag_results = await asyncio.gather(*[rag_generation_task(result) for result in retrieval_results])
    
    # Store RAG answers back to results
    for result, rag_answer in rag_results:
        result["rag_answer"] = rag_answer
    
    logger.info(f"Paraphrase attack - Phase 3 complete: Generated {len(rag_results)} RAG answers")
    
    # ========== PHASE 4: Process results ==========
    
    # LLM verification flow - verify answers in async with batch_size
    logger.info(f"Paraphrase attack - Phase 4: Verifying answers (batch_size={batch_size})...")
    
    async def verify_task(result):
        async with semaphore:
            if not result["paraphrased_combined"] or not result["rag_answer"]:
                return (result, False)
            
            rag_answer = result["rag_answer"]
            if _is_answer_unanswerable(rag_answer):
                return (result, False)
            
            is_correct = await detector._verify_answer_async(
                result["question"], rag_answer, result["ko"], verbose=False
            )
            return (result, is_correct)
    
    # Run all verification tasks in parallel
    verification_results = await asyncio.gather(*[verify_task(result) for result in retrieval_results])
    
    # Store verification results
    for result, is_correct in verification_results:
        result["is_correct"] = is_correct
    
    # Build per_question_results and per_ko_results
    per_question_results = []
    per_ko_results = []
    total_correct = 0
    total_questions = 0
    
    # Group results by ko_idx
    ko_results_map = {}
    for result in retrieval_results:
        ko_idx = result["ko_idx"]
        if ko_idx not in ko_results_map:
            ko_results_map[ko_idx] = []
        ko_results_map[ko_idx].append(result)
    
    for ko_idx, ko in enumerate(selected_kos):
        start_idx = ko_idx * questions_per_ko
        end_idx = min(start_idx + questions_per_ko, len(all_questions))
        ko_questions = all_questions[start_idx:end_idx]
        
        ko_correct = 0
        ko_question_results = ko_results_map.get(ko_idx, [])
        
        for result in ko_question_results:
            is_correct = result.get("is_correct", False)
            ko_correct += int(is_correct)
            total_correct += int(is_correct)
            total_questions += 1
            
            orig_combined = "\n".join(result["separate_docs"]) if result["separate_docs"] else ""
            
            per_question_results.append({
                "ko_index": ko_idx,
                "question_index": result["q_idx"],
                "question": result["question"],
                "original_document": orig_combined,
                "paraphrased_document": result["paraphrased_combined"],
                "rag_answer": result.get("rag_answer", ""),
                "is_correct": is_correct
            })
        
        p_val = binomial_test_greater(ko_correct, max(1, len(ko_questions)), p0=p0)
        per_ko_results.append({
            "ko_index": ko_idx,
            "num_questions": len(ko_questions),
            "num_correct": ko_correct,
            "detection_rate": ko_correct / max(1, len(ko_questions)),
            "p_value": p_val,
            "questions": ko_questions
        })
    
    overall_rate = total_correct / max(1, total_questions)
    overall_p_value = binomial_test_greater(total_correct, total_questions, p0=p0)
    
    logger.info(f"Paraphrase attack - Complete: {total_correct}/{total_questions} correct, rate={overall_rate:.4f}")
    
    return {
        "per_question": per_question_results,
        "per_ko": per_ko_results,
        "overall": {
            "total_questions": total_questions,
            "total_correct": total_correct,
            "overall_detection_rate": overall_rate,
            "overall_p_value": overall_p_value,
            "overall_significant": overall_p_value <= alpha
        }
    }


async def generate_all_questions_async(detector, selected_kos, watermark_texts, questions_per_ko, 
                                   question_generation_mode, logger, batch_size=10):
    """Generate questions for all KOs concurrently based on the specified generation mode."""
    import asyncio
    
    semaphore = asyncio.Semaphore(batch_size)
    
    async def bound_task(task_coro):
        async with semaphore:
            return await task_coro

    tasks = []
    
    for idx, ko in enumerate(selected_kos):
        if question_generation_mode == "watermark_text_based":
            if idx < len(watermark_texts):
                tasks.append(bound_task(detector._generate_questions_from_watermark_async(watermark_texts[idx], verbose=False)))
            else:
                logger.warning(f"No watermark text available for KO #{idx+1}")
                # Append a dummy task that returns empty list
                async def empty_task(): return []
                tasks.append(empty_task())
        else:
            tasks.append(bound_task(detector._generate_questions_async(ko, verbose=False)))
            
    results = await asyncio.gather(*tasks)
    
    all_questions = []
    all_correct_answers = []
    for idx, questions in enumerate(results):
        if questions:
            questions = questions[:questions_per_ko]
            # Handle both old format (list of strings) and new format (list of objects with question and correct_answer)
            for q in questions:
                if isinstance(q, dict):
                    all_questions.append(q.get("question", ""))
                    all_correct_answers.append(q.get("correct_answer", ""))
                else:
                    all_questions.append(q)
                    all_correct_answers.append("")  # No correct answer in old format
            logger.info(f"KO #{idx+1} generated questions: {[q.get('question', q) if isinstance(q, dict) else q for q in questions]}")
        else:
            logger.error(f"Failed to generate questions for KO #{idx+1}")
            
    return all_questions, all_correct_answers


async def perform_watermark_detection(full_dataset, vectorstore, rllm_client, injection_result, 
                                top_k, questions_per_ko,
                                p0, alpha, test_clean, do_paraphrase,
                                cleanup_watermarks_flag, basepath, logger, rllm_kwargs: dict = None,
                                dllm_client=None, dllm_kwargs: dict = None,
                                watermark_query_count=None, skip_detection=False, batch_size=10,
                                disable_generation=False, hard_mode=False, xhard_mode=False, verification_mode="correct_answer_based",
                                do_translate=False, translate_lang="zh-cn",
                                do_partial_theft=False, theft_ratios=None, search_k=100):
    """Perform watermark detection using same questions for both clean and contaminated databases.
    
    Args:
        full_dataset: The full dataset
        vectorstore: Vector store instance
        rllm_client: LLM client for RAG answer generation
        injection_result: Injection result dictionary
        top_k: Number of top-k documents to retrieve
        questions_per_ko: Number of questions per KO
        p0: Null hypothesis probability
        alpha: Significance level
        test_clean: Whether to test on clean database
        do_paraphrase: Whether to perform paraphrase attack
        cleanup_watermarks_flag: Whether to cleanup watermarks after detection
        basepath: Base path for saving results
        logger: Logger instance
        rllm_kwargs: Additional kwargs for RLLM client
        dllm_client: LLM client for watermark detection/verification (if None, uses rllm_client)
        dllm_kwargs: Additional kwargs for DLLM client
        watermark_query_count: Optional number of KOs to sample for detection (None = use all)
        skip_detection: If True, skip original watermark detection and only run attack(s)
        batch_size: Number of concurrent async requests for RAG generation and verification
        disable_generation: If True, skip RAG response generation and verification, only do retrieval
        hard_mode: If True, use a defensive RAG generation prompt to reduce prompt-injection risk
        xhard_mode: If True, use an extra-hard defensive RAG generation prompt
        verification_mode: Answer verification mode - "ko_based" (use KO facts) or "correct_answer_based" (use generated correct_answer)
        do_translate: Whether to perform translation attack
        translate_lang: Target language for translation attack (default: zh-cn for Chinese)
        do_partial_theft: Whether to simulate partial theft attack (random subset of corpus)
        theft_ratios: List of ratios of corpus to keep in partial theft attack (default: [0.5])
        search_k: Candidate pool size for retrieval before filtering in partial theft mode (default: 100)
    """
    # Handle backwards compatibility: convert single theft_ratio to list
    if theft_ratios is None:
        theft_ratios = [0.5]
    elif not isinstance(theft_ratios, list):
        theft_ratios = [theft_ratios]
    
    selected_kos = injection_result['selected_kos']
    watermark_texts = injection_result['watermark_texts']
    
    verification_client = dllm_client if dllm_client is not None else rllm_client
    verification_kwargs = dllm_kwargs if dllm_kwargs is not None else {}
    
    # Log verification mode
    logger.info(f"Using verification mode: {verification_mode}")
    
    if disable_generation:
        logger.info("Generation disabled - skipping RAG response generation and verification")
    if hard_mode:
        logger.info("Hard mode enabled - using defensive RAG generation cue")
    if xhard_mode:
        logger.info("XHard mode enabled - using extra-hard defensive RAG generation cue")
    
    # Keep the clean control genuinely clean. Watermarks are injected only after
    # the optional clean-system run has completed.
    watermark_ids = []
    allowed_ids = None
    allowed_ids_dict = None
    use_multi_ratio = False
    
    # Step 2: Sample KOs if requested (for detection only, not injection)
    if watermark_query_count and watermark_query_count < len(selected_kos):
        logger.info(f"Sampling {watermark_query_count} KOs from {len(selected_kos)} available for detection...")
        import random
        indices = random.sample(range(len(selected_kos)), watermark_query_count)
        sampled_kos = [selected_kos[i] for i in indices]
        sampled_texts = [watermark_texts[i] for i in indices]
        logger.info(f"Sampled {len(sampled_kos)} KOs for detection.")
    else:
        indices = list(range(len(selected_kos)))
        sampled_kos = selected_kos
        sampled_texts = watermark_texts
    
    # Step 3: Load pre-generated questions from the injection result.
    pre_generated_questions = injection_result.get('all_questions', [])
    pre_generated_answers = injection_result.get('all_correct_answers', [])
    
    if not pre_generated_questions:
        raise ValueError(
            "Injection result does not contain pre-generated questions. "
            "Run sentinelrag-inject-watermark first so detection can reuse its watermark_questions/all_questions."
        )

    logger.info(f"Found {len(pre_generated_questions)} pre-generated questions in injection result")

    # Handle sampling if watermark_query_count is specified
    if watermark_query_count and watermark_query_count < len(selected_kos):
        # Need to sample questions corresponding to the sampled KOs
        watermark_questions = injection_result.get('watermark_questions', [])
        if watermark_questions:
            # Re-build all_questions and all_correct_answers from sampled indices
            all_questions = []
            all_correct_answers = []
            for idx in indices:  # indices from the sampling step above
                if idx < len(watermark_questions):
                    qd = watermark_questions[idx]
                    all_questions.extend(qd.get("questions", [])[:questions_per_ko])
                    all_correct_answers.extend(qd.get("correct_answers", [])[:questions_per_ko])
            logger.info(f"Sampled {len(all_questions)} questions from pre-generated questions")
        else:
            # Fallback: use flat list and sample based on questions_per_ko
            all_questions = []
            all_correct_answers = []
            for idx in indices:
                start = idx * questions_per_ko
                end = start + questions_per_ko
                all_questions.extend(pre_generated_questions[start:end])
                all_correct_answers.extend(pre_generated_answers[start:end] if pre_generated_answers else [""] * questions_per_ko)
    else:
        all_questions = pre_generated_questions
        all_correct_answers = pre_generated_answers if pre_generated_answers else [""] * len(pre_generated_questions)
    
    if not all_questions:
        raise ValueError("No pre-generated questions are available for detection")
    
    logger.info(f"Using {len(all_questions)} questions for detection")
    
    # Step 4: Test on original (clean) database (only if test_clean flag is set)
    result_original = None
    detector_original = None
    
    if test_clean:
        detector_original = RAGWatermarkDetector(
            full_dataset, vectorstore, verification_client, rllm_client,
            top_k=top_k, questions_per_ko=questions_per_ko, rllm_kwargs=rllm_kwargs,
            dllm_client=verification_client, dllm_kwargs=verification_kwargs,
            disable_generation=disable_generation, hard_mode=hard_mode, xhard_mode=xhard_mode,
            allowed_ids=None, allowed_ids_dict=None, search_k=search_k
        )
        detector_original.generated_questions = all_questions
        
        clean_results = await test_questions_on_database_async(
            detector_original, sampled_kos, all_questions, questions_per_ko, p0,
            logger, batch_size=batch_size, db_label="Original",
            watermark_texts=sampled_texts, watermark_ids=[],
            all_watermark_ids=[], correct_answers=all_correct_answers,
            disable_generation=disable_generation,
            verification_mode=verification_mode
        )
        
        result_original = {
            "per_ko": clean_results,
            "overall": calculate_overall_results(clean_results, p0, alpha)
        }

    # Inject only after the clean control has finished. The former ordering
    # inserted the sentinels first, causing --test_clean to query a contaminated
    # shared ChromaDB collection and report false positives by construction.
    watermark_ids = inject_watermarks(vectorstore, watermark_texts, logger)
    sampled_watermark_ids = [watermark_ids[i] for i in indices]
    logger.info(f"Injected {len(watermark_ids)} watermark documents into the database")

    # Configure partial-theft filtering against the contaminated collection.
    if do_partial_theft:
        import random
        all_ids = vectorstore.collection.get(include=[])['ids']

        if len(theft_ratios) > 1:
            use_multi_ratio = True
            allowed_ids_dict = {}
            logger.info(f"Simulating Partial Theft with {len(theft_ratios)} ratios: {theft_ratios}")

            for ratio in theft_ratios:
                keep_count = int(len(all_ids) * ratio)
                allowed_ids_dict[ratio] = set(random.sample(all_ids, keep_count))
                logger.info(f"  Ratio {ratio}: Selected {len(allowed_ids_dict[ratio])} IDs as 'stolen' subset")

            logger.info(f"Multi-ratio partial theft configured (search_k={search_k})")
        else:
            theft_ratio = theft_ratios[0]
            logger.info(f"Simulating Partial Theft: Keeping {theft_ratio*100:.1f}% of the corpus...")
            keep_count = int(len(all_ids) * theft_ratio)
            allowed_ids = set(random.sample(all_ids, keep_count))
            logger.info(f"Selected {len(allowed_ids)} IDs as the 'stolen' subset (search_k={search_k}).")
    
    # Step 5: Test on contaminated database using same questions (skip if skip_detection)
    result_watermarked = None
    detector_watermarked = None
    multi_ratio_results = None
    
    if not skip_detection:
        if use_multi_ratio:
            # Multi-ratio partial theft: use multi-ratio retrieval
            logger.info(f"Using multi-ratio retrieval for {len(theft_ratios)} theft ratios...")
            detector_watermarked = RAGWatermarkDetector(
                full_dataset, vectorstore, verification_client, rllm_client,
                top_k=top_k, questions_per_ko=questions_per_ko, rllm_kwargs=rllm_kwargs,
                dllm_client=verification_client, dllm_kwargs=verification_kwargs,
                disable_generation=True, hard_mode=hard_mode, xhard_mode=xhard_mode,  # Multi-ratio mode is retrieval-only
                allowed_ids=None, allowed_ids_dict=allowed_ids_dict, search_k=search_k
            )
            detector_watermarked.generated_questions = all_questions
            
            multi_ratio_results = await test_questions_multi_ratio_async(
                detector_watermarked, sampled_kos, all_questions, questions_per_ko, p0,
                logger, batch_size=batch_size, db_label="Contaminated",
                watermark_texts=sampled_texts, watermark_ids=sampled_watermark_ids,
                all_watermark_ids=watermark_ids
            )
            
            # Calculate overall results for each ratio
            result_watermarked = {"multi_ratio": {}}
            for ratio, per_ko in multi_ratio_results.items():
                total_retrieved = sum(r.get("num_watermark_retrieved", 0) for r in per_ko)
                total_questions = sum(r.get("num_questions", 0) for r in per_ko)
                retrieval_rate = total_retrieved / max(1, total_questions)
                result_watermarked["multi_ratio"][ratio] = {
                    "per_ko": per_ko,
                    "overall": {
                        "total_questions": total_questions,
                        "total_watermark_retrieved": total_retrieved,
                        "watermark_retrieval_rate": retrieval_rate,
                        "theft_ratio": ratio,
                    }
                }
                logger.info(f"Ratio {ratio}: Retrieval rate = {retrieval_rate:.4f} ({total_retrieved}/{total_questions})")
        else:
            # Single ratio or no partial theft: use standard testing
            detector_watermarked = RAGWatermarkDetector(
                full_dataset, vectorstore, verification_client, rllm_client,
                top_k=top_k, questions_per_ko=questions_per_ko, rllm_kwargs=rllm_kwargs,
                dllm_client=verification_client, dllm_kwargs=verification_kwargs,
                disable_generation=disable_generation, hard_mode=hard_mode, xhard_mode=xhard_mode,
                allowed_ids=allowed_ids, allowed_ids_dict=None, search_k=search_k
            )
            detector_watermarked.generated_questions = all_questions
            
            contaminated_results = await test_questions_on_database_async(
                detector_watermarked, sampled_kos, all_questions, questions_per_ko, p0,
                logger, batch_size=batch_size, db_label="Contaminated",
                watermark_texts=sampled_texts, watermark_ids=sampled_watermark_ids,
                all_watermark_ids=watermark_ids, correct_answers=all_correct_answers,
                disable_generation=disable_generation,
                verification_mode=verification_mode
            )
            
            result_watermarked = {
                "per_ko": contaminated_results,
                "overall": calculate_overall_results(contaminated_results, p0, alpha)
            }
    else:
        logger.info("Skipping original detection (skip_detection mode)")
        # Create a minimal detector for attack(s) to use
        detector_watermarked = RAGWatermarkDetector(
            full_dataset, vectorstore, verification_client, rllm_client,
            top_k=top_k, questions_per_ko=questions_per_ko, rllm_kwargs=rllm_kwargs,
            dllm_client=verification_client, dllm_kwargs=verification_kwargs,
            disable_generation=disable_generation, hard_mode=hard_mode, xhard_mode=xhard_mode,
            allowed_ids=allowed_ids, allowed_ids_dict=allowed_ids_dict, search_k=search_k
        )
        detector_watermarked.generated_questions = all_questions
    
    # Step 6: Optional paraphrase attack
    paraphrase_result = None
    paraphrase_results_path = None
    
    if do_paraphrase:
        paraphrase_attack_results = await run_paraphrase_attack(
            detector_watermarked, sampled_kos, all_questions, rllm_client,
            questions_per_ko, p0, alpha, logger, llm_kwargs=rllm_kwargs,
            batch_size=batch_size
        )
        
        # Reorder results to put overall before per_question
        paraphrase_result = {
            "attack_metadata": {
                "timestamp": datetime.now().isoformat(),
                "injection_source": injection_result.get('injection_metadata', {}),
                "top_k": top_k,
                "questions_per_ko": questions_per_ko,
                "alpha": alpha,
                "p0": p0,
                "detection_method": "llm_verification"
            },
        }
        # Add overall first, then per_ko, then per_question (if present)
        if "overall" in paraphrase_attack_results:
            paraphrase_result["overall"] = paraphrase_attack_results["overall"]
        if "per_ko" in paraphrase_attack_results:
            paraphrase_result["per_ko"] = paraphrase_attack_results["per_ko"]
        if "per_question" in paraphrase_attack_results:
            paraphrase_result["per_question"] = paraphrase_attack_results["per_question"]
        
        paraphrase_results_path = os.path.join(basepath, 'paraphrase_attack_results.json')
        save_json(paraphrase_result, paraphrase_results_path)
        logger.info(f"Paraphrase attack results saved to: {paraphrase_results_path}")
    
    # Step 6: Optional translation attack
    translate_result = None
    translate_results_path = None
    
    if do_translate:
        translate_attack_results = await run_translate_attack(
            detector_watermarked, sampled_kos, all_questions,
            questions_per_ko, p0, alpha, logger,
            batch_size=batch_size,
            dest_lang=translate_lang
        )
        
        # Reorder results to put overall before per_question
        translate_result = {
            "attack_metadata": {
                "timestamp": datetime.now().isoformat(),
                "injection_source": injection_result.get('injection_metadata', {}),
                "top_k": top_k,
                "questions_per_ko": questions_per_ko,
                "alpha": alpha,
                "p0": p0,
                "detection_method": "llm_verification",
                "attack_type": "translation",
                "translate_lang": translate_lang,
            },
        }
        # Add overall first, then per_ko, then per_question (if present)
        if "overall" in translate_attack_results:
            translate_result["overall"] = translate_attack_results["overall"]
        if "per_ko" in translate_attack_results:
            translate_result["per_ko"] = translate_attack_results["per_ko"]
        if "per_question" in translate_attack_results:
            translate_result["per_question"] = translate_attack_results["per_question"]
        
        translate_results_path = os.path.join(basepath, 'translate_attack_results.json')
        save_json(translate_result, translate_results_path)
        logger.info(f"Translation attack results saved to: {translate_results_path}")
    
    # Save results
    documents_original_path = None
    if detector_original is not None:
        documents_original_path = os.path.join(basepath, 'retrieved_documents_original_dataset.json')
        save_json({"retrieved_documents": detector_original.retrieved_documents}, documents_original_path)
    
    documents_watermarked_path = None
    if detector_watermarked is not None:
        if detector_watermarked.retrieved_documents:
            documents_watermarked_path = os.path.join(basepath, 'retrieved_documents_watermarked_dataset.json')
            # Documents already have the new format - just save them directly
            save_json({"retrieved_documents": detector_watermarked.retrieved_documents}, documents_watermarked_path)
        
        # Save multi-ratio retrieved documents if they exist
        if detector_watermarked.retrieved_documents_multi:
            for ratio, docs in detector_watermarked.retrieved_documents_multi.items():
                multi_ratio_path = os.path.join(basepath, f'retrieved_documents_ratio_{ratio}.json')
                save_json({"theft_ratio": ratio, "retrieved_documents": docs}, multi_ratio_path)
                logger.info(f"Saved multi-ratio retrieved documents for ratio {ratio} to: {multi_ratio_path}")
    
    # Clean up watermark documents if requested
    cleanup_watermarks(vectorstore, watermark_ids, cleanup_watermarks_flag, logger)
    
    return (result_original, result_watermarked, detector_original, detector_watermarked,
            documents_original_path, documents_watermarked_path, watermark_ids,
            paraphrase_result, paraphrase_results_path,
            translate_result, translate_results_path)


def save_detection_results(basepath, injection_result, full_dataset, result_original, result_watermarked,
                           detector_original, detector_watermarked,
                           documents_original_path, documents_watermarked_path, vectorstore_info, watermark_ids, 
                           paraphrase_result, paraphrase_results_path,
                           eval_dataset, split, top_k, questions_per_ko, alpha, p0, cleanup_watermarks_flag,
                           logger, rllm_model=None, dllm_model=None,
                           translate_result=None, translate_results_path=None):
    """Save detection results to files."""
    
    # Save vector database information
    db_detection_info = {
        **vectorstore_info,
        "watermark_ids": watermark_ids,
        "detection_timestamp": datetime.now().isoformat(),
        "cleanup_performed": cleanup_watermarks_flag
    }
    db_info_path = os.path.join(basepath, 'vectorstore_detection_info.json')
    save_json(db_detection_info, db_info_path)
    
    # Save complete detection results (main file - only overall results)
    detection_results = {
        'detection_metadata': {
            'detection_timestamp': datetime.now().isoformat(),
            'injection_source': injection_result['injection_metadata'],
            'detection_parameters': {
                'top_k': top_k,
                'questions_per_ko': questions_per_ko,
                'alpha': alpha,
                'p0': p0,
                'rllm_model': rllm_model,
                'dllm_model': dllm_model
            }
        },
        'hypothesis_testing': {
            'alpha': alpha,
            'p0': p0
        },
        'test_results': {},
    }

    if result_original is not None:
        detection_results['test_results']['original_dataset'] = {
            'overall': result_original['overall'],
        }

        per_question_original = {
            'description': 'Detailed per-KO detection results on the original dataset',
            'per_ko': result_original['per_ko'],
            'questions': detector_original.generated_questions,
            'retrieved_documents_count': len(detector_original.retrieved_documents),
        }
        per_question_original_path = os.path.join(
            basepath, 'per_question_results_original.json'
        )
        save_json(per_question_original, per_question_original_path)
        logger.info(
            f'Original per-question results saved to: {per_question_original_path}'
        )

    # Only add watermarked_dataset overall if it was tested
    if result_watermarked is not None:
        # Check if this is multi-ratio mode
        if 'multi_ratio' in result_watermarked:
            # Multi-ratio mode: save results for each ratio
            detection_results['test_results']['watermarked_dataset'] = {
                'mode': 'multi_ratio',
                'ratios': list(result_watermarked['multi_ratio'].keys()),
                'multi_ratio': {
                    str(ratio): data['overall'] 
                    for ratio, data in result_watermarked['multi_ratio'].items()
                }
            }
            
            # Save per-ratio detailed results to separate files
            for ratio, data in result_watermarked['multi_ratio'].items():
                per_ratio_results = {
                    'description': f'Detailed per-KO detection results for theft ratio {ratio}',
                    'theft_ratio': ratio,
                    'per_ko': data['per_ko'],
                    'overall': data['overall'],
                }
                per_ratio_path = os.path.join(basepath, f'per_question_results_ratio_{ratio}.json')
                save_json(per_ratio_results, per_ratio_path)
                logger.info(f'Per-question results for ratio {ratio} saved to: {per_ratio_path}')
        else:
            # Standard single-ratio mode
            detection_results['test_results']['watermarked_dataset'] = {
                'overall': result_watermarked['overall'],
            }
            
            # Save per-question detailed results to a separate file
            per_question_results = {
                'description': 'Detailed per-KO detection results on watermark-contaminated dataset',
                'per_ko': result_watermarked['per_ko'],
                'questions': detector_watermarked.generated_questions,
                'retrieved_documents_count': len(detector_watermarked.retrieved_documents)
            }
            per_question_path = os.path.join(basepath, 'per_question_results.json')
            save_json(per_question_results, per_question_path)
            logger.info(f'Per-question results saved to: {per_question_path}')
    
    result_path = os.path.join(basepath, 'detection_results.json')
    save_json(detection_results, result_path)
    logger.info(f'Detection results saved to: {result_path}')
    
    return result_path, db_info_path


class RAGWatermarkDetector:
    """Watermark detector based on real RAG system"""

    def __init__(self, dataset, vectorstore, llm_client, rllm_client=None, top_k: int = 5, questions_per_ko: int = 3, llm_kwargs: dict = None, rllm_kwargs: dict = None, dllm_client=None, dllm_kwargs: dict = None, disable_generation: bool = False, hard_mode: bool = False, xhard_mode: bool = False, allowed_ids=None, allowed_ids_dict=None, search_k=None):
        """
        Initialize watermark detector
        
        Args:
            dataset: Dataset
            vectorstore: Vector store
            llm_client: LLM client for question generation and evaluation
            rllm_client: LLM client specifically for RAG answer generation (if None, uses llm_client)
            top_k: Number of top-k documents to retrieve
            questions_per_ko: Number of verification questions per KO
            llm_kwargs: Additional kwargs to pass to llm_client.ask_llm()
            rllm_kwargs: Additional kwargs to pass to rllm_client.ask_llm()
            dllm_client: LLM client specifically for watermark detection/verification (if None, uses llm_client)
            dllm_kwargs: Additional kwargs to pass to dllm_client.ask_llm()
            disable_generation: If True, skip RAG response generation
            hard_mode: If True, use hardened RAG prompt with defensive anti-injection cue
            xhard_mode: If True, use extra-hard RAG prompt with stricter anti-injection rules
            allowed_ids: Optional set of allowed document IDs for partial theft filtering (single ratio)
            allowed_ids_dict: Optional dict mapping ratio (float) -> set of allowed document IDs.
                              If provided, a single retrieval will produce results for multiple theft ratios.
            search_k: Optional int specifying the candidate pool size before filtering
        """
        self.dataset = dataset
        self.vectorstore = vectorstore
        self.llm_client = llm_client
        self.rllm_client = rllm_client if rllm_client is not None else llm_client
        self.dllm_client = dllm_client if dllm_client is not None else llm_client
        self.visitor = SimpleRAGVisitor(vectorstore, top_k=top_k)
        self.generated_questions = []  # Save generated questions
        self.retrieved_documents = []  # Save retrieved documents for each question
        self.retrieved_documents_multi = {}  # For multi-ratio: {ratio: [retrieved_docs]}
        self.questions_per_ko = questions_per_ko
        self.llm_kwargs = llm_kwargs or {}
        self.rllm_kwargs = rllm_kwargs or {}
        self.dllm_kwargs = dllm_kwargs or {}
        self.disable_generation = disable_generation
        self.hard_mode = hard_mode
        self.xhard_mode = xhard_mode
        self.allowed_ids = allowed_ids
        self.allowed_ids_dict = allowed_ids_dict  # Dict mapping ratio -> allowed_ids set
        self.search_k = search_k

    def _evaluate_question_quality(self, questions, ground_truth_ko):
        """Evaluate question quality, check if too generic or too complex"""
        ko_str = json.dumps(ground_truth_ko, indent=2, ensure_ascii=False)
        prompt = PromptTemplates.evaluate_question_quality(questions, ko_str)
        
        try:
            kwargs = {**self.llm_kwargs, "is_json": True}
            result = self.llm_client.ask_llm(prompt, **kwargs)
            return result
        except Exception as e:
            print(f"Question quality evaluation failed: {e}")
            return {"overall_quality": "unknown", "need_regeneration": False}

    def _generate_questions(self, ground_truth_ko, max_attempts=2, verbose=False):
        """Generate verification questions with quality check and regeneration mechanism"""
        if verbose:
            print("  > Generating verification questions based on KO...")
        ko_str = json.dumps(ground_truth_ko, indent=2, ensure_ascii=False)
        prompt = PromptTemplates.generate_questions(ko_str, num_questions=self.questions_per_ko)
        
        for attempt in range(max_attempts):
            try:
                kwargs = {**self.llm_kwargs, "is_json": True}
                result = self.llm_client.ask_llm(prompt, **kwargs)
                questions = result.get("questions", [])
                
                if not questions:
                    if verbose:
                        print(f"    - Attempt {attempt + 1}: No questions generated, retrying...")
                    continue
                
                if verbose:
                    print(f"    - Generated {len(questions)} questions")
                
                # Evaluate question quality
                if attempt < max_attempts - 1:  # Only evaluate if not last attempt
                    evaluation = self._evaluate_question_quality(questions, ground_truth_ko)
                    
                    if evaluation.get("overall_quality") == "good" or not evaluation.get("need_regeneration", False):
                        self.generated_questions = questions
                        return questions
                    else:
                        if verbose:
                            print("    - Question quality poor, regenerating...")
                        # Add improvement suggestions to prompt
                        if attempt < max_attempts - 1:
                            prompt += f"\n\n**Please improve the following questions:**\n"
                            for eval_item in evaluation.get("evaluation", []):
                                if eval_item.get("quality") == "poor":
                                    prompt += f"- {eval_item.get('question')}: {eval_item.get('reason')}\n"
                        continue
                else:
                    # Last attempt, use directly
                    self.generated_questions = questions
                    return questions
                    
            except Exception as e:
                if verbose:
                    print(f"    - Error generating questions: {e}")
                if attempt == max_attempts - 1:
                    return []
        
        return []

    def _generate_questions_from_watermark(self, watermark_text, max_attempts=2, verbose=False):
        """Generate verification questions based on watermark text"""
        if verbose:
            print("  > Generating verification questions based on watermark text...")
        # prompt = PromptTemplates.generate_questions_from_watermark(watermark_text, num_questions=self.questions_per_ko)
        prompt = PromptTemplates.generate_simple_verification_questions(watermark_text, num_questions=self.questions_per_ko)
        
        for attempt in range(max_attempts):
            try:
                kwargs = {**self.llm_kwargs, "is_json": True}
                result = self.llm_client.ask_llm(prompt, **kwargs)
                questions = result.get("questions", [])
                
                if not questions:
                    if verbose:
                        print(f"    - Attempt {attempt + 1}: No questions generated, retrying...")
                    continue
                
                if verbose:
                    print(f"    - Generated {len(questions)} questions")
                self.generated_questions = questions
                return questions
                    
            except Exception as e:
                if verbose:
                    print(f"    - Error generating questions: {e}")
                if attempt == max_attempts - 1:
                    return []
        
        return []

    async def _evaluate_question_quality_async(self, questions, ground_truth_ko):
        """Async version of _evaluate_question_quality"""
        ko_str = json.dumps(ground_truth_ko, indent=2, ensure_ascii=False)
        prompt = PromptTemplates.evaluate_question_quality(questions, ko_str)
        
        try:
            kwargs = {**self.llm_kwargs, "is_json": True}
            result = await self.llm_client.ask_llm_async(prompt, **kwargs)
            return result
        except Exception as e:
            print(f"Question quality evaluation failed: {e}")
            return {"overall_quality": "unknown", "need_regeneration": False}

    async def _generate_questions_async(self, ground_truth_ko, max_attempts=2, verbose=False):
        """Async version of _generate_questions"""
        if verbose:
            print("  > Generating verification questions based on KO...")
        ko_str = json.dumps(ground_truth_ko, indent=2, ensure_ascii=False)
        prompt = PromptTemplates.generate_questions(ko_str, num_questions=self.questions_per_ko)
        
        for attempt in range(max_attempts):
            try:
                kwargs = {**self.llm_kwargs, "is_json": True}
                result = await self.llm_client.ask_llm_async(prompt, **kwargs)
                questions = result.get("questions", [])
                
                if not questions:
                    if verbose:
                        print(f"    - Attempt {attempt + 1}: No questions generated, retrying...")
                    continue
                
                if verbose:
                    print(f"    - Generated {len(questions)} questions")
                
                # Evaluate question quality
                if attempt < max_attempts - 1:  # Only evaluate if not last attempt
                    evaluation = await self._evaluate_question_quality_async(questions, ground_truth_ko)
                    
                    if evaluation.get("overall_quality") == "good" or not evaluation.get("need_regeneration", False):
                        # Note: we don't save to self.generated_questions here because concurrent calls might overwrite it
                        # The caller should handle aggregation
                        return questions
                    else:
                        if verbose:
                            print("    - Question quality poor, regenerating...")
                        # Add improvement suggestions to prompt
                        if attempt < max_attempts - 1:
                            prompt += f"\n\n**Please improve the following questions:**\n"
                            for eval_item in evaluation.get("evaluation", []):
                                if eval_item.get("quality") == "poor":
                                    prompt += f"- {eval_item.get('question')}: {eval_item.get('reason')}\n"
                        continue
                else:
                    # Last attempt, use directly
                    return questions
                    
            except Exception as e:
                if verbose:
                    print(f"    - Error generating questions: {e}")
                if attempt == max_attempts - 1:
                    return []
        
        return []

    async def _generate_questions_from_watermark_async(self, watermark_text, max_attempts=2, verbose=False):
        """Async version of _generate_questions_from_watermark"""
        if verbose:
            print("  > Generating verification questions based on watermark text...")
        prompt = PromptTemplates.generate_simple_verification_questions(watermark_text, num_questions=self.questions_per_ko)
        
        for attempt in range(max_attempts):
            try:
                kwargs = {**self.llm_kwargs, "is_json": True}
                result = await self.llm_client.ask_llm_async(prompt, **kwargs)
                questions = result.get("questions", [])
                
                if not questions:
                    if verbose:
                        print(f"    - Attempt {attempt + 1}: No questions generated, retrying...")
                    continue
                
                if verbose:
                    print(f"    - Generated {len(questions)} questions")
                return questions
                    
            except Exception as e:
                if verbose:
                    print(f"    - Error generating questions: {e}")
                if attempt == max_attempts - 1:
                    return []
        
        return []

    def _answer_question_with_rag(self, question, verbose=False, 
                                   watermark_text=None, watermark_id=None, watermark_ids=None,
                                   correct_answer=None):
        """Answer question using RAG system
        
        Args:
            question: The question to answer
            verbose: Whether to print debug information
            watermark_text: The watermark text that generated this question
            watermark_id: The watermark ID in the vector store for this question's watermark
            watermark_ids: List of all watermark IDs in the vector store
            correct_answer: The correct answer for this question (from the watermark text)
        """
        if verbose:
            print(f"  > Answering question: '{question}'")
        
        watermark_ids = watermark_ids or []
        
        # Phase 1: Fast retrieval with default top-k
        separate_docs, db_ids, distances = self.visitor.ask_wm_separate(query=question, allowed_ids=self.allowed_ids, search_k=self.search_k)
        
        # Check if watermark is in first retrieval
        is_watermark_in_topk = watermark_id and watermark_id in db_ids
        
        # Phase 2: Calculate watermark rank and score
        watermark_rank = None
        watermark_score = None
        used_extended_search = False
        
        if watermark_id:
            if is_watermark_in_topk:
                # Watermark is in top-k, get rank and score from initial retrieval
                wm_idx = db_ids.index(watermark_id)
                watermark_rank = wm_idx + 1  # 1-indexed rank
                if wm_idx < len(distances):
                    watermark_score = float(distances[wm_idx]) if distances[wm_idx] is not None else None
            else:
                # Watermark not in top-k, do extended search with 10000 results
                extended_ids, extended_distances = self.visitor.ask_wm_extended(extended_n_results=10000, query=question, allowed_ids=self.allowed_ids)
                
                if watermark_id in extended_ids:
                    # Found watermark in extended search (10000)
                    wm_idx = extended_ids.index(watermark_id)
                    watermark_rank = wm_idx + 1  # 1-indexed rank
                    if wm_idx < len(extended_distances):
                        watermark_score = float(extended_distances[wm_idx]) if extended_distances[wm_idx] is not None else None
                    
                    # Use extended search top-k as the new retrieval results
                    # Get documents for the extended top-k ids
                    extended_top_k_ids = extended_ids[:self.visitor.top_k]
                    extended_top_k_distances = extended_distances[:self.visitor.top_k]
                    
                    # Fetch documents for the extended search results
                    try:
                        extended_results = self.vectorstore.collection.get(ids=extended_top_k_ids, include=["documents"])
                        extended_docs = extended_results.get("documents", [])
                        
                        # Update to use extended search results
                        separate_docs = extended_docs
                        db_ids = extended_top_k_ids
                        distances = extended_top_k_distances
                        used_extended_search = True
                    except Exception as e:
                        # If fetching fails, keep original results
                        pass
                else:
                    # Watermark not in top-10000, do another extended search with 30000 results
                    extended_ids, extended_distances = self.visitor.ask_wm_extended(extended_n_results=30000, query=question, allowed_ids=self.allowed_ids)
                    
                    if watermark_id in extended_ids:
                        # Found watermark in deep extended search (30000)
                        wm_idx = extended_ids.index(watermark_id)
                        watermark_rank = wm_idx + 1  # 1-indexed rank
                        if wm_idx < len(extended_distances):
                            watermark_score = float(extended_distances[wm_idx]) if extended_distances[wm_idx] is not None else None
                        
                        # Use extended search top-k as the new retrieval results
                        extended_top_k_ids = extended_ids[:self.visitor.top_k]
                        extended_top_k_distances = extended_distances[:self.visitor.top_k]
                        
                        # Fetch documents for the extended search results
                        try:
                            extended_results = self.vectorstore.collection.get(ids=extended_top_k_ids, include=["documents"])
                            extended_docs = extended_results.get("documents", [])
                            
                            # Update to use extended search results
                            separate_docs = extended_docs
                            db_ids = extended_top_k_ids
                            distances = extended_top_k_distances
                            used_extended_search = True
                        except Exception as e:
                            # If fetching fails, keep original results
                            pass
        
        # Build document_ids dict with ids as keys and documents as values
        documents_dict = {}
        for doc_id, doc in zip(db_ids, separate_docs):
            documents_dict[doc_id] = doc
        
        # Build similarity scores dict (lower distance = higher similarity for cosine/L2)
        similarity_scores = {}
        for doc_id, dist in zip(db_ids, distances):
            similarity_scores[doc_id] = float(dist) if dist is not None else None
        
        # Check if the specific watermark for this question is in the top-k documents
        is_watermark_retrieved = watermark_id in db_ids if watermark_id else False
        
        # Combine documents for RAG answer generation
        rag_document = "\n".join(separate_docs) if separate_docs else ""
        
        # Save retrieved document information with new format
        retrieved_info = {
            "question": question,
            "watermark": {watermark_id: watermark_text or ""},
            "document": documents_dict,
            "similarity_scores": similarity_scores,
            "is_watermark_retrieved": is_watermark_retrieved,
            "watermark_rank": watermark_rank,
            "watermark_score": watermark_score,
            "used_extended_search": used_extended_search,
        }
        
        self.retrieved_documents.append(retrieved_info)
        
        # Generate answer based on retrieved documents using RLLM
        prompt = PromptTemplates.answer_with_rag(
            rag_document,
            question,
            hard_mode=self.hard_mode,
            xhard_mode=self.xhard_mode,
        )
        try:
            answer = self.rllm_client.ask_llm(prompt, **self.rllm_kwargs)
        except Exception as e:
            print(f"    [Error] RAG answer generation failed: {e}")
            answer = "LLM Error"
        
        # Save answer and correct_answer to retrieval info
        retrieved_info["answer"] = answer
        retrieved_info["correct_answer"] = correct_answer or ""
        
        if verbose:
            print(f"    Answer: {answer}")
        
        return answer

    async def _retrieve_and_paraphrase_separately(self, question, llm_client, llm_kwargs, verbose=False, logger=None):
        """Retrieve top-k documents and paraphrase each separately in parallel.
        
        Returns:
            tuple: (paraphrased_combined_document, separate_documents, paraphrased_documents, db_ids, distances)
        """
        import asyncio
        
        if verbose:
            print(f"  > Retrieving and paraphrasing documents for question: '{question}'")
        
        # Get separate documents - pass query directly to avoid race condition
        separate_docs, db_ids, distances = self.visitor.ask_wm_separate(query=question, allowed_ids=self.allowed_ids, search_k=self.search_k)
        
        if not separate_docs:
            if logger:
                logger.warning(f"No documents retrieved for question: {question}")
            return "", [], [], [], []
        
        if logger:
            logger.info(f"Retrieved {len(separate_docs)} documents, paraphrasing each in parallel...")
        
        # Paraphrase each document in parallel
        paraphrase_tasks = [
            paraphrase_text_async(doc, llm_client, llm_kwargs=llm_kwargs)
            for doc in separate_docs
        ]
        
        paraphrased_docs = await asyncio.gather(*paraphrase_tasks)
        
        # Concatenate paraphrased documents
        paraphrased_combined = "\n".join(paraphrased_docs)
        
        if logger:
            logger.info(f"Paraphrased {len(paraphrased_docs)} documents (combined length: {len(paraphrased_combined)} chars)")
        
        return paraphrased_combined, separate_docs, paraphrased_docs, db_ids, distances
        
    async def _answer_question_with_rag_async(self, question, verbose=False, 
                                              watermark_text=None, watermark_id=None, watermark_ids=None,
                                              correct_answer=None):
        """Async version of _answer_question_with_rag
        
        Args:
            question: The question to answer
            verbose: Whether to print debug information
            watermark_text: The watermark text that generated this question
            watermark_id: The watermark ID in the vector store for this question's watermark
            watermark_ids: List of all watermark IDs in the vector store
            correct_answer: The correct answer for this question (from the watermark text)
        """
        if verbose:
            print(f"  > Answering question: '{question}'")
        
        watermark_ids = watermark_ids or []
        
        # Phase 1: Fast retrieval with default top-k
        separate_docs, db_ids, distances = self.visitor.ask_wm_separate(query=question, allowed_ids=self.allowed_ids, search_k=self.search_k)
        
        # Check if watermark is in first retrieval
        is_watermark_in_topk = watermark_id and watermark_id in db_ids
        
        # Phase 2: Calculate watermark rank and score
        watermark_rank = None
        watermark_score = None
        used_extended_search = False
        
        if watermark_id:
            if is_watermark_in_topk:
                # Watermark is in top-k, get rank and score from initial retrieval
                wm_idx = db_ids.index(watermark_id)
                watermark_rank = wm_idx + 1  # 1-indexed rank
                if wm_idx < len(distances):
                    watermark_score = float(distances[wm_idx]) if distances[wm_idx] is not None else None
            else:
                # Watermark not in top-k, do extended search with 10000 results
                extended_ids, extended_distances = self.visitor.ask_wm_extended(extended_n_results=10000, query=question, allowed_ids=self.allowed_ids)
                
                if watermark_id in extended_ids:
                    # Found watermark in extended search (10000)
                    wm_idx = extended_ids.index(watermark_id)
                    watermark_rank = wm_idx + 1  # 1-indexed rank
                    if wm_idx < len(extended_distances):
                        watermark_score = float(extended_distances[wm_idx]) if extended_distances[wm_idx] is not None else None
                    
                    # Use extended search top-k as the new retrieval results
                    # Get documents for the extended top-k ids
                    extended_top_k_ids = extended_ids[:self.visitor.top_k]
                    extended_top_k_distances = extended_distances[:self.visitor.top_k]
                    
                    # Fetch documents for the extended search results
                    try:
                        extended_results = self.vectorstore.collection.get(ids=extended_top_k_ids, include=["documents"])
                        extended_docs = extended_results.get("documents", [])
                        
                        # Update to use extended search results
                        separate_docs = extended_docs
                        db_ids = extended_top_k_ids
                        distances = extended_top_k_distances
                        used_extended_search = True
                    except Exception as e:
                        # If fetching fails, keep original results
                        pass
                else:
                    # Watermark not in top-10000, do another extended search with 30000 results
                    extended_ids, extended_distances = self.visitor.ask_wm_extended(extended_n_results=30000, query=question, allowed_ids=self.allowed_ids)
                    
                    if watermark_id in extended_ids:
                        # Found watermark in deep extended search (30000)
                        wm_idx = extended_ids.index(watermark_id)
                        watermark_rank = wm_idx + 1  # 1-indexed rank
                        if wm_idx < len(extended_distances):
                            watermark_score = float(extended_distances[wm_idx]) if extended_distances[wm_idx] is not None else None
                        
                        # Use extended search top-k as the new retrieval results
                        extended_top_k_ids = extended_ids[:self.visitor.top_k]
                        extended_top_k_distances = extended_distances[:self.visitor.top_k]
                        
                        # Fetch documents for the extended search results
                        try:
                            extended_results = self.vectorstore.collection.get(ids=extended_top_k_ids, include=["documents"])
                            extended_docs = extended_results.get("documents", [])
                            
                            # Update to use extended search results
                            separate_docs = extended_docs
                            db_ids = extended_top_k_ids
                            distances = extended_top_k_distances
                            used_extended_search = True
                        except Exception as e:
                            # If fetching fails, keep original results
                            pass
        
        # Build document_ids dict with ids as keys and documents as values
        documents_dict = {}
        for doc_id, doc in zip(db_ids, separate_docs):
            documents_dict[doc_id] = doc
        
        # Build similarity scores dict (lower distance = higher similarity for cosine/L2)
        similarity_scores = {}
        for doc_id, dist in zip(db_ids, distances):
            similarity_scores[doc_id] = float(dist) if dist is not None else None
        
        # Check if the specific watermark for this question is in the top-k documents
        is_watermark_retrieved = watermark_id in db_ids if watermark_id else False
        
        # Combine documents for RAG answer generation
        rag_document = "\n".join(separate_docs) if separate_docs else ""
        
        # Save retrieved document information with new format
        retrieved_info = {
            "question": question,
            "watermark": {watermark_id: watermark_text or ""},
            "documents": documents_dict,
            "similarity_scores": similarity_scores,
            "is_watermark_retrieved": is_watermark_retrieved,
            "watermark_rank": watermark_rank,
            "watermark_score": watermark_score,
            "used_extended_search": used_extended_search,
        }
        
        self.retrieved_documents.append(retrieved_info)
        
        # Skip LLM generation if disable_generation is True
        if self.disable_generation:
            retrieved_info["answer"] = ""
            retrieved_info["correct_answer"] = correct_answer or ""
            if verbose:
                print(f"    [Generation disabled - skipping answer generation]")
            return ""
        
        # Generate answer based on retrieved documents using RLLM (Async)
        prompt = PromptTemplates.answer_with_rag(
            rag_document,
            question,
            hard_mode=self.hard_mode,
            xhard_mode=self.xhard_mode,
        )
        try:
            answer = await self.rllm_client.ask_llm_async(prompt, **self.rllm_kwargs)
        except Exception as e:
            print(f"    [Error] Async RAG answer generation failed: {e}")
            answer = "LLM Error"
        
        # Save answer and correct_answer to retrieval info
        retrieved_info["answer"] = answer
        retrieved_info["correct_answer"] = correct_answer or ""
        
        if verbose:
            print(f"    Answer: {answer}")
        
        return answer

    async def _retrieve_multi_ratio_async(self, question, verbose=False, 
                                          watermark_text=None, watermark_id=None, watermark_ids=None):
        """Perform retrieval for multiple theft ratios in a single query.
        
        Uses ask_wm_separate_multi to retrieve once and filter for each ratio's allowed_ids.
        
        Args:
            question: The question to retrieve documents for
            verbose: Whether to print debug information
            watermark_text: The watermark text that generated this question
            watermark_id: The watermark ID in the vector store for this question's watermark
            watermark_ids: List of all watermark IDs in the vector store
        
        Returns:
            dict: Mapping ratio -> retrieved_info dict with documents, similarity_scores, etc.
        """
        if verbose:
            print(f"  > Multi-ratio retrieval for question: '{question}'")
        
        watermark_ids = watermark_ids or []
        
        if not self.allowed_ids_dict:
            raise ValueError("allowed_ids_dict must be set for multi-ratio retrieval")
        
        # Single retrieval with multi-ratio filtering
        multi_results = self.visitor.ask_wm_separate_multi(
            query=question, 
            allowed_ids_dict=self.allowed_ids_dict, 
            search_k=self.search_k
        )
        
        ratio_results = {}
        
        for ratio, (separate_docs, db_ids, distances) in multi_results.items():
            # Check if watermark is in retrieved documents
            is_watermark_in_topk = watermark_id and watermark_id in db_ids
            
            # Calculate watermark rank and score
            watermark_rank = None
            watermark_score = None
            
            if watermark_id and is_watermark_in_topk:
                wm_idx = db_ids.index(watermark_id)
                watermark_rank = wm_idx + 1  # 1-indexed rank
                if wm_idx < len(distances):
                    watermark_score = float(distances[wm_idx]) if distances[wm_idx] is not None else None
            
            # Build document_ids dict with ids as keys and documents as values
            documents_dict = {}
            for doc_id, doc in zip(db_ids, separate_docs):
                documents_dict[doc_id] = doc
            
            # Build similarity scores dict
            similarity_scores = {}
            for doc_id, dist in zip(db_ids, distances):
                similarity_scores[doc_id] = float(dist) if dist is not None else None
            
            # Check if watermark is retrieved
            is_watermark_retrieved = watermark_id in db_ids if watermark_id else False
            
            # Save retrieved document information
            retrieved_info = {
                "question": question,
                "watermark": {watermark_id: watermark_text or ""} if watermark_id else {},
                "documents": documents_dict,
                "similarity_scores": similarity_scores,
                "is_watermark_retrieved": is_watermark_retrieved,
                "watermark_rank": watermark_rank,
                "watermark_score": watermark_score,
                "used_extended_search": False,
                "theft_ratio": ratio,
            }
            
            ratio_results[ratio] = retrieved_info
            
            # Store in per-ratio retrieved_documents
            if ratio not in self.retrieved_documents_multi:
                self.retrieved_documents_multi[ratio] = []
            self.retrieved_documents_multi[ratio].append(retrieved_info)
        
        return ratio_results

    async def _verify_answer_async(self, question, answer, ground_truth_ko, verbose=False):
        """Async version of _verify_answer"""
        ko_str = json.dumps(ground_truth_ko, indent=2, ensure_ascii=False)
        prompt = PromptTemplates.verify_answer(question, answer, ko_str)
        
        try:
            result = await self.dllm_client.ask_llm_async(prompt, **self.dllm_kwargs)
            result = result.lower().strip()
            is_correct = "yes" in result
        except Exception as e:
            print(f"    [Error] Async verification failed: {e}")
            is_correct = False
        
        if verbose:
            print(f"    > Verification result: {'Correct' if is_correct else 'Incorrect'}")
            print(f"      Question: {question}")
            print(f"      Answer: {answer}")
        
        return is_correct

    def _verify_answer(self, question, answer, ground_truth_ko, verbose=False):
        """Verify answer correctness using KO facts"""
        ko_str = json.dumps(ground_truth_ko, indent=2, ensure_ascii=False)
        prompt = PromptTemplates.verify_answer(question, answer, ko_str)
        
        try:
            result = self.dllm_client.ask_llm(prompt, **self.dllm_kwargs).lower().strip()
            is_correct = "yes" in result
        except Exception as e:
            print(f"    [Error] Verification failed: {e}")
            is_correct = False
        
        if verbose:
            print(f"    > Verification result: {'Correct' if is_correct else 'Incorrect'}")
            print(f"      Question: {question}")
            print(f"      Answer: {answer}")
        
        return is_correct

    async def _verify_answer_with_correct_answer_async(self, question, answer, correct_answer, verbose=False):
        """Async version of _verify_answer_with_correct_answer - verify answer using the generated correct_answer"""
        prompt = PromptTemplates.judge_answer_correctness(question, answer, correct_answer)
        
        try:
            result = await self.dllm_client.ask_llm_async(prompt, **self.dllm_kwargs)
            result = result.lower().strip()
            is_correct = "yes" in result
        except Exception as e:
            print(f"    [Error] Async verification (correct_answer) failed: {e}")
            is_correct = False
        
        if verbose:
            print(f"    > Verification result (correct_answer_based): {'Correct' if is_correct else 'Incorrect'}")
            print(f"      Question: {question}")
            print(f"      Answer: {answer}")
            print(f"      Expected: {correct_answer}")
        
        return is_correct

    def _verify_answer_with_correct_answer(self, question, answer, correct_answer, verbose=False):
        """Verify answer correctness using the generated correct_answer instead of KO facts"""
        prompt = PromptTemplates.judge_answer_correctness(question, answer, correct_answer)
        
        try:
            result = self.dllm_client.ask_llm(prompt, **self.dllm_kwargs).lower().strip()
            is_correct = "yes" in result
        except Exception as e:
            print(f"    [Error] Verification (correct_answer) failed: {e}")
            is_correct = False
        
        if verbose:
            print(f"    > Verification result (correct_answer_based): {'Correct' if is_correct else 'Incorrect'}")
            print(f"      Question: {question}")
            print(f"      Answer: {answer}")
            print(f"      Expected: {correct_answer}")
        
        return is_correct

    def detect_single_ko(self, ground_truth_ko, questions_per_ko: int = None, p0: float = 0.5, 
                        watermark_text: str = None, question_generation_mode: str = "ko_based"):
        """
        Perform watermark detection for single KO, return detailed statistics
        
        Args:
            ground_truth_ko: Ground truth KO for generating questions
            questions_per_ko: Number of questions per KO
            p0: Null hypothesis probability
            watermark_text: Watermark text for generating questions (when mode is watermark_text_based)
            question_generation_mode: "ko_based" or "watermark_text_based"
        """
        questions_per_ko = questions_per_ko or self.questions_per_ko
        
        # Generate questions based on mode
        if question_generation_mode == "watermark_text_based":
            if watermark_text is None:
                return {
                    "num_questions": 0,
                    "num_correct": 0,
                    "detection_rate": 0.0,
                    "p_value": 1.0,
                    "questions": [],
                }
            questions = self._generate_questions_from_watermark(watermark_text)
        else:
            questions = self._generate_questions(ground_truth_ko)
        
        if not questions:
            return {
                "num_questions": 0,
                "num_correct": 0,
                "detection_rate": 0.0,
                "p_value": 1.0,
                "questions": [],
            }
            
        questions = questions[:questions_per_ko]
        
        # First collect answers for all questions
        qa_pairs = []
        for q in questions:
            answer = self._answer_question_with_rag(q)
            qa_pairs.append((q, answer))
        
        # After all questions are asked, perform unified verification and detection
        positive = 0
        for q, answer in qa_pairs:
            if "cannot answer" in answer.lower() or "I cannot answer" in answer.lower() or "cannot be answered" in answer.lower():
                is_correct = False
            else:
                is_correct = self._verify_answer(q, answer, ground_truth_ko)
            if is_correct:
                positive += 1
        
        rate = positive / max(1, len(questions))
        p_val = binomial_test_greater(positive, len(questions), p0=p0)
        
        return {
            "num_questions": len(questions),
            "num_correct": positive,
            "detection_rate": rate,
            "p_value": p_val,
            "questions": questions,
        }

    def detect_multi_ko(self, ground_truth_kos, questions_per_ko: int = None, alpha: float = 0.05, p0: float = 0.5, 
                       watermark_texts: list = None, question_generation_mode: str = "ko_based"):
        """
        Perform detection for multiple KOs with Holm-Bonferroni correction and calculate overall p-value
        
        Args:
            ground_truth_kos: List of ground truth KOs
            questions_per_ko: Number of questions per KO
            alpha: Significance level for Holm-Bonferroni correction
            p0: Null hypothesis probability
            watermark_texts: List of watermark texts (when mode is watermark_text_based)
            question_generation_mode: "ko_based" or "watermark_text_based"
        """
        results = []
        total_questions = 0
        total_correct = 0
        
        for idx, ko in enumerate(ground_truth_kos):
            watermark_text = watermark_texts[idx] if watermark_texts and idx < len(watermark_texts) else None
            res = self.detect_single_ko(ko, questions_per_ko=questions_per_ko, p0=p0, 
                                       watermark_text=watermark_text, 
                                       question_generation_mode=question_generation_mode)
            results.append(res)
            total_questions += res["num_questions"]
            total_correct += res["num_correct"]
        
        # Calculate overall detection rate and p-value
        overall_detection_rate = total_correct / max(1, total_questions)
        overall_p_value = binomial_test_greater(total_correct, total_questions, p0=p0)
        overall_significant = overall_p_value <= alpha
        
        # Holm-Bonferroni correction
        m = len(results)
        order = sorted(range(m), key=lambda i: results[i]["p_value"])
        rejections = [False] * m
        thresholds = [None] * m
        stop = False
        
        for i, idx in enumerate(order):
            thresh = alpha / (m - i)
            thresholds[idx] = thresh
            if not stop and results[idx]["p_value"] <= thresh:
                rejections[idx] = True
            else:
                stop = True
                
        for i, r in enumerate(results):
            r["holm_threshold"] = thresholds[i]
            r["significant"] = rejections[i]
        
        overall = {
            "num_kos": m,
            "num_significant": sum(1 for x in rejections if x),
            "alpha": alpha,
            "p0": p0,
            "overall_stats": {
                "total_questions": total_questions,
                "total_correct": total_correct,
                "overall_detection_rate": overall_detection_rate,
                "overall_p_value": overall_p_value,
                "overall_significant": overall_significant
            }
        }
        
        return {"per_ko": results, "overall": overall}

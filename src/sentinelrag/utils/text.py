#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text processing utilities.
"""

import re
import json
import hashlib


def clean_str(s):
    """Clean and normalize a string"""
    try:
        s = str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s = s.strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()


def is_valid_json(data: str) -> bool:
    """Check whether the data is valid JSON format"""
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


def extract_doc(WT):
    """Extract watermark text from LLM response"""
    pattern = re.compile(r'{.*}', re.DOTALL)
    matches = pattern.findall(WT)
    
    if len(matches) and is_valid_json(matches[0]):
        watermarks = json.loads(matches[0])
        print(f'watermarks: {watermarks}, {type(watermarks)}')
        print(watermarks['watermark_text'])
        return watermarks['watermark_text']
    return None


def extract_doc_list(WT):
    """Extract list of documents from LLM response"""
    pattern = re.compile(r'\[.*?\]', re.DOTALL)
    matches = pattern.findall(WT)
    
    if len(matches) and is_valid_json(matches[0]):
        WT_list = json.loads(matches[0])
        print(WT_list)
        return WT_list
    return []


def find_substrings_containing_sentences(text, sentences):
    """Find substrings containing specified sentences"""
    
    def split_text(text):
        return re.split(r'\.|\n', text)

    def contains_sentences(substring, sentences):
        return all(sentence.lower() in substring.lower() for sentence in sentences)

    def filter_substrings(text, sentences):
        substrings = split_text(text)
        matched_substrings = [
            substring for substring in substrings 
            if contains_sentences(substring, sentences)
        ]
        return matched_substrings
    
    matched_substrings = filter_substrings(text, sentences)
    concatenated_text = '. '.join(
        substring.strip() for substring in matched_substrings if substring.strip()
    )
    
    return [concatenated_text + '.', len(matched_substrings)]


def documents_hash(documents):
    """Hash documents for deduplication"""
    hash_documents = []
    for document in documents:
        sha256_obj = hashlib.sha256()
        sha256_obj.update(str(document).encode('utf-8'))
        sha256_hash = sha256_obj.hexdigest()
        hash_documents.append(sha256_hash)
    return hash_documents


def remove_duplicates_with_indices(results):
    """Remove duplicates and return indices of removed items"""
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    hashed_documents = documents_hash(documents)

    seen = set()
    indices_removed = []
    result = []
    delete_ids = []
    
    for i, value in enumerate(hashed_documents):
        if value not in seen:
            seen.add(value)
            result.append(value)
        else:
            indices_removed.append(i)
            if metadatas[i]["change"]:
                delete_ids.append(ids[i])
    
    return delete_ids, indices_removed

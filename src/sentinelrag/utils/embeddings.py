#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding model utilities.
Wraps Contriever and other embedding models.
"""

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# Import Contriever from vendored third-party code.
from sentinelrag.vendor.contriever_src.contriever import Contriever


# Model code to model name mappings
model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}


def contriever_get_emb(model, input):
    """Get embeddings from Contriever model"""
    return model(**input)


def dpr_get_emb(model, input):
    """Get embeddings from DPR model"""
    return model(**input).pooler_output


def ance_get_emb(model, input):
    """Get embeddings from ANCE model"""
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]


def load_models(model_code):
    """
    Load embedding models based on model code.
    
    Args:
        model_code: One of 'contriever', 'contriever-msmarco', 'ance'
        
    Returns:
        tuple: (query_model, candidate_model, tokenizer, get_emb_function)
    """
    assert (model_code in model_code_to_qmodel_name and 
            model_code in model_code_to_cmodel_name), \
        f"Model code {model_code} not supported!"
    
    if 'contriever' in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError(f"Model {model_code} not implemented")
    
    return model, c_model, tokenizer, get_emb

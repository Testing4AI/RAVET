#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified RAG visitor for watermark detection
"""


class SimpleRAGVisitor:
    """Simplified RAG visitor."""
    
    def __init__(self, vectorstore, top_k: int = 5):
        """
        Initialize RAG visitor.
        
        Args:
            vectorstore: Vector store.
            top_k: Number of top-k documents to return from retrieval.
        """
        self.vectorstore = vectorstore
        self.wm_unit = None
        self.top_k = top_k
        self.last_documents = []
    
    def ask_wm(self, allowed_ids=None, search_k=None):
        """Retrieve relevant documents with the RAG system.
        
        Args:
            allowed_ids: Optional set of allowed document IDs for partial theft filtering.
                         If provided, only documents with IDs in this set will be returned.
            search_k: Optional int specifying the candidate pool size before filtering.
                      Only used when allowed_ids is provided.
        """
        if self.wm_unit and len(self.wm_unit) > 0:
            query = self.wm_unit[0]  # Use the first element as the query
        else:
            query = "default query"
        
        # Determine n_results based on partial theft mode
        if allowed_ids is not None and search_k is not None:
            n_results = search_k  # Fetch more to filter later
        else:
            n_results = self.top_k
        
        # Search relevant documents through the vector store
        try:
            results = self.vectorstore.search_context(query, n_results=n_results)
            self.last_documents = []
            
            if results and 'documents' in results and len(results['documents']) > 0:
                # Normalize data structure
                documents = results['documents']
                distances = results.get('distances', [])
                
                # ChromaDB query returns documents as a list of lists.
                # For a single query, documents[0] contains all matched documents.
                if len(documents) > 0:
                    if isinstance(documents[0], list):
                        # Standard ChromaDB format: [[doc1, doc2, doc3]]
                        flattened_docs = documents[0]  # Take the first query result
                    else:
                        # Already flattened format: [doc1, doc2, doc3]
                        flattened_docs = documents
                    
                    # Normalize distance scores
                    flattened_distances = []
                    if distances:
                        if isinstance(distances[0], list):
                            flattened_distances = distances[0]  # Take the first query result
                        else:
                            flattened_distances = distances
                    
                    # Ensure all elements are strings
                    flattened_docs = [str(doc) for doc in flattened_docs if doc is not None]
                else:
                    flattened_docs = []
                    flattened_distances = []
                
                if flattened_docs:
                    # Normalize ids, which may also be a nested list
                    ids = results.get('ids', [])
                    if ids and isinstance(ids[0], list):
                        flattened_ids = ids[0]  # Take the first query result
                    else:
                        flattened_ids = ids
                    
                    # Apply partial theft filtering if allowed_ids is provided
                    if allowed_ids is not None:
                        filtered_docs = []
                        filtered_ids = []
                        filtered_distances = []
                        for doc, doc_id, dist in zip(flattened_docs, flattened_ids, flattened_distances):
                            if doc_id in allowed_ids:
                                filtered_docs.append(doc)
                                filtered_ids.append(doc_id)
                                filtered_distances.append(dist)
                                if len(filtered_docs) >= self.top_k:
                                    break
                        flattened_docs = filtered_docs
                        flattened_ids = filtered_ids
                        flattened_distances = filtered_distances
                    else:
                        # Truncate to top_k
                        flattened_docs = flattened_docs[:self.top_k]
                        flattened_ids = flattened_ids[:self.top_k]
                        flattened_distances = flattened_distances[:self.top_k]
                    
                    self.last_documents = flattened_docs
                    
                    # Merge retrieved documents
                    combined_text = "\n".join(flattened_docs)
                    
                    return combined_text, flattened_ids, flattened_distances
                else:
                    return "未找到有效文档", [], []
            else:
                self.last_documents = []
                return "未找到相关文档", [], []
                
        except Exception as e:
            # Suppress terminal output; callers may log errors upstream
            self.last_documents = []
            return "检索系统错误", [], []

    def ask_wm_separate(self, query=None, allowed_ids=None, search_k=None):
        """Retrieve relevant documents with RAG and return a separated document list.
        
        Args:
            query: Optional query string. If not provided, uses self.wm_unit[0]
            allowed_ids: Optional set of allowed document IDs for partial theft filtering.
                         If provided, only documents with IDs in this set will be returned.
            search_k: Optional int specifying the candidate pool size before filtering.
                      Only used when allowed_ids is provided.
        """
        if query is None:
            if self.wm_unit and len(self.wm_unit) > 0:
                query = self.wm_unit[0]  # Use the first element as the query
            else:
                query = "default query"
        
        # Determine n_results based on partial theft mode
        if allowed_ids is not None and search_k is not None:
            n_results = search_k  # Fetch more to filter later
        else:
            n_results = self.top_k
        
        # Search relevant documents through the vector store
        try:
            results = self.vectorstore.search_context(query, n_results=n_results)
            self.last_documents = []
            
            if results and 'documents' in results and len(results['documents']) > 0:
                # Normalize data structure
                documents = results['documents']
                distances = results.get('distances', [])
                
                # ChromaDB query returns documents as a list of lists.
                # For a single query, documents[0] contains all matched documents.
                if len(documents) > 0:
                    if isinstance(documents[0], list):
                        # Standard ChromaDB format: [[doc1, doc2, doc3]]
                        flattened_docs = documents[0]  # Take the first query result
                    else:
                        # Already flattened format: [doc1, doc2, doc3]
                        flattened_docs = documents
                    
                    # Normalize distance scores
                    flattened_distances = []
                    if distances:
                        if isinstance(distances[0], list):
                            flattened_distances = distances[0]  # Take the first query result
                        else:
                            flattened_distances = distances
                    
                    # Ensure all elements are strings
                    flattened_docs = [str(doc) for doc in flattened_docs if doc is not None]
                else:
                    flattened_docs = []
                    flattened_distances = []
                
                if flattened_docs:
                    # Normalize ids, which may also be a nested list
                    ids = results.get('ids', [])
                    if ids and isinstance(ids[0], list):
                        flattened_ids = ids[0]  # Take the first query result
                    else:
                        flattened_ids = ids
                    
                    # Apply partial theft filtering if allowed_ids is provided
                    if allowed_ids is not None:
                        filtered_docs = []
                        filtered_ids = []
                        filtered_distances = []
                        for doc, doc_id, dist in zip(flattened_docs, flattened_ids, flattened_distances):
                            if doc_id in allowed_ids:
                                filtered_docs.append(doc)
                                filtered_ids.append(doc_id)
                                filtered_distances.append(dist)
                                if len(filtered_docs) >= self.top_k:
                                    break
                        flattened_docs = filtered_docs
                        flattened_ids = filtered_ids
                        flattened_distances = filtered_distances
                    else:
                        # Truncate to top_k
                        flattened_docs = flattened_docs[:self.top_k]
                        flattened_ids = flattened_ids[:self.top_k]
                        flattened_distances = flattened_distances[:self.top_k]
                    
                    self.last_documents = flattened_docs
                    
                    # Return separated document list
                    return flattened_docs, flattened_ids, flattened_distances
                else:
                    return [], [], []
            else:
                self.last_documents = []
                return [], [], []
                
        except Exception as e:
            # Suppress terminal output; callers may log errors upstream
            self.last_documents = []
            return [], [], []

    def ask_wm_separate_multi(self, query=None, allowed_ids_dict=None, search_k=None):
        """Retrieve RAG documents for multiple allowed_ids sets in one search.
        
        Args:
            query: Optional query string. If not provided, uses self.wm_unit[0]
            allowed_ids_dict: Dict mapping ratio (float) -> set of allowed document IDs.
                              Each ratio will get its own filtered results.
                              If None or empty, returns unfiltered results under key 'full'.
            search_k: Candidate pool size before filtering (required when allowed_ids_dict is provided).
        
        Returns:
            dict: Mapping ratio -> (docs, ids, distances) for each ratio in allowed_ids_dict.
                  If allowed_ids_dict is None, returns {'full': (docs, ids, distances)}.
        """
        if query is None:
            if self.wm_unit and len(self.wm_unit) > 0:
                query = self.wm_unit[0]
            else:
                query = "default query"
        
        # Determine n_results based on partial theft mode
        if allowed_ids_dict and search_k is not None:
            n_results = search_k
        else:
            n_results = self.top_k
        
        try:
            results = self.vectorstore.search_context(query, n_results=n_results)
            
            if results and 'documents' in results and len(results['documents']) > 0:
                documents = results['documents']
                distances = results.get('distances', [])
                
                # Flatten documents
                if len(documents) > 0:
                    if isinstance(documents[0], list):
                        flattened_docs = documents[0]
                    else:
                        flattened_docs = documents
                    
                    flattened_distances = []
                    if distances:
                        if isinstance(distances[0], list):
                            flattened_distances = distances[0]
                        else:
                            flattened_distances = distances
                    
                    flattened_docs = [str(doc) for doc in flattened_docs if doc is not None]
                else:
                    flattened_docs = []
                    flattened_distances = []
                
                if not flattened_docs:
                    if allowed_ids_dict:
                        return {ratio: ([], [], []) for ratio in allowed_ids_dict}
                    return {'full': ([], [], [])}
                
                # Get IDs
                ids = results.get('ids', [])
                if ids and isinstance(ids[0], list):
                    flattened_ids = ids[0]
                else:
                    flattened_ids = ids
                
                # If no allowed_ids_dict, return full results
                if not allowed_ids_dict:
                    truncated_docs = flattened_docs[:self.top_k]
                    truncated_ids = flattened_ids[:self.top_k]
                    truncated_distances = flattened_distances[:self.top_k]
                    self.last_documents = truncated_docs
                    return {'full': (truncated_docs, truncated_ids, truncated_distances)}
                
                # Filter for each ratio's allowed_ids
                multi_results = {}
                for ratio, allowed_ids in allowed_ids_dict.items():
                    filtered_docs = []
                    filtered_ids = []
                    filtered_distances = []
                    for doc, doc_id, dist in zip(flattened_docs, flattened_ids, flattened_distances):
                        if doc_id in allowed_ids:
                            filtered_docs.append(doc)
                            filtered_ids.append(doc_id)
                            filtered_distances.append(dist)
                            if len(filtered_docs) >= self.top_k:
                                break
                    multi_results[ratio] = (filtered_docs, filtered_ids, filtered_distances)
                
                # Store the first ratio's docs as last_documents for backward compatibility
                if multi_results:
                    first_ratio = next(iter(multi_results))
                    self.last_documents = multi_results[first_ratio][0]
                
                return multi_results
            else:
                if allowed_ids_dict:
                    return {ratio: ([], [], []) for ratio in allowed_ids_dict}
                return {'full': ([], [], [])}
                
        except Exception as e:
            if allowed_ids_dict:
                return {ratio: ([], [], []) for ratio in allowed_ids_dict}
            return {'full': ([], [], [])}

    def ask_wm_extended(self, extended_n_results=100, query=None, allowed_ids=None):
        """Retrieve more RAG documents to locate watermark positions.
        
        Args:
            extended_n_results: Number of results for the extended search (default: 100)
            query: Optional query string. If not provided, uses self.wm_unit[0]
            allowed_ids: Optional set of allowed document IDs for partial theft filtering
            
        Returns:
            tuple: (extended_ids, extended_distances) for the extended search results
        """
        if query is None:
            if self.wm_unit and len(self.wm_unit) > 0:
                query = self.wm_unit[0]
            else:
                query = "default query"
        
        try:
            results = self.vectorstore.search_context(query, n_results=extended_n_results)
            
            if results and 'ids' in results and len(results['ids']) > 0:
                ids = results.get('ids', [])
                distances = results.get('distances', [])
                
                # Normalize nested list format
                if ids and isinstance(ids[0], list):
                    flattened_ids = ids[0]
                else:
                    flattened_ids = ids
                    
                if distances and isinstance(distances[0], list):
                    flattened_distances = distances[0]
                else:
                    flattened_distances = distances
                
                # Apply partial theft filtering if allowed_ids is provided
                if allowed_ids is not None:
                    filtered_ids = []
                    filtered_distances = []
                    for doc_id, dist in zip(flattened_ids, flattened_distances):
                        if doc_id in allowed_ids:
                            filtered_ids.append(doc_id)
                            filtered_distances.append(dist)
                    return filtered_ids, filtered_distances
                else:
                    return flattened_ids, flattened_distances
            else:
                return [], []
                
        except Exception as e:
            return [], []

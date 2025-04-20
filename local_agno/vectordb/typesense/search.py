from typing import Dict, List, Optional, Union

import numpy as np

from agno.document import Document
from agno.utils.log import log_debug


def build_filter_string(filters: Dict) -> str:
    """Build Typesense filter string from dictionary

    Args:
        filters: Dictionary of filters

    Returns:
        str: Filter string
    """
    filter_parts = []
    for field, value in filters.items():
        if isinstance(value, (list, tuple)):
            filter_parts.append(f"{field}:=[{','.join(map(str, value))}]")
        else:
            filter_parts.append(f"{field}:={value}")
    return " && ".join(filter_parts)


class TypesenseSearch:
    """Handles different types of searches in Typesense"""

    def __init__(self, collection, client=None):
        """Initialize search handler

        Args:
            collection: Typesense collection
            client: Typesense client
        """
        self.collection = collection
        self.client = client  # Store client separately

    def _process_results(self, results: Dict) -> List[Document]:
        """Process search results into Documents

        Args:
            results: Typesense search results

        Returns:
            List[Document]: List of documents
        """
        documents = []
        for hit in results.get('hits', []):
            doc = hit.get('document', {})
            try:
                document = Document(
                    id=doc.get('id'),
                    content=doc.get('content', ''),
                    meta_data={
                        'vector_distance': hit.get('vector_distance'),
                        **{k: v for k, v in doc.items() if k not in ['id', 'content']}
                    },
                    embedding=np.array(doc.get('embedding')) if doc.get('embedding') else None
                )
                documents.append(document)
            except Exception as e:
                log_debug(f"Error processing search result: {e}")
                continue
        return documents

    def vector_search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """Perform vector search using multi_search endpoint

        Args:
            query_vector: Query vector
            limit: Number of results
            filters: Search filters

        Returns:
            List[Document]: Search results
        """
        log_debug(f"Performing vector search with limit {limit}")
        
        try:
            # Log vector dimensions
            log_debug(f"Vector dimensions: {len(query_vector)}")
            
            # Format vector as comma-separated values
            vector_str = ",".join(str(v) for v in query_vector)
            
            # Log vector format (truncated to avoid huge logs)
            log_debug(f"Vector format (first 5 values): {vector_str[:100]}...")
            
            search_request = {
                "searches": [{
                    "q": "*",
                    "collection": self.collection.name,
                    "vector_query": f"embedding:([{vector_str}], k:{limit})",
                    "limit": limit
                }]
            }

            if filters:
                search_request["searches"][0]["filter_by"] = build_filter_string(filters)

            log_debug(f"Search parameters: {search_request}")
            
            # Use multi_search endpoint from client directly
            log_debug(f"Sending multi-search request to Typesense...")
            results = self.client.multi_search.perform(search_request, {})
            log_debug(f"Multi-search complete")
            
            # Process multi_search results
            if results and results.get('results') and results['results'][0].get('hits'):
                hits = results['results'][0]['hits']
                log_debug(f"Found {len(hits)} hits with multi-search")
                return self._process_results({'hits': hits})
            else:
                log_debug(f"No hits found in multi-search results: {results}")
            return []
        except Exception as e:
            log_debug(f"Error during vector search: {e}")
            log_debug(f"Query vector type: {type(query_vector)}")
            return []

    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """Perform keyword search

        Args:
            query: Search query
            limit: Number of results
            filters: Search filters

        Returns:
            List[Document]: Search results
        """
        log_debug(f"Performing keyword search for query: {query}")
        
        search_params = {
            'q': query,
            'query_by': 'content',
            'limit': limit
        }

        if filters:
            search_params['filter_by'] = build_filter_string(filters)

        log_debug(f"Search parameters: {search_params}")
        results = self.collection.documents.search(search_params)
        log_debug(f"Hi! Found {len(results.get('hits', []))} results")
        
        return self._process_results(results)

    def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """Perform hybrid search (vector + keyword) using multi_search endpoint
        
        Typesense does hybrid search by combining the q parameter with vector_query parameter.
        It automatically applies rank fusion with weights: 0.7 for keyword search, 0.3 for vector search.

        Args:
            query: Text query for keyword search
            query_vector: Vector query for similarity search
            limit: Maximum number of results
            filters: Search filters

        Returns:
            List[Document]: Search results
        """
        # print(f"DEBUG: Performing hybrid search with query: '{query}'")
        # log_debug(f"Performing hybrid search with query: {query}")
        
        try:
            # Format vector as comma-separated values
            vector_str = ",".join(str(v) for v in query_vector)
            
            # Create search parameters for multi_search endpoint
            # For hybrid search, we need both q (non-wildcard) and vector_query
            search_request = {
                "searches": [{
                    "q": query,  # Query string for text search
                    "collection": self.collection.name,
                    "query_by": "content",  # Search in content field
                    "vector_query": f"embedding:([{vector_str}])",  # Vector part
                    "limit": limit
                }]
            }
            
            if filters:
                search_request["searches"][0]["filter_by"] = build_filter_string(filters)
            
            #print(f"DEBUG: Hybrid search parameters: {search_request}")
            # log_debug(f"Search parameters: {search_request}")
            
            # Use multi_search endpoint for hybrid search
            #print("DEBUG: Sending hybrid search request to Typesense...")
            results = self.client.multi_search.perform(search_request, {})
            #print("DEBUG: Hybrid search request complete")
            
            # Process multi_search results
            if results and results.get('results') and results['results'][0].get('hits'):
                hits = results['results'][0]['hits']
                print(f"DEBUG: Found {len(hits)} hits with hybrid search")
                log_debug(f"Found {len(hits)} hits with hybrid search")
                return self._process_results({'hits': hits})
            else:
                print("DEBUG: No hits found in hybrid search results")
                if results and results.get('results'):
                    print(f"DEBUG: Results structure: {results['results'][0].keys()}")
                log_debug(f"No hits found in hybrid search results")
            return []
        except Exception as e:
            print(f"DEBUG: Error during hybrid search: {str(e)}")
            log_debug(f"Error during hybrid search: {e}")
            return [] 
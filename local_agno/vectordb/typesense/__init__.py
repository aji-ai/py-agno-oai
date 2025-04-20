from typing import Any, Dict, List, Optional, Union
import uuid

import numpy as np
from typesense.exceptions import TypesenseClientError, ObjectNotFound
from typesense.types.multi_search import MultiSearchRequestSchema, MultiSearchResponse
from typesense.types.document import DocumentSchema, SearchResponse

from agno.document import Document
from agno.embedder import Embedder
from agno.embedder.openai import OpenAIEmbedder
from agno.reranker.base import Reranker
from agno.utils.log import log_info, log_debug, log_error
from agno.vectordb.base import VectorDb
from agno.vectordb.search import SearchType

from .index import Distance, HNSWConfig, create_schema
from .search import TypesenseSearch, build_filter_string

import typesense
from typesense.configuration import Configuration
from typesense.api_call import ApiCall

class TypesenseDb(VectorDb):
    """Typesense vector database implementation with hybrid search support"""

    def __init__(
        self,
        name: str,
        dimension: int = 512,
        api_key: str = None,
        host: str = "localhost",
        port: int = 8108,
        protocol: str = "http",
        distance: Distance = Distance.COSINE,
        hnsw_config: Optional[HNSWConfig] = None,
        embedder: Optional[Embedder] = None,
        search_type: SearchType = SearchType.vector,
        reranker: Optional[Reranker] = None,
    ):
        """Initialize Typesense database"""
        self.name = name
        self.dimension = dimension
        self.distance = distance
        self.hnsw_config = hnsw_config or HNSWConfig()
        
        # Create config dictionary
        config_dict = {
            'api_key': api_key,
            'nodes': [{
                'host': host,
                'port': str(port),
                'protocol': protocol
            }],
            'connection_timeout_seconds': 2,
            'retry_interval_seconds': 0.1,
            'num_retries': 3
        }
        
        # Initialize client with config dictionary
        self.client = typesense.Client(config_dict)
        
        # Create schema
        self.schema = create_schema(
            name=self.name,
            dimension=self.dimension,
            distance=self.distance,
            hnsw_config=self.hnsw_config
        )
        
        # Initialize search handler
        self._search = None
        
        # Search configuration
        if embedder is None:
            embedder = OpenAIEmbedder()
            log_info("Embedder not provided, using OpenAIEmbedder as default")
        self.embedder = embedder
        self.search_type = search_type
        self.reranker = reranker

    @property
    def search_handler(self) -> TypesenseSearch:
        """Get or create search handler

        Returns:
            TypesenseSearch: Search handler

        Raises:
            TypesenseClientError: If search handler creation fails
        """
        if self._search is None:
            try:
                # Create collection if it doesn't exist
                if not self.exists():
                    self.create()
                collection = self.client.collections[self.name]
                # Pass both collection and client
                self._search = TypesenseSearch(collection, self.client)
                log_debug(f"Created search handler for collection {self.name}")
            except TypesenseClientError as e:
                log_error(f"Error creating search handler: {e}")
                raise
        return self._search

    def create(self) -> None:
        """Create the collection if it doesn't exist

        Raises:
            TypesenseClientError: If collection creation fails
        """
        try:
            if not self.exists():
                log_debug(f"Creating collection '{self.name}'")
                self.client.collections.create(self.schema)
                log_debug(f"Collection '{self.name}' created successfully")
            else:
                log_debug(f"Collection '{self.name}' already exists")
        except TypesenseClientError as e:
            log_error(f"Error creating collection: {e}")
            raise

    async def async_create(self) -> None:
        """Create the collection asynchronously"""
        self.create()

    def doc_exists(self, document: Document) -> bool:
        """Check if document exists

        Args:
            document: Document to check

        Returns:
            bool: True if document exists, False otherwise
        """
        try:
            self.client.collections[self.name].documents[document.id].retrieve()
            return True
        except ObjectNotFound:
            return False
        except TypesenseClientError:
            return False

    async def async_doc_exists(self, document: Document) -> bool:
        """Check if document exists asynchronously"""
        return self.doc_exists(document)

    def name_exists(self, name: str) -> bool:
        """Check if collection exists"""
        return self.client.collections.exists(name)

    def async_name_exists(self, name: str) -> bool:
        """Check if collection exists asynchronously"""
        return self.name_exists(name)

    def insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """Insert documents

        Args:
            documents: List of documents to insert
            filters: Optional filters for insertion

        Raises:
            TypesenseClientError: If document insertion fails
        """
        if not documents:
            log_debug("No documents to insert")
            return

        print(f"INSERTING {len(documents)} DOCUMENTS")
        
        # Ensure collection exists
        if not self.exists():
            log_debug("Collection does not exist, creating...")
            self.create()

        docs_to_insert = []
        for doc in documents:
            try:
                print(f"Processing document: {doc.id or 'no-id'} - Content: {doc.content[:50]}...")
                
                # Generate embedding if not present
                if doc.embedding is None and doc.content:
                    print(f"Generating embedding for document {doc.id}")
                    doc.embedding = self.embedder.get_embedding(doc.content)
                    print(f"Got embedding with {len(doc.embedding)} dimensions")
                
                # Convert embedding to list if it's a numpy array
                embedding = doc.embedding.tolist() if isinstance(doc.embedding, np.ndarray) else doc.embedding
                print(f"Using embedding with {len(embedding)} dimensions")
                
                # Clean and prepare meta_data
                meta_data = doc.meta_data if doc.meta_data else {}
                if not isinstance(meta_data, dict):
                    meta_data = {"value": str(meta_data)}
                
                # Prepare document with required fields
                doc_dict = {
                    'id': doc.id or str(uuid.uuid4()),
                    'content': doc.content,
                    'embedding': embedding,
                    'meta_data': meta_data
                }
                docs_to_insert.append(doc_dict)
                print(f"Prepared document {doc_dict['id']} for insertion")
            except Exception as e:
                print(f"ERROR preparing document for insertion: {e}")
                log_error(f"Error preparing document for insertion: {e}")
                raise

        if docs_to_insert:
            try:
                collection = self.client.collections[self.name]
                print(f"Inserting {len(docs_to_insert)} documents into collection '{self.name}'")
                
                # Insert documents one by one instead of bulk import
                results = []
                for doc_dict in docs_to_insert:
                    try:
                        print(f"Inserting document: {doc_dict['id']}")
                        result = collection.documents.create(doc_dict)
                        results.append({"success": True, "document": doc_dict['id']})
                        print(f"Document {doc_dict['id']} inserted successfully")
                    except Exception as e:
                        print(f"Error inserting document {doc_dict['id']}: {e}")
                        results.append({"success": False, "document": doc_dict['id'], "error": str(e)})
                
                print(f"Documents inserted, result: {results}")
                log_debug(f"Documents insertion completed with results: {results}")
            except Exception as e:
                print(f"ERROR inserting documents: {e}")
                log_error(f"Error inserting documents: {e}")
                raise

    async def async_insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """Insert documents asynchronously"""
        self.insert(documents, filters)

    def upsert_available(self) -> bool:
        """Check if upsert is available"""
        return True

    def upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """Upsert documents"""
        self.insert(documents, filters)  # Typesense handles upserts automatically

    async def async_upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """Upsert documents asynchronously"""
        self.upsert(documents, filters)

    def vector_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform vector search

        Args:
            query: Query string
            limit: Maximum number of results
            filters: Optional filters for filtering results

        Returns:
            List[Document]: Search results

        Raises:
            TypesenseClientError: If search fails
        """
        try:
            query_vector = self.embedder.get_embedding(query)
            # Handle both numpy arrays and lists from embedder
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()
                
            # Call search handler's vector search
            return self.search_handler.vector_search(
                query_vector=query_vector,
                limit=limit,
                filters=filters
            )
        except Exception as e:
            log_error(f"Vector search failed: {e}")
            return []

    def keyword_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform keyword search

        Args:
            query: Query string
            limit: Maximum number of results
            filters: Optional filters for filtering results

        Returns:
            List[Document]: Search results

        Raises:
            TypesenseClientError: If search fails
        """
        try:
            return self.search_handler.keyword_search(
                query=query,
                limit=limit,
                filters=filters
            )
        except Exception as e:
            log_error(f"Keyword search failed: {e}")
            return []

    def hybrid_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform hybrid search

        Args:
            query: Query string
            limit: Maximum number of results
            filters: Optional filters for filtering results

        Returns:
            List[Document]: Search results

        Raises:
            TypesenseClientError: If search fails
        """
        try:
            print(f"DEBUG: TypesenseDb hybrid_search called with query: '{query}', limit: {limit}")
            # Generate embedding for query
            query_vector = self.embedder.get_embedding(query)
            
            # Handle both numpy arrays and lists from embedder
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()
            
            # Call search handler's hybrid search
            results = self.search_handler.hybrid_search(
                query=query,
                query_vector=query_vector,
                limit=limit,
                filters=filters
            )
            
            # print(f"DEBUG: Hybrid search returned {len(results)} results")
            return results
        except Exception as e:
            print(f"DEBUG: Error in TypesenseDb hybrid_search: {str(e)}")
            log_error(f"Hybrid search failed: {e}")
            return []

    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform search based on configured search type
        
        Args:
            query: Query string
            limit: Maximum number of results
            filters: Optional filters for filtering results
            
        Returns:
            List[Document]: Search results
        """
#        limit = 7
        try:
            log_debug(f"Performing search with type {self.search_type}")
            
            # Use dedicated methods based on search type
            if self.search_type == SearchType.vector:
                log_debug("Using vector search")
                return self.vector_search(query, limit, filters)
            elif self.search_type == SearchType.keyword:
                log_debug("Using keyword search")
                return self.keyword_search(query, limit, filters)
            else:  # hybrid search
                log_debug("Using hybrid search")
                return self.hybrid_search(query, limit, filters)
        except Exception as e:
            log_error(f"Search failed: {e}")
            return []

    async def async_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search asynchronously"""
        return self.search(query, limit, filters)

    def drop(self) -> None:
        """Drop the collection

        Raises:
            TypesenseClientError: If collection deletion fails
        """
        try:
            if self.exists():
                self.client.collections[self.name].delete()
                self._search = None  # Reset search handler
                log_debug(f"Collection '{self.name}' dropped successfully")
            else:
                log_debug(f"Collection '{self.name}' does not exist")
        except TypesenseClientError as e:
            log_error(f"Error dropping collection: {e}")
            raise

    async def async_drop(self) -> None:
        """Drop the collection asynchronously"""
        self.drop()

    def exists(self) -> bool:
        """Check if collection exists

        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            self.client.collections[self.name].retrieve()
            return True
        except ObjectNotFound:
            return False
        except TypesenseClientError as e:
            log_error(f"Error checking collection existence: {e}")
            return False

    async def async_exists(self) -> bool:
        """Check if collection exists asynchronously"""
        return self.exists()

    def delete(self) -> bool:
        """Delete the collection"""
        try:
            self.drop()
            return True
        except TypesenseClientError:
            return False

    # List all collections
    def list_collections(self):
        try:
            # Get list of collections
            collections = self.client.collections.retrieve()
            if collections:
                print(f"Found collections: {[c.get('name') for c in collections]}")
            else:
                print("No collections found")
        except TypesenseClientError as e:
            print(f"Collection operation failed: {str(e)}")
            return 
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


class Distance(str, Enum):
    """Vector distance metrics"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class HNSWConfig:
    """HNSW index configuration"""
    max_elements: int = 1000000  # Maximum number of vectors to index
    m: int = 16  # Number of bidirectional links created for every new element
    ef_construction: int = 100  # Size of the dynamic candidate list for constructing the graph
    ef: int = 64  # Size of the dynamic candidate list for searching the graph

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "max_elements": self.max_elements,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef": self.ef
        }


def create_schema(
    name: str,
    dimension: int,
    distance: Distance = Distance.COSINE,
    hnsw_config: Optional[HNSWConfig] = None,
    additional_fields: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Create Typesense collection schema

    Args:
        name: Collection name
        dimension: Vector dimension
        distance: Distance metric
        hnsw_config: HNSW index configuration
        additional_fields: Additional fields to add to schema

    Returns:
        Dict: Collection schema
    """
    # Create embedding field with correct parameters
    embedding_field = {
        'name': 'embedding',
        'type': 'float[]',
        'num_dim': dimension,
        'vec_dist': distance.value  # Add vec_dist parameter directly to the field
    }
    
    # Add HNSW parameters directly to the embedding field if provided
    if hnsw_config:
        embedding_field['hnsw_params'] = {
            'M': hnsw_config.m,
            'ef_construction': hnsw_config.ef_construction
        }
    
    schema = {
        'name': name,
        'enable_nested_fields': True,  # Enable nested fields for metadata
        'fields': [
            {'name': 'id', 'type': 'string'},
            {'name': 'content', 'type': 'string'},
            {'name': 'meta_data', 'type': 'object'},
            embedding_field
        ]
    }

    # Add additional fields
    if additional_fields:
        schema['fields'].extend(additional_fields)

    return schema 
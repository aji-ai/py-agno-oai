import os
from pathlib import Path
import argparse
from typing import List, Union, IO, Any
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.document import Document
from agno.document.reader.text_reader import TextReader
from local_agno.vectordb.typesense import TypesenseDb, SearchType
from agno.tools.knowledge import KnowledgeTools

from agno.models.anthropic import Claude

class ImprovedTextReader(TextReader):
    """Text reader with enhanced metadata including relative paths."""
    
    def __init__(self, base_dir: str = None):
        """Initialize with optional base directory for relative paths."""
        super().__init__()
        self.base_dir = Path(base_dir) if base_dir else None
    
    def read(self, file: Union[Path, IO[Any]]) -> List[Document]:
        """Read a file and return documents with enhanced metadata."""
        try:
            if isinstance(file, Path):
                if not file.exists():
                    raise FileNotFoundError(f"Could not find file: {file}")
                
                file_name = file.name
                file_contents = file.read_text("utf-8")
                
                # Use relative path if base_dir is provided
                if self.base_dir:
                    try:
                        rel_path = file.relative_to(self.base_dir)
                        file_path = str(Path(self.base_dir.name) / rel_path)
                    except ValueError:
                        # Fall back to filename if path is not relative to base_dir
                        file_path = file_name
                else:
                    file_path = file_name
                
                # Enhanced metadata with correct source path
                meta_data = {
                    "filename": file_name,
                    "source_path": file_path,
                    "file_type": "text",
                    "file_size": file.stat().st_size
                }
                print(">> meta_data: ", meta_data)
            else:
                # Handle file-like objects
                file_name = getattr(file, 'name', 'unknown')
                if isinstance(file_name, str):
                    file_name = file_name.split(os.sep)[-1]  # Just the filename
                file.seek(0)
                file_contents = file.read().decode("utf-8")
                meta_data = {"filename": file_name}
            
            documents = [
                Document(
                    id=file_path if 'file_path' in locals() else file_name,
                    content=file_contents,
                    meta_data=meta_data
                )
            ]
            
            if self.chunk:
                chunked_documents = []
                for document in documents:
                    chunked_documents.extend(self.chunk_document(document))
                return chunked_documents
            return documents
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            return []


def setup_typesense_knowledge(
    collection_name: str,
    input_path: str,
    api_key: str,
    host: str = "localhost",
    port: int = 8108,
    protocol: str = "http",
    recreate: bool = False,
    append: bool = False,
    formats: List[str] = None
) -> TextKnowledgeBase:
    """Set up a Typesense knowledge base from files.
    
    Args:
        collection_name: Name of the Typesense collection
        input_path: Path to file or directory of files
        api_key: Typesense API key
        host: Typesense host
        port: Typesense port
        recreate: Whether to recreate the collection if it exists
        append: Whether to append new documents to existing collection
        formats: List of file extensions to include (defaults to common text formats)
        
    Returns:
        TextKnowledgeBase: Knowledge base ready for use with an agent
    """
    # Default supported formats
    if formats is None:
        formats = [".txt", ".md", ".text", ".log", ".csv"]
    
    # Initialize OpenAI embedder
    embedder = OpenAIEmbedder(id="text-embedding-3-small", dimensions=512)
    
    # Initialize Typesense database
    vector_db = TypesenseDb(
        name=collection_name,
        dimension=512,
        api_key=api_key,
        host=host,
        port=port,
        protocol=protocol,
        search_type=SearchType.hybrid,
        embedder=embedder
    )
    
    # Create or recreate collection if needed
    if recreate and vector_db.exists():
        print(f"Dropping existing collection '{collection_name}'")
        vector_db.drop()
    
    if not vector_db.exists():
        print(f"Creating collection '{collection_name}'")
        vector_db.create()
    
    # Create knowledge base with the input path
    path = Path(input_path)
    reader = ImprovedTextReader(base_dir=str(path) if path.is_dir() else None)
    
    knowledge_base = TextKnowledgeBase(
        path=input_path,
        vector_db=vector_db,
        chunk_size=500,
        chunk_overlap=50,
        reader=reader,
        formats=formats,
        num_documents=10
    )
    
    # Count documents in collection
    doc_count = vector_db.client.collections[collection_name].retrieve().get('num_documents', 0)
    
    # Load documents if collection is empty or recreating or appending
    if doc_count == 0 or recreate:
        print(f"Loading documents from {input_path}")
        knowledge_base.load()
        print("Documents loaded successfully")
    elif append:
        print(f"Appending new documents from {input_path} to existing collection ({doc_count} existing documents)")
        knowledge_base.load()
        print("Documents appended successfully")
    else:
        print(f"Using existing {doc_count} documents in collection")
        print("To append new documents, use the --append flag")
    
    return knowledge_base


def create_agent(knowledge_base, model_id="gpt-4.1-mini"): #model_id="claude-3-5-sonnet-20240620"):
    """Create an agent with the given knowledge base."""
    return Agent(
#        model=Claude(id=model_id),
        model=OpenAIChat(id=model_id),
        description="Document analysis assistant",
        instructions=[
            "Search the knowledge base for information relevant to the query.",
            "Base answers on information only found in the documents.",
            "ALWAYS include the FULL source_path (including directory name) in your responses.",
            "Format sources as 'Source: [complete source_path from meta_data]'",
            "The complete source_path should be exactly as it appears in the document meta_data.",
            "When citing multiple sources, list each full source_path on a new line.",
            "If information isn't found in the knowledge base, state this clearly."
        ],
        tools=[KnowledgeTools(knowledge_base)],
#        knowledge=knowledge_base,
        show_tool_calls=True,
        markdown=True
    )

def main():
    """Run the Typesense knowledge example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Typesense Knowledge Example')
    parser.add_argument('--collection', type=str, default='example-docs',
                       help='Name of the Typesense collection')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to file or directory to process')
    parser.add_argument('--recreate', action='store_true',
                       help='Recreate collection if it exists')
    parser.add_argument('--append', action='store_true',
                       help='Append new documents to existing collection')
    parser.add_argument('--localts', action='store_true',
                       help='Use local Typesense server instead of remote one')
    args = parser.parse_args()
    
    # Get Typesense credentials from environment variables
    api_key = os.getenv("TYPESENSE_API_KEY")
    host = os.getenv("TYPESENSE_HOST")
    port = os.getenv("TYPESENSE_PORT")
    protocol = os.getenv("TYPESENSE_PROTOCOL")
    
    if not api_key:
        raise ValueError("Please set TYPESENSE_API_KEY environment variable")
    
    if args.localts:
        # Use local Typesense server
        host = "localhost"
        port = 8108
        protocol = "http"
    else:
        # Use remote Typesense server
        if not host:
            raise ValueError("Please set TYPESENSE_HOST environment variable")
        if not port:
            raise ValueError("Please set TYPESENSE_PORT environment variable")
        if not protocol:
            raise ValueError("Please set TYPESENSE_PROTOCOL environment variable")
        port = int(port)  # Convert port to integer
    
    # Set up knowledge base
    knowledge_base = setup_typesense_knowledge(
        collection_name=args.collection,
        input_path=args.input,
        api_key=api_key,
        host=host,
        port=port,
        protocol=protocol,
        recreate=args.recreate,
        append=args.append
    )
    
    # Create agent
    agent = create_agent(knowledge_base)
    
    # Run interactive loop
    print("\n===== Document Assistant =====")
    print(f"Collection: {args.collection}")
    print(f"Knowledge source: {args.input}")
    print(f"Typesense server: {protocol}://{host}:{port}")
    print("Ask questions or type 'exit' to quit\n")
    
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if query.strip():
            print("\nSearching knowledge base...\n")
            agent.print_response(query, stream=True)


if __name__ == "__main__":
    main()
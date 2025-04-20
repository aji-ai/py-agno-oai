import chainlit as cl
from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
import os
from pathlib import Path
from typing import List, Union, IO, Any
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.document import Document
from agno.document.reader.text_reader import TextReader
from local_agno.vectordb.typesense import TypesenseDb, SearchType
from agno.tools.knowledge import KnowledgeTools
from agno.models.anthropic import Claude

# Get Typesense info from environment variables
typesense_host = os.environ.get('TYPESENSE_HOST')
typesense_port = os.environ.get('TYPESENSE_PORT')
typesense_protocol = os.environ.get('TYPESENSE_PROTOCOL')
typesense_api_key = os.environ.get('TYPESENSE_API_KEY')

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

@cl.on_chat_start
async def on_chat_start():

    api_key = typesense_api_key
    host = typesense_host
    port = typesense_port
    protocol = typesense_protocol
    
    # Set up knowledge base
    knowledge_base = setup_typesense_knowledge(
        collection_name='example-docs',
        input_path='.',
        api_key=api_key,
        host=host,
        port=port,
        protocol=protocol,
        recreate=False,
        append=False
    )
    agent = create_agent(knowledge_base)

    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")

    msg = cl.Message(content="")
    for chunk in await cl.make_async(agent.run)(message.content, stream=True):
        await msg.stream_token(chunk.get_content_as_string())
    
    await msg.send()
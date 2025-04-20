from textwrap import dedent
from rich.pretty import pprint

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
import os

# Use Render PostgreSQL connection
db_url = os.environ.get('POSTGRES_DB_URL')  
if not db_url:
    raise ValueError("POSTGRES_DB_URL environment variable is not set")

# Create a storage backend using the Postgres database
storage = PostgresAgentStorage(
    table_name="agent_sessions",
    db_url=db_url,
)

# Add storage to the Agent
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    storage=storage,
    session_id="fixed_id_for_demo",  # Set fixed session ID to test persistence
    add_history_to_messages=True,
    num_history_runs=3,  # Number of previous runs to include in history
)

# This pattern will demonstrate if memory is working
print("First question - should have no context:")
agent.print_response("What was my last question?")

print("\nAsking about France:")
agent.print_response("What is the capital of France?")

print("\nChecking memory - should remember previous question:")
agent.print_response("What was my last question?")

# View all messages to verify storage
print("\nAll messages in session:")
pprint(agent.get_messages_for_session())
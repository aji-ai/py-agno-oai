"""Run `pip install agno openai sqlalchemy psycopg2-binary` to install dependencies."""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
import os

# Define a tool that adds an item to the shopping list
def add_item(agent: Agent, item: str) -> str:
    """Add an item to the shopping list."""
    if item not in agent.session_state["shopping_list"]:
        agent.session_state["shopping_list"].append(item)
    return f"The shopping list is now {agent.session_state['shopping_list']}"


# Use Render PostgreSQL connection
db_url = os.environ.get('POSTGRES_DB_URL')
if not db_url:
    raise ValueError("POSTGRES_DB_URL environment variable is not set")

# delete the session from the database
# delete from agent_sessions where session_id = 'shopping_list_demo';
# my_storage = PostgresAgentStorage(
#     table_name="agent_sessions",
#     db_url=db_url
# )
# my_storage.delete_session("shopping_list_demo")
# print("deleted the session from the database")

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    # Fix the session id to continue the same session across execution cycles
    session_id="shopping_list_demo",
    # Initialize the session state with an empty shopping list
    session_state={"shopping_list": []},
    # Add a tool that adds an item to the shopping list
    tools=[add_item],
    # Store the session state in PostgreSQL database
    storage=PostgresAgentStorage(
        table_name="agent_sessions",
        db_url=db_url
    ),
    # Add the current shopping list from the state in the instructions
    instructions="Current shopping list is: {shopping_list}",
    # Important: Set `add_state_in_messages=True`
    # to make `{shopping_list}` available in the instructions
    add_state_in_messages=True,
    markdown=True,
)

# Example usage
print("\nChecking initial shopping list:")
agent.print_response("What's on my shopping list?", stream=True)
print(f"Session state: {agent.session_state}")

print("\nAdding items to shopping list:")
agent.print_response("Add milk, eggs, and bread", stream=True)
print(f"Session state: {agent.session_state}")

print("\nVerifying persistence - checking list again:")
agent.print_response("What's on my shopping list now?", stream=True)
print(f"Final session state: {agent.session_state}") 
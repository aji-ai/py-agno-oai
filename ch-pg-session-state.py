import chainlit as cl
import os
from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage

# Get database URL from shell environment variable
db_url = os.environ.get('POSTGRES_DB_URL')
if not db_url:
    raise ValueError("POSTGRES_DB_URL environment variable is not set")

# Define a tool that adds an item to the shopping list
def add_item(agent: Agent, item: str) -> str:
    """Add an item to the shopping list."""
    if item not in agent.session_state["shopping_list"]:
        agent.session_state["shopping_list"].append(item)
    return f"The shopping list is now {agent.session_state['shopping_list']}"


# Use Render PostgreSQL connection
@cl.on_chat_start
async def on_chat_start():
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
        show_tool_calls=True,
    )
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")

    msg = cl.Message(content="")
    for chunk in await cl.make_async(agent.run)(message.content, stream=True):
        await msg.stream_token(chunk.get_content_as_string())
    
    await msg.send()
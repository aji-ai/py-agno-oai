
# This is how I got started and didn't necessary go where I wanted to go

from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
import os
# Use your personal PostgreSQL connection
# db_url = "postgresql+psycopg://johnmaeda:nextnext@localhost:5432/postgres"
# Use Render PostgreSQL connection
db_url = os.environ.get('POSTGRES_DB_URL')
if not db_url:
    raise ValueError("POSTGRES_DB_URL environment variable is not set")

# Create a storage backend using the Postgres database
storage = PostgresAgentStorage(
    # store sessions in the ai.sessions table
    table_name="agent_sessions",
    # db_url: Postgres database URL
    db_url=db_url,
)

# Add storage to the Agent
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    storage=storage,
    instructions=dedent("""\
        You are an extremely concise responder to a question. One sentence no more.
    """),
    markdown=True
)

# Run the agent with multiple interactions to test storage
print("First interaction:")
session1 = agent.run("What is the capital of the moon?")
print(session1)

print("\nSecond interaction:")
session2 = agent.run("Tell me more about the moon's geography")
print(session2)

# After your existing code, add these verification steps:

# After your existing code, add these verification steps:

# 1. Verify the session was stored by retrieving it directly
print("\nVerifying storage:")
stored_session = storage.read(session1.session_id)
if stored_session:
    print(f"✅ Successfully retrieved session {stored_session.session_id}")
    if stored_session.memory and "runs" in stored_session.memory:
        runs = stored_session.memory["runs"]
        total_messages = sum(len(run.get("messages", [])) for run in runs if isinstance(run, dict))
        print(f"Number of messages in memory: {total_messages}")
        if runs and runs[0].get("messages"):
            print(f"First message content: {runs[0]['messages'][0].get('content')}")
    else:
        print("No messages in memory yet")
else:
    print("❌ Failed to retrieve session")

# 2. Verify we can get all sessions (without user_id filter)
all_sessions = storage.get_all_sessions()  # Removed user_id filter since RunResponse doesn't have it
print(f"\nTotal sessions: {len(all_sessions)}")

# 3. Verify the table structure
if storage.table_exists():
    print("\n✅ PostgreSQL table exists")
    print(f"Table name: {storage.schema}.{storage.table_name}")
else:
    print("\n❌ PostgreSQL table does not exist")
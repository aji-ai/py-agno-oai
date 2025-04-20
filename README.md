# py-agno-oai: Agno with OpenAI and Chainlit

This repository demonstrates how to build AI agents using Agno, OpenAI, and Chainlit with PostgreSQL storage for persistent conversations.

## Prerequisites

- Python 3.9+
- PostgreSQL database (we use Render's hosted PostgreSQL)
- OpenAI API key
- Render.com account (for hosted PostgreSQL)

## Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/py-agno-oai.git
cd py-agno-oai
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Add these to your shell (e.g., `~/.zshrc` or `~/.bashrc`):
```bash

For LLM:

export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."  # If using Anthropic models

For storage:

export POSTGRES_DB_URL="postgresql://user:password@your-db-host.render.com/dbname"

For vectorDB:

export TYPESENSE_API_KEY="..."         # If using Typesense search
export TYPESENSE_HOST="..."
export TYPESENSE_PORT="..."
export TYPESENSE_PROTOCOL="..."

```

After adding to your shell configuration, reload it:
```bash
source ~/.zshrc  # or source ~/.bashrc
```

5. **Verify Environment Setup**
```bash
# Check if variables are set
echo $POSTGRES_DB_URL
echo $OPENAI_API_KEY
```

## Available Scripts

### Basic Chainlit Agent
Run a simple Chainlit chat interface:
```bash
chainlit run ch-basicagno.py
```

### PostgreSQL Storage Agent
Run an agent with persistent storage:
```bash
chainlit run ch-pg-session-state.py
```

## Project Structure

- `ch-basicagno.py`: Basic Chainlit agent setup
- `ch-pg-session-state.py`: Agent with PostgreSQL storage
- `requirements.txt`: Project dependencies

## Environment Variables Reference

### Required Variables
- `POSTGRES_DB_URL`: Full PostgreSQL connection string
  - Format: `postgresql://user:password@host:port/dbname`
  - Example: `postgresql://johnmaeda:pass@db.render.com/mydb`
- `OPENAI_API_KEY`: Your OpenAI API key
  - Get from: [OpenAI API Keys](https://platform.openai.com/api-keys)

### Typesense Configuration
- `TYPESENSE_API_KEY`: Your Typesense API key
- `TYPESENSE_HOST`: Typesense host (default: "localhost")
- `TYPESENSE_PORT`: Typesense port (default: "8108")
- `TYPESENSE_PROTOCOL`: Protocol to use (default: "http")
This is unused:
- `TYPESENSE_COLLECTION_NAME`: Name of your Typesense collection

Example Typesense configuration:
```bash
# Typesense Configuration
export TYPESENSE_API_KEY="your_typesense_api_key"
export TYPESENSE_HOST="localhost"
export TYPESENSE_PORT="8108"
export TYPESENSE_PROTOCOL="http"
This is unused:
export TYPESENSE_COLLECTION_NAME="your_collection_name"
```

### Optional LLM API Keys
- `ANTHROPIC_API_KEY`: For Anthropic Claude models
  - Get from: [Anthropic Console](https://console.anthropic.com/)

## Development

To add new features:
1. Create a new Python file for your feature
2. Import necessary Agno components
3. Set up your agent with desired tools and storage
4. Run with Chainlit

## Troubleshooting

Common issues:
- **Database Connection**: 
  - Ensure your PostgreSQL URL is correct
  - Check if database is accessible (`psql $POSTGRES_DB_URL`)
  - Verify all required tables exist
- **API Key**: 
  - Verify your OpenAI API key is set and valid
  - Check API key permissions and quotas
- **Dependencies**: 
  - Make sure all requirements are installed: `pip list | grep agno`
  - Verify Python version: `python --version`
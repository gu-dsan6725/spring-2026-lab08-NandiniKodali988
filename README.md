# Advanced Agentic Patterns: Course Labs

This repository contains two hands-on lab assignments that teach fundamental concepts in AI agent development, from raw implementations to advanced multi-agent orchestration.

## Course Structure

### Problem 1: Streaming Stock Query Agent (50 points)

**Directory**: [streaming-stock-agent/](streaming-stock-agent/) | **Exercise**: [EXERCISE.md](streaming-stock-agent/EXERCISE.md)

Build a conversational stock market agent from scratch using FastAPI and raw LLM tool calling (no frameworks). This lab demonstrates how LLMs can call functions through tool schemas, how streaming responses work with Server-Sent Events (SSE), and how to maintain multi-turn conversation state. You'll implement a stock comparison tool by defining its JSON schema, writing the function, and registering it with the agent. The system uses Yahoo Finance for real-time stock data and Groq's Qwen3-32b model for reliable function calling. This raw implementation teaches the fundamentals of tool calling and streaming before moving to higher-level frameworks.

**Key Learning Objectives**:
- Understanding how LLM tool calling works at a fundamental level
- Implementing streaming responses with Server-Sent Events (SSE)
- Managing multi-turn conversation state with in-memory circular buffers
- Defining tool schemas with JSON Schema for function calling
- Building FastAPI services with async Python

### Problem 2: Multi-Agent Financial Orchestrator (100 points)

**Directory**: [personal-financial-analyst/](personal-financial-analyst/) | **Exercise**: [EXERCISE.md](personal-financial-analyst/EXERCISE.md)

Implement a hierarchical multi-agent system using Claude Agent SDK that demonstrates the orchestrator-workers pattern. An orchestrator agent fetches financial data from MCP servers, analyzes transactions, and coordinates three specialized sub-agents running in parallel for research, negotiation, and tax optimization. This lab teaches advanced concepts including MCP protocol integration for external data sources, cost-optimized model selection (Sonnet for orchestration, Haiku for execution), file-based agent communication, dynamic tool permissions with callbacks, and streaming output with user feedback. The architecture achieves 75% cost reduction while maintaining quality through intelligent orchestration and efficient parallel execution.

**Key Learning Objectives**:
- Multi-agent orchestration with parallel execution
- MCP protocol integration for external data sources
- Cost-optimized model selection strategy
- Dynamic tool permissions with callbacks
- Streaming output with progress feedback
- File-based agent communication patterns
- When to use tools vs sub-agents

## Prerequisites

- Python 3.11+
- API keys:
  - **Anthropic Claude** (Required for Problem 2): https://console.anthropic.com/
  - **Groq** (Required for Problem 1): https://console.groq.com/ - FREE tier, no credit card

## Quick Start

### 1. Install uv (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Install Dependencies

```bash
cd advanced-agentic-patterns
uv sync
```

### 3. Configure API Keys

Each lab has its own `.env.example` file:

```bash
# For Problem 1
cd streaming-stock-agent
cp .env.example .env
# Edit .env and add GROQ_API_KEY

# For Problem 2
cd ../personal-financial-analyst
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY
```

### 4. Work on Labs

**Problem 1**:
```bash
cd streaming-stock-agent
./start.sh
# Follow instructions in EXERCISE.md
```

**Problem 2**:
```bash
cd personal-financial-analyst
# Start MCP servers
cd mcp_servers
./start_servers.sh

# Follow instructions in EXERCISE.md
```

## Project Structure

```
advanced-agentic-patterns/
├── README.md                              # This file
├── pyproject.toml                         # Project dependencies
│
├── streaming-stock-agent/                 # Problem 1 (50 points)
│   ├── EXERCISE.md                        # Lab instructions
│   ├── README.md                          # Technical documentation
│   ├── architecture.md                    # System architecture
│   ├── agent.py                           # Tool definitions (TODO: add compare_stocks)
│   ├── main.py                            # FastAPI server
│   ├── session_manager.py                 # Conversation state
│   ├── test_client.py                     # Test harness
│   └── start.sh                           # Server startup script
│
└── personal-financial-analyst/            # Problem 2 (100 points)
    ├── EXERCISE.md                        # Lab instructions
    ├── README.md                          # Technical documentation
    ├── agent/
    │   ├── architecture.md                # Comprehensive implementation guide
    │   ├── financial_orchestrator.py      # Main file (TODO: implement)
    │   └── prompts/                       # Agent prompts
    └── mcp_servers/                       # FastMCP servers
        ├── bank_server.py                 # Bank transaction data
        ├── credit_card_server.py          # Credit card data
        └── mock_data/                     # Sample CSV data
```

## Grading

### Problem 1: Streaming Stock Agent (50 points)

**Deliverables**:
- [x] Implemented `_compare_stocks()` function in [agent.py](streaming-stock-agent/agent.py)
- [x] Tool schema added to `STOCK_TOOLS` list
- [x] Test output files: `output1.txt`, `output2.txt`, `output3.txt`
- [x] Code follows project standards (type hints, docstrings, error handling)

**Evaluation Criteria**:
- Correct tool schema definition (15 points)
- Working function implementation (20 points)
- Multi-turn conversation support (10 points)
- Code quality and standards (5 points)

### Problem 2: Multi-Agent Financial Orchestrator (100 points)

**Deliverables**:
- [x] Completed `financial_orchestrator.py` with all TODO sections implemented
- [x] Output files created:
  - `data/raw_data/bank_transactions.json`
  - `data/raw_data/credit_card_transactions.json`
  - `data/agent_outputs/research_results.md`
  - `data/agent_outputs/negotiation_scripts.md`
  - `data/agent_outputs/tax_analysis.md`
  - `data/final_report.md`
- [x] Code passes syntax validation

**Evaluation Criteria**:
- Subscription detection implementation (10 points)
- MCP server configuration and data fetching (20 points)
- Sub-agent definitions (20 points)
- Orchestrator configuration (20 points)
- Agent execution and streaming (20 points)
- Code quality and standards (10 points)

## Resources

### Official Documentation
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) - Multi-agent orchestration
- [FastMCP Documentation](https://github.com/gofastmcp/fastmcp) - MCP server framework
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification

### Learning Resources
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - Anthropic's guide
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents) - Agent patterns

## Troubleshooting

### MCP Server Connection Issues
```bash
# Check if servers are running
curl http://127.0.0.1:5001/health
curl http://127.0.0.1:5002/health

# Check for port conflicts
lsof -i :5001
lsof -i :5003
```

### API Key Issues
```bash
# Verify keys are set
echo $ANTHROPIC_API_KEY
echo $GROQ_API_KEY
```

### Claude Agent SDK Environment
The Claude Agent SDK cannot run inside Claude Code sessions. Run in a regular terminal or:
```bash
unset CLAUDECODE
uv run python financial_orchestrator.py ...
```

## Submission

Submit to your course platform:
1. Your completed code files
2. All output files (output1.txt, output2.txt, output3.txt for Problem 1)
3. All data outputs for Problem 2 (6 files in data/ directory)
4. A brief write-up answering the reflection questions in each EXERCISE.md

## Solutions Branch

Reference implementations are available in the `solutions` branch:
```bash
git checkout solutions
```

**Important**: Use solutions only for reference if you're stuck. Attempt to implement on your own first for maximum learning.

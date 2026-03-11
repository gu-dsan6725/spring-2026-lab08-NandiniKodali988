# Student Exercise: Add Stock Comparison Tool

## Objective
Add a new tool to the stock query agent that allows users to compare two stocks side-by-side. This exercise reinforces understanding of:
- Tool schema definition with JSON Schema
- Tool function implementation
- Tool registration in the agent system
- Testing streaming responses with multi-turn conversations

## Background
The current agent has tools to get stock prices, price history, and basic company info. Users often want to compare two stocks directly (e.g., "Compare AAPL and MSFT" or "Which is better, Tesla or Ford?"). Your task is to add this capability.

## Requirements

### 1. Implement the Tool Function
Create a new function `_compare_stocks()` in [agent.py](agent.py):

**Function Signature:**
```python
def _compare_stocks(
    symbol1: str,
    symbol2: str
) -> Dict[str, Any]:
    """Compare two stocks side-by-side.

    Args:
        symbol1: First stock symbol (e.g., 'AAPL')
        symbol2: Second stock symbol (e.g., 'MSFT')

    Returns:
        Dictionary with comparison data for both stocks
    """
```

**Expected Return Format:**
```python
{
    "comparison": {
        "symbol1": "AAPL",
        "symbol2": "MSFT",
        "stock1": {
            "symbol": "AAPL",
            "current_price": 175.50,
            "company_name": "Apple Inc.",
            "market_cap": "2.8T"
        },
        "stock2": {
            "symbol": "MSFT",
            "current_price": 420.30,
            "company_name": "Microsoft Corporation",
            "market_cap": "3.1T"
        }
    }
}
```

### 2. Define the Tool Schema
Add a new tool definition to the `STOCK_TOOLS` list in [agent.py](agent.py):

**Required Fields:**
- `name`: "compare_stocks"
- `description`: Clear explanation of what the tool does
- `parameters`: JSON Schema with two required string parameters (symbol1, symbol2)
- `function`: Reference to your `_compare_stocks` function

**JSON Schema Structure:**
```python
{
    "name": "compare_stocks",
    "description": "Compare two stocks side-by-side...",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol1": {
                "type": "string",
                "description": "First stock symbol to compare"
            },
            "symbol2": {
                "type": "string",
                "description": "Second stock symbol to compare"
            }
        },
        "required": ["symbol1", "symbol2"]
    },
    "function": _compare_stocks
}
```

### 3. Test Your Implementation
Test with these example queries:

**Query 1: Direct comparison**
```bash
uv run python test_client.py "Compare AAPL and MSFT"
```

**Query 2: Natural language**
```bash
uv run python test_client.py "Which is better, Tesla or Ford?"
```

**Query 3: Multi-turn conversation**
```bash
uv run python test_client.py "Show me AAPL stock"
# Then in a new query with same session:
uv run python test_client.py "Now compare it with GOOGL"
```

## Implementation Hints

- Look at existing `_get_stock_price()` function - you can call it twice for each stock
- Remember to add try/except for error handling
- Private functions go at the top of the file with other tool functions
- Add your tool schema to the `STOCK_TOOLS` list after implementing the function

## Expected Behavior

When you query: **"Compare Apple and Microsoft"**

The agent should:
1. Recognize this as a comparison request
2. Call the `compare_stocks` tool with `symbol1="AAPL"` and `symbol2="MSFT"`
3. Execute your `_compare_stocks()` function
4. Return formatted comparison data
5. Stream a natural language response like:

```
[Calling tool: compare_stocks with args {'symbol1': 'AAPL', 'symbol2': 'MSFT'}]
[Tool calls completed]

Here's the comparison between Apple (AAPL) and Microsoft (MSFT):

- AAPL: $175.50 (Market Cap: $2.8T)
- MSFT: $420.30 (Market Cap: $3.1T)

Microsoft currently has a higher stock price and larger market capitalization than Apple.
```

## Verification Checklist

Before submitting, verify:
- [x] Function `_compare_stocks()` is implemented and placed at top of file
- [x] Tool schema is added to `STOCK_TOOLS` list
- [x] Tool has clear description that helps the LLM understand when to use it
- [x] Both parameters are marked as required in the schema
- [x] Error handling for invalid stock symbols
- [x] Code follows project standards (type hints, docstrings)
- [x] Tested with at least 2 different stock comparisons
- [x] Multi-turn conversation works (agent remembers context)
- [x] Test output saved: `output1.txt` (AAPL vs MSFT comparison)
- [x] Test output saved: `output2.txt` (TSLA vs F comparison)
- [x] Test output saved: `output3.txt` (multi-turn conversation test)

## Testing Commands

**Start the server:**
```bash
cd /home/ubuntu/repos/advanced-agentic-patterns/streaming-stock-agent
./start.sh
```

**Run test queries (in a new terminal):**
```bash
# Test 1: Basic comparison (save to output1.txt)
uv run python test_client.py "Compare AAPL and MSFT" > output1.txt

# Test 2: Different stocks (save to output2.txt)
uv run python test_client.py "Compare Tesla and Ford" > output2.txt

# Test 3: Multi-turn with context (save to output3.txt)
uv run python test_client.py > output3.txt
```

## Questions to Consider

After completing the exercise, reflect on:
1. How did the LLM know to call your new tool?
2. What happens if the tool schema description is unclear?
3. How does the agent decide between `compare_stocks` and `get_stock_price`?
4. How would you add validation for parameter values?
5. What if the user asks to compare 3 stocks instead of 2?

# Project Notes

## Project overview

This project is a small Python chat assistant that uses an OpenAI-compatible chat-completions API. It can respond with model-generated text and call local Python functions for the current time, weather in a city, and the configured city (currently hard-coded as Hyderabad).

### Main files

- `simple_ai_assistance.py` — loads configuration, maintains chat history, streams model responses, and runs tool calls.
- `tools.py` — implements the tools, describes them to the model with JSON schemas, and dispatches requested calls to Python functions.
- `utilities/utilities.py` — provides optional debug logging controlled by `DEBUG`.

## Setup and run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install openai python-dotenv requests
```

Create a local `.env` file with configuration for your OpenAI-compatible endpoint and model. The code reads `BASE_URL`, `API_KEY`, and `MODEL`; keep provider credentials out of source control. For local Ollama use, the base URL is commonly `http://localhost:11434/v1`, and the model name must match one available in your Ollama installation.

Start Ollama in one terminal and run the assistant in another:

```bash
ollama serve
python simple_ai_assistance.py
```

The assistant exits when the user enters `exit`, `quit`, `bye`, or `goodbye`. Set `DEBUG=True` in `.env` to enable tool and step logs.

## How a request flows

1. The chat loop appends the user message to the in-memory `messages` history.
2. `stream_chat()` sends the history and `TOOLS` schemas to the model and consumes the streamed response.
3. Text fragments are printed as they arrive. Tool-call fragments are accumulated by tool index because names and JSON arguments may arrive across multiple chunks.
4. If the model requested tools, the assistant tool-call message is added to history. `run_tool()` parses the arguments and dispatches the function by name through `FUNCTION_NAME_TO_FUNCTION_MAP`.
5. Each tool result is added to the conversation, and the model is called again to produce a user-facing response.
6. `MAX_RUN_STEPS` limits a single turn to three model/tool cycles.

## Interview questions and answer notes

### 1. What does this project demonstrate?

It demonstrates an LLM assistant that can combine generated text with calls to ordinary Python functions. The model chooses from tools described in a structured schema, while the application executes the corresponding local code and sends results back to the model.

### 2. How is tool calling different from asking the model to answer directly?

The model can request an action using a structured function name and arguments. The application—not the model—executes the Python function, then provides its result to the model. This lets the assistant use live or application-specific data that may not be in the model's training data.

### 3. How does the model know which tools are available?

`tools.py` defines the `TOOLS` metadata: tool names, descriptions, parameter types, and required fields. `simple_ai_assistance.py` passes that metadata with each chat-completion request. The system prompt further guides when tools should be used.

### 4. How does a tool request become a Python function call?

The streaming code joins partial tool-call names and arguments, then converts the collected calls into the assistant-message payload format. `run_tool()` parses the JSON arguments, looks up the function name in `FUNCTION_NAME_TO_FUNCTION_MAP`, and invokes the matching Python function.

### 5. Why does the application keep conversation history?

The model API does not automatically retain this application's earlier turns. Keeping system, user, assistant, and tool messages in `messages` gives each request the context needed to continue the conversation and interpret tool results. The history is in memory, so it is lost when the process exits.

### 6. Why stream the response, and what extra work does tool calling add?

Streaming lets the user see generated text before the full response is ready. Tool-call data can also arrive in fragments, so the client accumulates each call by its stream index and combines its function name and arguments before dispatching it.

### 7. Why use environment variables and a `.env` file?

They keep endpoint, model, and credential configuration out of the source code, making local setup easier to change. `python-dotenv` loads the values and `os.getenv()` reads them. The `.env` file is ignored by Git and should not be committed.

### 8. What does each tool do, and what are its dependencies?

- `get_current_time()` uses Python's `datetime` module.
- `get_city_weather(city)` requests a current-weather string from `wttr.in` using `requests`, with a 10-second timeout.
- `get_city()` returns the fixed string `"Hyderabad"`.

The weather tool depends on network access and an external service; the other two do not.

### 9. How does the assistant handle tool errors and repeated tool requests?

`run_tool()` catches errors from JSON parsing, lookup, and execution and returns an error string to the conversation. The outer loop allows at most three tool-calling cycles per user turn and then displays a fallback message if the model continues requesting tools. This is basic handling; production systems should validate arguments and distinguish expected failures from unexpected bugs.

### 10. What are the current limitations?

- Conversation history is only in memory and grows throughout the process.
- The configured city is hard-coded, and the weather lookup depends on an external service.
- Tool schemas and implementations are manually maintained and have limited input validation.
- The tool runner catches broad exceptions, so failures are returned as text rather than handled by specific error types.
- There is no authentication or authorization layer for tools; this is suitable only for the simple local functions shown.
- API compatibility details can vary by provider, so request and tool-result message formats should be tested against the chosen endpoint.

### 11. How would you extend the project?

Add a tool by implementing a Python function, describing its parameters in `TOOLS`, and registering it in `FUNCTION_NAME_TO_FUNCTION_MAP`. Further improvements could include schema validation, tests for tool dispatch and streaming fragments, more specific error handling, bounded or persistent conversation history, and configurable city or weather-provider settings.

## Key takeaway

The model proposes tool calls; the Python application parses and dispatches those calls, executes the functions, and returns their results to the model. This separation is the core idea behind the project's tool-using assistant workflow.

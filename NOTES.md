# Project Notes

## Project overview

This project is a terminal-based Python assistant that uses an OpenAI-compatible chat-completions API. It can stream normal responses, call Python tools, and answer company questions using retrieval-augmented generation (RAG) over local text documents.

The main assistant tools provide the current time, weather for a supplied city, a configured city value, and semantic search of the company knowledge base. The knowledge-base search returns a relevant source passage for the assistant to use and cite.

## Main files

- `simple_ai_assistance.py` — configures the chat client and system prompt, keeps in-memory conversation history, streams model output, and coordinates tool calls.
- `tools.py` — defines the available tool schemas, their Python implementations, and the name-to-function dispatch map.
- `rag.py` — reads and chunks knowledge-base text, creates or loads embeddings, persists them in a JSON vector store, and ranks search results.
- `knowledge_base/` — local policy and process documents used for retrieval, plus the generated/persisted vector store.
- `utilities/utilities.py` — provides debug logging when `DEBUG=True`.
- `coding_snippets/embedding.py` — creates embeddings for example strings and prints vector information.
- `coding_snippets/sementic_search.py` and `doc_search.py` — embed text or whole documents and rank them by cosine similarity.
- `coding_snippets/chunk_search.py` — splits documents into paragraphs, caches chunk embeddings in JSON, and searches them.
- `coding_snippets/chroma_simple.py` — demonstrates storing and querying sample embeddings in a persistent Chroma collection.
- `coding_snippets/stream_text.py` and `stream_tools.py` — focused examples of streaming text and assembling streamed tool-call fragments.
- `requirements.txt` — Python package dependencies.

## Setup and run

Create and activate a virtual environment, then install the packages used by the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a local `.env` file. Set `BASE_URL`, `API_KEY`, and `MODEL` for the OpenAI-compatible chat and embedding endpoint. Also set `EMBED_MODEL`, `KB_FOLDER`, and `VECTOR_STORE_FILE` for retrieval. For example, with an Ollama server:

```env
BASE_URL=http://localhost:11434/v1
API_KEY=ollama
MODEL=<installed-chat-model>
EMBED_MODEL=<installed-embedding-model>
KB_FOLDER=knowledge_base
VECTOR_STORE_FILE=knowledge_base/vector_store.json
DEBUG=False
```

Use model names available from the configured endpoint. Keep provider credentials out of source control. Start Ollama if using it locally, then run:

```bash
ollama serve
python simple_ai_assistance.py
```

Enter `exit`, `quit`, `bye`, or `goodbye` to leave the chat. The first run creates embeddings and writes the vector store if the configured file does not exist; later runs load that file. Set `DEBUG=True` to print tool and retrieval logs.

The scripts in `coding_snippets/` are standalone examples. They also need the environment variables for their configured API endpoint and model. In particular, embedding and retrieval examples need `EMBED_MODEL`; chunk search also needs `KB_FOLDER` and `VECTOR_STORE_FILE`.

## Request and tool flow

1. The chat loop appends the user's message to the in-memory history.
2. `stream_chat()` sends the history and `TOOLS` definitions to the model. It prints text as it arrives and accumulates streamed tool-call name and argument fragments by tool index.
3. If the model requested tools, the assistant tool-call message is added to history. `run_tool()` parses the arguments, looks up the function in `FUNCTION_NAME_TO_FUNCTION_MAP`, and invokes it.
4. Tool results are added to history as tool messages. The assistant makes another model request to form a response using those results.
5. `MAX_RUN_STEPS` limits a turn to three model/tool cycles.

The model proposes a tool call; the Python application is responsible for executing it and returning its result. Conversation history exists only in memory and is lost when the process exits.

## How knowledge retrieval works

1. `rag.py` loads `.txt` files from `KB_FOLDER` and splits each document on blank lines, assigning each non-empty paragraph a `source` and `chunk_id`.
2. It creates one embedding per chunk through the configured OpenAI-compatible embeddings endpoint.
3. It saves chunk text, source metadata, and vectors as JSON in `VECTOR_STORE_FILE`. If that file already exists, it loads the saved vectors instead of rebuilding them.
4. For a search, it embeds the query, computes cosine similarity against the stored chunk vectors using NumPy, sorts by score, and returns the highest-ranked results.
5. `search_knowledge_base()` in `tools.py` requests the top one result and formats it with the source filename and chunk number. The system prompt instructs the assistant to use this for Zorvex company questions, answer only from the retrieved passage, cite the source, and use a fixed response if the answer is not present.

## Interview questions and answer notes

### 0. How does a request move through this repository?

`simple_ai_assistance.py` adds the user message to `messages` and sends that history, the system message, and `TOOLS` to the chat API. If the model requests a tool, `run_tool()` looks up and calls the matching Python function. Tool output is appended as a tool message and the model is called again. The loop stops after a normal text response or after three tool/model cycles.

### 1. What does this project demonstrate?

It demonstrates an LLM assistant that combines streamed chat responses, function/tool calling, and RAG. The model decides when a tool is useful, while application code runs the tool and returns its result.

### 2. What is RAG, and how is it used here?

Retrieval-augmented generation grounds an LLM response in external documents. Here, local text documents are split into paragraph chunks and embedded. A user's question is embedded too; cosine similarity retrieves the closest chunk, which is passed back to the assistant as a tool result.

### 3. What is an embedding?

An embedding is a numeric vector representing text in a semantic space. The project uses an embedding model through the configured API to represent both chunks and queries, then compares their vectors to find semantically similar text.

### 4. How does semantic search rank results?

`rag.search()` embeds the query, computes cosine similarity for every stored chunk, sorts results in descending score order, and returns up to `top_k` matches. The assistant's knowledge-base tool currently asks for one result.

### 5. Why split documents into chunks?

Paragraph-sized chunks keep retrieved context focused and preserve a source filename and chunk number for citation. The current implementation splits on blank lines; it does not enforce a token or character limit or combine adjacent short paragraphs.

### 6. How are embeddings reused between runs?

`load_chunks()` loads the JSON vector store when it already exists. Otherwise, it reads the configured text files, creates chunks and embeddings, and saves the result. The current implementation does not check whether source documents or the embedding model have changed since that cache was created.

### 7. How does a streamed tool call become a Python function call?

Tool calls can arrive as partial fragments. `stream_chat()` combines each call's name and argument fragments by tool index. It then builds the assistant tool-call message; `run_tool()` decodes the JSON arguments, dispatches by function name, and returns the function result.

### 8. How does the assistant handle company questions differently from general questions?

The system prompt asks the model to answer general questions from its own knowledge and use the knowledge-base search tool for company topics. For company answers, it directs the model to rely only on retrieved text, cite the source, and return a specified message when the passage does not contain the answer.

### 9. Why use an OpenAI-compatible API and environment variables?

The OpenAI client can send chat and embedding requests to a compatible service by setting its base URL and model names. Environment variables allow the endpoint, credentials, models, and data paths to be configured without hard-coding them in the Python source.

### 10. What are the available tools?

- `get_current_time()` returns the local date and time.
- `get_city_weather(city)` requests a weather string from `wttr.in` with a 10-second timeout.
- `get_city()` returns the fixed value `"Hyderabad"`.
- `search_knowledge_base(query)` retrieves the top-ranked paragraph from the local knowledge base.

Tool schemas are declared in `TOOLS`; the corresponding implementations are registered in `FUNCTION_NAME_TO_FUNCTION_MAP`.

### 11. What happens when a tool fails or the model keeps requesting tools?

`run_tool()` catches exceptions and returns an error string to the conversation. The assistant loop allows at most three tool-calling cycles per user turn and then displays a fallback response. These are basic safeguards; production code should use more specific error handling, validate tool arguments, and consider retries and timeouts for model requests.

### 12. What are the project's current limitations?

- Conversation history is only in memory and grows during the process.
- The vector store is reused based only on whether its file exists; it can become stale when documents or the embedding model change.
- Search uses cosine similarity over every chunk in the JSON file, without a vector database, metadata filters, or an explicit relevance threshold.
- Retrieval returns only one chunk, so relevant information split across passages may be missed.
- Paragraph splitting does not account for model token limits or document structure beyond blank lines.
- The city tool returns a hard-coded value, while weather lookup depends on an external service.
- The system prompt refers to `get_user_city`, but the registered tool is named `get_city`; the prompt and tool schema should be kept in sync.
- Tool errors are caught broadly and represented as text; tool argument validation is limited.
- The system prompt guides model behavior but is not an enforcement boundary. Retrieved content and model-produced tool requests should not be treated as inherently trusted.

### 13. How would you improve or extend it?

Add tests for chunking, cosine ranking, tool dispatch, and fragmented streaming calls. Rebuild or version the vector store when source documents or the embedding model change; validate vector dimensions and handle zero-length vectors. Consider returning multiple relevant chunks with a minimum relevance threshold, adding bounded or persistent chat history, making the city configurable, and using specific error types and input validation.

### 14. What is the difference between the retrieval examples?

`doc_search.py` embeds whole text documents for a search run. `chunk_search.py` splits each document into paragraphs and saves its chunks and vectors in a JSON file for reuse. `rag.py` uses a similar paragraph-and-JSON approach for the main assistant. `chroma_simple.py` is a separate demonstration that stores sample texts and embeddings in a persistent Chroma collection and queries that collection.

### 15. What is the purpose of cosine similarity?

Cosine similarity compares the direction of two vectors, rather than their raw magnitude. The code computes the dot product divided by both vector norms, then sorts scores from highest to lowest. It is the ranking method in the NumPy search examples; the Chroma example configures its collection to use cosine distance.

### 16. What does streaming change in the user experience?

With `stream=True`, the client receives response fragments as they are generated. `stream_text.py` prints text fragments immediately and also collects the complete response. In the assistant, `stream_chat()` prints text as it arrives and separately reconstructs any tool calls from their fragments.

### 17. Why must streamed tool-call fragments be assembled?

A tool name, ID, or JSON arguments can arrive over multiple chunks. The code groups fragments by the tool-call index and concatenates the name and argument pieces before dispatching the completed call. The index lets it keep multiple calls separate.

### 18. How are tool definitions related to tool implementations?

`TOOLS` describes tool names, descriptions, and JSON parameter schemas for the model. `FUNCTION_NAME_TO_FUNCTION_MAP` maps those names to Python functions. `run_tool()` parses the model-provided argument JSON and dispatches to the matching implementation. Keeping the schema and map aligned is essential.

### 19. What is the role of the system prompt?

The system message sets the assistant's intended behavior. In this project it distinguishes general questions from company questions, tells the model when to search the knowledge base, constrains company answers to retrieved text, requests a source citation, and sets a response length. It is guidance to the model, not a substitute for application-side validation.

### 20. Does the assistant remember previous runs?

No. The `messages` list retains the conversation only while the process is running. It is not written to disk, and the list can grow during a long session. Persistent or bounded conversation history would be a separate feature.

### 21. What work happens before the first RAG search?

`tools.py` imports `search` from `rag.py`. `rag.py` calls `load_chunks()` at module import time, so it loads an existing JSON vector store or reads the text files, requests embeddings, and writes the store if no cache file exists. This means importing the tools can require valid retrieval configuration and a reachable embedding endpoint.

### 22. What does the Chroma example demonstrate that the JSON examples do not?

The example uses `chromadb.PersistentClient` to keep a collection on disk and lets Chroma perform the nearest-neighbor query. The NumPy examples explicitly compare a query vector with each stored vector in Python. The Chroma example uses hard-coded sample records; it is a demonstration, not the storage backend wired into `rag.py`.

### 23. What is the difference between chat and embedding models here?

The chat model, configured by `MODEL`, generates assistant responses and can request tools. The embedding model, configured by `EMBED_MODEL`, turns text into vectors for retrieval. The code sends these requests through the configured OpenAI-compatible API endpoint; the chosen service must provide compatible chat and embedding endpoints and support the requested models.

### 24. How would you test this project without calling a live model?

Test deterministic parts independently: paragraph splitting, cosine-score ordering, result formatting, JSON store loading, and tool-name dispatch can be tested with small local fixtures and mocked API responses. Keep live endpoint checks as integration tests, since they depend on network/service availability and configured model names.

### 25. What should be improved before using this as a production service?

Add automated tests and input/schema validation; use targeted error handling and clear service timeouts; track vector-store version, source-document changes, and embedding model; handle empty or mismatched vectors; evaluate retrieval quality and relevance thresholds; bound or persist conversation history; and verify provider-specific tool-call message formats. Retrieved text and model-generated tool requests should be treated as untrusted input.

### 26. Why might the assistant return an incorrect or unsupported company answer?

Retrieval is approximate, and the main assistant requests only the single top-ranked chunk. The matching passage may omit part of the answer or not be relevant enough. The system prompt asks the model to cite sources and avoid outside knowledge, but the code does not independently verify that every claim is supported. Improving this requires retrieval evaluation, relevance checks, and possibly returning multiple passages.

## Short project pitch

> This is a terminal-based Python assistant built around an OpenAI-compatible API. It streams model responses, lets the model request a small set of Python tools, and uses embeddings plus cosine-similarity search to retrieve local company-document passages. The main RAG example persists chunk vectors as JSON and returns a source-labelled passage to the assistant. The repository also contains smaller examples for streaming, whole-document search, chunk search, and persistent Chroma storage.

## Interview preparation checklist

- Explain the request path from user input through streamed completion, optional tool execution, tool result, and final completion.
- Draw the RAG path: source documents, paragraph chunks, embeddings, vector persistence, query embedding, similarity ranking, retrieved context.
- Distinguish the chat model from the embedding model and explain where each is used.
- Be ready to discuss trade-offs: chunk size, top-k, similarity thresholds, cache invalidation, latency, and context limits.
- State current limitations honestly, especially in-memory chat history, one-chunk retrieval, and configuration/service dependencies.
- Describe what you would test and improve next; do not claim a vector database is used by the main `rag.py` path.

## Key takeaway

The project separates three responsibilities: the model generates responses and proposes tool calls; Python executes tools and manages the conversation; and the RAG layer retrieves local source passages to ground company-specific answers.

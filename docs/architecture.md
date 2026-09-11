# Architecture

## Purpose

The Autonomous Financial Research Agent demonstrates how a goal-oriented LLM can coordinate multiple tools, retrieve grounded context, tolerate partial failures, and produce a structured research briefing.

## High-Level Flow

```text
User request
   |
   v
Agent node (LLM + agent charter)
   |
   |-- needs market data? ------> Stock price / history tools
   |
   |-- needs current context? --> Financial news search
   |
   |-- needs tone signal? ------> Sentiment analysis
   |
   |-- needs private context? --> RAG research retriever (optional)
   |
   v
Tool results returned to LangGraph state
   |
   v
Agent evaluates remaining information needs
   |
   +----> repeat tool loop as needed
   |
   v
Final structured research briefing
```

## Core Components

### 1. Agent Charter
The system prompt defines the agent's mission, required analysis sections, transparency requirements, and behavior when data or tools are unavailable.

### 2. LangGraph State Management
A `StateGraph` maintains conversation state and conditionally routes between:

- **Agent node** — reasons about the request and decides whether to call tools.
- **Tool node** — executes the requested tool calls.
- **End state** — returns the final response when the model no longer requests tools.

This creates an autonomous loop:

```text
Agent -> Tools -> Agent -> ... -> Final Response
```

### 3. Public Market Data
`yfinance` provides current pricing, market capitalization, trading ranges, volume, and historical stock performance.

### 4. Financial News Search
Tavily is used to retrieve recent financial and company news so the agent can supplement structured market data with current context.

### 5. Sentiment Analysis
Financial text can be evaluated by the LLM for positive, negative, or neutral sentiment. A simple keyword fallback is included so the workflow can continue when the LLM call fails.

### 6. Retrieval-Augmented Generation (Optional)
The RAG pipeline loads local PDF research documents, splits them into overlapping chunks, generates embeddings, stores them in ChromaDB, and retrieves semantically relevant passages for a research question.

```text
PDF documents
   -> text extraction
   -> recursive chunking
   -> embeddings
   -> ChromaDB vector store
   -> similarity retrieval
   -> grounded LLM response
```

### 7. Memory
An optional LangGraph `MemorySaver` checkpoint allows follow-up questions to retain conversational context within the same thread.

## Error-Handling Principles

The portfolio implementation follows these design principles:

- Do not stop the full workflow because one source fails.
- State missing information explicitly.
- Continue with available evidence when appropriate.
- Reduce confidence when important evidence is unavailable.
- Do not invent missing facts.

## Security and Responsible Use

- API keys are read from environment variables and are never hard-coded in source files.
- Proprietary documents should not be committed to a public repository.
- Public demos should use public or synthetic information.
- Outputs are educational research demonstrations, not personalized financial advice.

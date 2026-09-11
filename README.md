# Autonomous Financial Research Agent

Agentic AI portfolio project demonstrating autonomous financial research, Retrieval-Augmented Generation (RAG), tool orchestration, sentiment analysis, error handling, source transparency, and LangGraph-based workflow management.

## Overview

This project explores how an AI agent can move beyond reactive question answering and operate as a goal-oriented financial research assistant. The agent proactively gathers market data, reviews historical performance, searches current financial news, analyzes sentiment, retrieves relevant information from private research documents, and synthesizes the findings into a structured investment research report.

The project was developed as part of my Johns Hopkins University Certificate Program in Agentic AI and is presented here as a portfolio demonstration of my hands-on work with autonomous AI systems.

## Business Problem

Investment research often requires analysts to manually gather information from multiple disconnected sources, including stock-price data, financial news, analyst reports, and strategic company documents. This process can be slow, repetitive, difficult to scale, and vulnerable to inconsistent analysis.

This agent was designed to demonstrate how autonomous AI workflows can coordinate multiple specialized tools and data sources to support faster, more structured research while maintaining transparency about sources, limitations, and confidence.

## What I Built

The solution demonstrates a goal-oriented agent that can:

- Gather current stock-price and market-cap information
- Analyze historical stock performance
- Search financial news
- Perform sentiment analysis
- Query private analyst and AI-initiative documents through RAG
- Select and orchestrate tools autonomously
- Continue operating when individual tools fail
- Cite information sources and acknowledge data gaps
- Produce structured research reports with risk analysis and recommendation confidence
- Compare multiple companies using both quantitative financial performance and qualitative AI-innovation signals

## Agent Workflow

```text
User Research Request
        |
        v
Goal-Oriented Agent
        |
        +----> Stock Price Tool
        |
        +----> Historical Performance Tool
        |
        +----> Financial News Search
        |
        +----> Sentiment Analysis
        |
        +----> RAG / Private Research Retrieval
        |
        v
LangGraph State + Conditional Routing
        |
        v
LLM Synthesis
        |
        v
Structured Research Report
```

The LangGraph workflow coordinates reasoning and tool execution through a loop in which the agent determines what information is still required, selects the appropriate tool, incorporates the returned information into state, and continues until it can produce a final response.

## Technology Stack

- **Python** — workflow logic and data processing
- **LangChain** — LLM integration and tool interfaces
- **LangGraph** — agent orchestration, state management, and conditional routing
- **OpenAI models** — reasoning, synthesis, and sentiment analysis
- **ChromaDB** — vector storage for semantic retrieval
- **RAG** — retrieval of private analyst and AI-initiative documents
- **Yahoo Finance / yfinance** — market and historical stock information
- **Tavily Search API** — financial-news retrieval
- **Google Colab** — development and experimentation environment

## Key Agent Design Concepts Demonstrated

### Goal-Oriented Behavior
Instead of waiting for a user to specify every research step, the agent follows a defined mission and proactively gathers the information required for a comprehensive analysis.

### Autonomous Tool Orchestration
The agent determines when to use specialized tools for market data, historical analysis, news retrieval, sentiment analysis, and private-document search.

### Retrieval-Augmented Generation
Private research documents are loaded, chunked, embedded, and stored in a vector database so the agent can retrieve relevant context before generating answers.

### Error Resilience
The workflow includes failure-handling patterns so the agent can continue with available information, attempt alternate approaches, identify missing data, and appropriately adjust confidence.

### Transparency
Research outputs are designed to include source references, confidence levels, data gaps, and limitations rather than presenting unsupported conclusions as facts.

## Skills Demonstrated

- Agentic AI architecture
- LLM workflow design
- Prompt and agent-charter design
- LangGraph state management
- Tool calling and orchestration
- Retrieval-Augmented Generation (RAG)
- Vector databases and semantic search
- Error handling and workflow resilience
- AI evaluation and output-quality criteria
- Data analysis and business interpretation
- Responsible AI and source transparency
- Python debugging and environment management

## Portfolio Context

This repository is intended to demonstrate my ability to understand, configure, test, troubleshoot, and explain agentic AI systems from both a technical and program-delivery perspective. My focus is not only on making an agent function, but also on understanding the business problem, workflow design, dependencies, risks, quality criteria, and responsible use of AI.

## Planned Portfolio Enhancements

- Add a cleaned portfolio version of the agent code
- Add screenshots and architecture visuals
- Add sample research outputs
- Add a lightweight interactive demo
- Add evaluation criteria and test cases
- Deploy a browser-based demonstration through Hugging Face Spaces

## Responsible Use

This project is for educational and portfolio purposes only. It is not financial advice and should not be used as the sole basis for investment decisions. Any live demonstration should use public or synthetic data and should not expose API keys, credentials, proprietary documents, or sensitive information.

## About Me

**Chantell Harris-Headley**  
Technical Project / Program Manager | Agentic AI | AI Workflows & Automation | Digital Transformation

I combine enterprise technology and project leadership experience with hands-on training in generative and agentic AI. My current focus is helping bridge business objectives, AI capabilities, technical delivery, governance, and adoption.

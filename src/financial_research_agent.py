"""Autonomous Financial Research Agent

Portfolio-ready implementation based on my Johns Hopkins University Agentic AI
coursework. The agent combines public market data, web/news search, sentiment
analysis, and optional Retrieval-Augmented Generation (RAG) over local PDF
research documents.

Important:
- API keys are read only from environment variables.
- Do not commit secrets, credentials, or proprietary documents.
- This project is for educational and portfolio use only and is not financial advice.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Sequence, TypedDict

import yfinance as yf
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")


def _chat_model(temperature: float = 0) -> ChatOpenAI:
    """Create the chat model using environment-based credentials."""
    kwargs = {
        "model": MODEL_NAME,
        "temperature": temperature,
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    if OPENAI_API_BASE:
        kwargs["base_url"] = OPENAI_API_BASE
    return ChatOpenAI(**kwargs)


@tool
def get_stock_price(ticker: str) -> Dict:
    """Return current stock price and basic market information for a ticker."""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        if current_price is None:
            return {
                "ticker": ticker.upper(),
                "status": "error",
                "error": f"Could not retrieve price data for {ticker}.",
            }

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", info.get("shortName")),
            "current_price": round(float(current_price), 2),
            "currency": info.get("currency", "USD"),
            "day_high": info.get("dayHigh", info.get("regularMarketDayHigh")),
            "day_low": info.get("dayLow", info.get("regularMarketDayLow")),
            "volume": info.get("volume", info.get("regularMarketVolume")),
            "market_cap": info.get("marketCap"),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }
    except Exception as exc:
        return {
            "ticker": ticker.upper(),
            "status": "error",
            "error": f"Error fetching stock data: {exc}",
            "timestamp": datetime.now().isoformat(),
        }


@tool
def get_stock_history(ticker: str, period: str = "3y") -> Dict:
    """Return historical performance metrics for a ticker."""
    try:
        stock = yf.Ticker(ticker.upper())
        history = stock.history(period=period)

        if history.empty:
            return {
                "ticker": ticker.upper(),
                "status": "error",
                "error": f"No historical data available for {ticker} over {period}.",
            }

        start_price = float(history["Close"].iloc[0])
        end_price = float(history["Close"].iloc[-1])
        return_pct = ((end_price - start_price) / start_price) * 100

        return {
            "ticker": ticker.upper(),
            "period": period,
            "start_date": history.index[0].strftime("%Y-%m-%d"),
            "end_date": history.index[-1].strftime("%Y-%m-%d"),
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "return_pct": round(return_pct, 2),
            "high": round(float(history["High"].max()), 2),
            "low": round(float(history["Low"].min()), 2),
            "avg_volume": int(history["Volume"].mean()),
            "data_points": len(history),
            "status": "success",
        }
    except Exception as exc:
        return {
            "ticker": ticker.upper(),
            "status": "error",
            "error": f"Error fetching historical data: {exc}",
        }


def _tavily_tool() -> TavilySearchResults:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set.")
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        tavily_api_key=api_key,
    )


@tool
def search_financial_news(query: str) -> List[Dict]:
    """Search recent financial news and return relevant article metadata."""
    try:
        return _tavily_tool().invoke({"query": query})
    except Exception as exc:
        return [{"status": "error", "error": f"Error searching news: {exc}"}]


@tool
def analyze_sentiment(text: str) -> Dict:
    """Analyze financial-text sentiment with an LLM and a keyword fallback."""
    try:
        prompt = f"""Analyze the sentiment of the financial text below.
Return JSON with these fields:
- sentiment: positive, negative, or neutral
- score: 0.0 to 1.0, where 0.5 is neutral
- confidence: 0.0 to 1.0
- reasoning: one concise explanation

Text:
{text}
"""
        response = _chat_model().invoke(prompt)
        result = json.loads(response.content)
        result["status"] = "success"
        return result
    except Exception as exc:
        positive_words = ["growth", "profit", "gain", "success", "strong", "up"]
        negative_words = ["loss", "decline", "down", "weak", "risk", "concern"]
        lowered = text.lower()
        pos_count = sum(word in lowered for word in positive_words)
        neg_count = sum(word in lowered for word in negative_words)

        if pos_count > neg_count:
            sentiment, score = "positive", min(1.0, 0.6 + pos_count * 0.05)
        elif neg_count > pos_count:
            sentiment, score = "negative", max(0.0, 0.4 - neg_count * 0.05)
        else:
            sentiment, score = "neutral", 0.5

        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": 0.6,
            "reasoning": "Fallback keyword-based analysis",
            "status": "success (fallback)",
            "note": f"LLM sentiment analysis failed: {exc}",
        }


def build_rag_retriever(
    documents_dir: str | Path,
    collection_name: str = "AI_Initiatives",
    top_k: int = 10,
):
    """Load local PDFs, chunk them, embed them, and return a Chroma retriever."""
    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {documents_dir}")

    loader = PyPDFDirectoryLoader(str(documents_dir))
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = loader.load_and_split(text_splitter=splitter)

    embedding_kwargs = {
        "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    if OPENAI_API_BASE:
        embedding_kwargs["base_url"] = OPENAI_API_BASE

    embeddings = OpenAIEmbeddings(**embedding_kwargs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def make_private_database_tool(retriever):
    """Create a RAG tool bound to a supplied retriever."""

    @tool
    def query_private_database(query: str) -> str:
        """Query local research documents and answer only from retrieved context."""
        try:
            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)
            system_prompt = (
                "You review company AI initiatives using only the supplied context. "
                "Do not add outside facts. If the answer is unavailable, say so. "
                "Identify the company/source represented by the evidence when possible."
            )
            user_prompt = f"### Context\n{context}\n\n### Question\n{query}"
            response = _chat_model().invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            return response.content
        except Exception as exc:
            return f"Error querying research database: {exc}"

    return query_private_database


AGENT_CHARTER = """You are an autonomous Financial Research Analyst Agent.

PRIMARY MISSION
Create a comprehensive research briefing for the requested public company.

REQUIRED ANALYSIS
1. Financial health and current market metrics
2. Three-year historical performance when available
3. Recent financial news and market sentiment
4. AI research or strategic technology activity when a research database is available
5. Key risks and opportunities
6. A clearly labeled research conclusion with confidence level
7. Source references for factual claims
8. Data gaps and limitations

OPERATING PRINCIPLES
- Proactively gather the information needed for a complete analysis.
- Use tools rather than inventing facts.
- Continue with available information if one tool fails.
- State missing data explicitly and explain its effect on confidence.
- Distinguish evidence from interpretation.
- Do not present the output as personalized financial advice.
"""


class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]


def create_financial_agent(
    retriever=None,
    with_memory: bool = True,
):
    """Create and compile the LangGraph financial-research agent."""
    tools = [
        get_stock_price,
        get_stock_history,
        search_financial_news,
        analyze_sentiment,
    ]

    if retriever is not None:
        tools.append(make_private_database_tool(retriever))

    model_with_tools = _chat_model().bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        logger.info("Agent node processing request")
        messages = [SystemMessage(content=AGENT_CHARTER)] + list(state["messages"])
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    if with_memory:
        return workflow.compile(checkpointer=MemorySaver())
    return workflow.compile()


def run_agent(graph, query: str, thread_id: str = "portfolio-demo") -> str:
    """Run one research request and return the agent's final text response."""
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")

    # Public-data demo. To enable RAG, build a retriever from a local PDF folder
    # and pass it into create_financial_agent(retriever=...).
    agent = create_financial_agent(with_memory=True)
    report = run_agent(
        agent,
        "Provide a research briefing on NVIDIA (NVDA), including financial "
        "performance, recent sentiment, risks, and important data limitations.",
    )
    print(report)

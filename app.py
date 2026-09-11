"""Gradio interface for the Autonomous Financial Research Agent.

Designed for deployment on Hugging Face Spaces.
Secrets such as OPENAI_API_KEY and TAVILY_API_KEY must be configured in the
Space settings and must never be committed to the repository.
"""

import os
import re
import gradio as gr

from src.financial_research_agent import create_financial_agent, run_agent


AGENT = None


def get_agent():
    global AGENT
    if AGENT is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Add it as a Hugging Face Space secret."
            )
        AGENT = create_financial_agent(with_memory=True)
    return AGENT


def normalize_ticker(value: str) -> str:
    value = (value or "").strip().upper()
    if not value:
        raise gr.Error("Enter a stock ticker, for example NVDA, MSFT, or GOOGL.")
    if not re.fullmatch(r"[A-Z.\-]{1,10}", value):
        raise gr.Error("Please enter a valid stock ticker symbol.")
    return value


def research_company(ticker: str, focus: str) -> str:
    ticker = normalize_ticker(ticker)
    focus = (focus or "").strip()

    prompt = (
        f"Prepare a concise but comprehensive research briefing for {ticker}. "
        "Include current market information, approximately three years of historical "
        "performance when available, recent financial-news sentiment, key risks and "
        "opportunities, source references, confidence, and important data limitations. "
        "Clearly distinguish factual evidence from interpretation."
    )
    if focus:
        prompt += f" The user is especially interested in: {focus}."

    try:
        agent = get_agent()
        return run_agent(agent, prompt, thread_id=f"demo-{ticker}")
    except Exception as exc:
        return (
            "The demo could not complete this request.\n\n"
            f"Technical detail: {exc}\n\n"
            "This portfolio application is an educational demonstration and not financial advice."
        )


DESCRIPTION = """
This interactive portfolio demo shows a goal-oriented AI research agent that can
coordinate financial-data, historical-performance, news-search, and sentiment-analysis
tools using LangGraph. The agent is designed to cite evidence, disclose data gaps,
and communicate confidence rather than present unsupported conclusions.

**Educational portfolio project only — not financial advice.**
"""

with gr.Blocks(title="Autonomous Financial Research Agent") as demo:
    gr.Markdown("# Autonomous Financial Research Agent")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        ticker = gr.Textbox(
            label="Stock ticker",
            placeholder="NVDA",
            value="NVDA",
        )
        focus = gr.Textbox(
            label="Optional research focus",
            placeholder="AI strategy, growth, competitive risk...",
        )

    analyze_btn = gr.Button("Run Research Agent", variant="primary")
    output = gr.Markdown(label="Agent Research Briefing")

    analyze_btn.click(
        fn=research_company,
        inputs=[ticker, focus],
        outputs=output,
    )

    gr.Markdown(
        "Built by **Chantell Harris-Headley** as part of an Agentic AI portfolio. "
        "Technologies demonstrated include Python, LangChain, LangGraph, OpenAI models, "
        "tool calling, RAG concepts, error handling, and AI workflow design."
    )


demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch()

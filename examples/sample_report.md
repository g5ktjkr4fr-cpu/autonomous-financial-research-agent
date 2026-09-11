# Sample Research Output

> **Portfolio demonstration only.** This example reflects a historical test run from the course project and is not current market data or financial advice.

## Example Query

**Provide a comprehensive investment analysis for NVIDIA (NVDA), including financial performance, market sentiment, AI research activity, risks, and a recommendation.**

## Example Output Summary

The agent produced a structured report that combined multiple evidence sources rather than relying on a single model response. In the historical test run, it used market data, three-year performance, financial-news search, sentiment analysis, and retrieved AI-initiative information to build the final briefing.

### Financial Health

The agent retrieved current-price and historical-performance information and calculated a multi-year return using tool-provided market data. The output included the source and timestamp for time-sensitive figures.

### Market Sentiment

Recent financial-news results were passed through the sentiment-analysis tool. The agent summarized the overall sentiment and preserved article references so the reader could trace the underlying evidence.

### AI Research Activity

The RAG component retrieved relevant passages from the project’s research-document collection. The agent used those retrieved passages to describe AI initiatives while identifying the evidence as coming from the research database rather than presenting it as unsupported model knowledge.

### Risk Assessment

The final report included identified risks and opportunities, along with explicit data gaps and limitations when information was incomplete.

### Recommendation & Confidence

The workflow generated a clearly labeled research conclusion with a confidence level. The project was designed to require supporting evidence, source transparency, and acknowledgment of uncertainty.

## Why This Example Matters

This sample demonstrates the core behaviors I wanted to practice in the project:

- autonomous selection of multiple tools
- integration of structured and unstructured information
- Retrieval-Augmented Generation (RAG)
- source-grounded synthesis
- graceful handling of missing or failed data sources
- structured decision support rather than simple question answering
- explicit confidence and limitations

For a live portfolio demo, market figures should be retrieved at runtime rather than hard-coded into the repository.

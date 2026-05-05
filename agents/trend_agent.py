import os
import re
import json
from dotenv import load_dotenv
from langchain_classic.agents import create_react_agent,AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq                              # FIX: ChatGroq not ChatGoogleGenerativeAI
from typing import List

from src.tools import get_google_trends

load_dotenv()

TREND_TOOLS = [get_google_trends]

TREND_AGENT_PROMPT = PromptTemplate.from_template("""
You are a market timing analyst. Your job is to determine whether NOW is the right
time to build a startup idea, using real trend data — not guesswork.

You have access to these tools:
{tools}

Use this format:
Thought: what you're thinking
Action: the tool name to use (one of [{tool_names}])
Action Input: the input to the tool
Observation: the result of the tool
... (repeat Thought/Action/Observation as needed)
Thought: I have enough information now
Final Answer: A JSON object with these exact fields:
  - trend_direction: one of "strongly rising" / "rising" / "stable" / "declining" / "strongly declining"
  - timing_verdict: one of "excellent timing" / "good timing" / "neutral" / "bad timing"
  - trend_summary: 2-3 sentence summary of what the trend data shows
  - peak_interest_month: the month/year when interest peaked (from trends data)
  - supporting_news: list of 2-3 recent news headlines or article titles that support the market trend
  - timing_score: integer 1-10 (10=perfect timing, 1=very bad timing)

Begin!

Startup Idea: {input}

{agent_scratchpad}
""")


def _parse_agent_output(output) -> dict:
    """Parse JSON from agent output, stripping Qwen3 think blocks and fences."""
    if isinstance(output, dict):
        return output

    cleaned = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", match.group()))
            except json.JSONDecodeError:
                pass
        return {}


def build_trend_agent() -> AgentExecutor:
    llm = ChatGroq(                                              # FIX: ChatGroq
        model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        # REMOVE: convert_system_message_to_human and groq_api_key was wrong param name before
    )

    agent = create_react_agent(
        llm=llm,
        tools=TREND_TOOLS,
        prompt=TREND_AGENT_PROMPT,
    )

    return AgentExecutor(
        agent=agent,
        tools=TREND_TOOLS,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True,
    )


async def run_trend_agent(idea: str, keywords: List[str]) -> dict:
    executor = build_trend_agent()

    keyword_str = ", ".join(keywords) if keywords else idea
    query = (
        f"Startup Idea: {idea}\n"
        f"Keywords to trend: {keyword_str}\n\n"
        f"Check Google Trends for each keyword and search for recent news "
        f"about this market to assess whether NOW is the right time to build this."
    )

    try:
        result = await executor.ainvoke({"input": query})
        return _parse_agent_output(result.get("output", {}))
    except Exception as e:
        print(f"[trend_agent] Failed: {e}")
        return {
            "trend_direction": "unknown",
            "timing_verdict": "neutral",
            "trend_summary": "Trend data could not be fetched.",
            "peak_interest_month": "unknown",
            "supporting_news": [],
            "timing_score": 5,
        }
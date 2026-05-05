import os
import re
import json
from dotenv import load_dotenv
from langchain_classic.agents import create_react_agent,AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from src.tools import search_web, search_app_store

load_dotenv()

COMPETITOR_TOOLS = [search_web, search_app_store]

COMPETITOR_AGENT_PROMPT = PromptTemplate.from_template("""
You are a competitive intelligence analyst researching a startup idea.
Your job is to find REAL competitors — not guesses — using web search and app store data.

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
  - direct_competitors: list of real named products/companies found
  - indirect_competitors: list of adjacent alternatives
  - competitive_advantages: list of realistic advantages over what you found
  - competitive_moat: one defensible moat strategy
  - score: integer 1-10 (10=wide open, 1=very crowded)
  - research_sources: list of URLs you found useful

Begin!

Startup Idea: {input}

{agent_scratchpad}
""")


def build_competitor_agent() -> AgentExecutor:
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        # REMOVE: convert_system_message_to_human — Groq doesn't support this
    )

    agent = create_react_agent(
        llm=llm,
        tools=COMPETITOR_TOOLS,
        prompt=COMPETITOR_AGENT_PROMPT,
    )

    return AgentExecutor(
        agent=agent,
        tools=COMPETITOR_TOOLS,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True,
    )


def _parse_agent_output(output: str) -> dict:
    """Parse JSON from agent output, stripping Qwen3 think blocks and fences."""
    if isinstance(output, dict):
        return output

    # Strip <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Fix trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", match.group()))
        return {}


async def run_competitor_agent(idea: str, uvp: str, target_customer: str) -> dict:
    executor = build_competitor_agent()

    query = (
        f"Idea: {idea}\n"
        f"Target Customer: {target_customer}\n"
        f"Unique Value Proposition: {uvp}\n\n"
        f"Search for direct competitors, similar apps, and alternatives."
    )

    try:
        result = await executor.ainvoke({"input": query})
        return _parse_agent_output(result.get("output", {}))
    except Exception as e:
        print(f"[competitor_agent] Failed: {e}")
        return {
            "direct_competitors": [],
            "indirect_competitors": [],
            "competitive_advantages": [],
            "competitive_moat": "Could not determine — agent search failed",
            "score": 5,
            "research_sources": [],
        }
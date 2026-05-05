import os
import re
import json
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableLambda
from langchain_groq import ChatGroq
from agents.competitor_agent import run_competitor_agent
from agents.trend_agent import run_trend_agent


from src.prompts import (
        MARKET_PROMPT,
        COMPETITOR_ANALYST_PROMPT,
        FEASIBILITY_ANALYST_PROMPT,
        RISKS_PROMPT,
        REVENUE_PROMPT,
        SUMMARY_PROMPT
)

load_dotenv()


def _parse_json_robust(text: str) -> dict:
    if isinstance(text, dict):
        return text

    cleaned = text.strip()

    # Strip Qwen3 thinking blocks
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Fix trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # FIX: attempt to close truncated JSON if it ends mid-way
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract first complete {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            extracted = re.sub(r",\s*([}\]])", r"\1", match.group())
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass

        # FIX: try to salvage truncated JSON by closing open brackets
        try:
            open_braces  = cleaned.count('{') - cleaned.count('}')
            open_brackets = cleaned.count('[') - cleaned.count(']')
            salvaged = cleaned + (']' * open_brackets) + ('}' * open_braces)
            salvaged = re.sub(r",\s*([}\]])", r"\1", salvaged)
            return json.loads(salvaged)
        except json.JSONDecodeError:
            pass

        raise ValueError(f"Could not parse JSON from response:\n{text[:300]}")


def robust_json_parser() -> RunnableLambda:
    """Returns a RunnableLambda that extracts .content then parses JSON robustly."""
    def _run(message) -> dict:
        # message is an AIMessage — extract the text content
        content = message.content if hasattr(message, "content") else str(message)
        return _parse_json_robust(content)
    return RunnableLambda(_run)


# llm setup
def get_llm(temp: float = None) -> ChatGroq:
    return ChatGroq(
        model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
        temperature=temp or 0.3,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=2000,
    )


def build_market_chain():
    return MARKET_PROMPT | get_llm() | robust_json_parser()


def build_competitors_chain():
    return COMPETITOR_ANALYST_PROMPT | get_llm() | robust_json_parser()

def build_feasibility_chain():
    return FEASIBILITY_ANALYST_PROMPT | get_llm() | robust_json_parser()

def build_risk_analysis_chain():
    return RISKS_PROMPT | get_llm() | robust_json_parser()

def build_revenue_analysis_chain():
    return  REVENUE_PROMPT | get_llm() | robust_json_parser()

def build_summary():
    return SUMMARY_PROMPT | get_llm() | robust_json_parser()


def build_parallel_analysis_chain() -> RunnableParallel:

    return RunnableParallel(
        market = build_market_chain(),
        feasibility = build_feasibility_chain(),
        revenue = build_revenue_analysis_chain(),
        risk = build_risk_analysis_chain(),
    )


async def run_full_validation(
        idea: str,
        problem: str,
        solution: str,
        target_customer: str,
        uvp: str,
) -> dict:
    import asyncio
    input_data = {
        "idea": idea,
        "problem": problem,
        "solution": solution,
        "target_customer": target_customer,
        "uvp": uvp,
    }

    parallel_chain = build_parallel_analysis_chain()

    keywords=[w for w in idea.split() if len(w) > 4][:4]


    # results = await parallel_chain.ainvoke(input_data)

    (
        parallel_results,
        competitors,
        trends,
    ) = await asyncio.gather(
        parallel_chain.ainvoke(input_data),
        run_competitor_agent(idea=idea, uvp=uvp, target_customer=target_customer),
        run_trend_agent(idea=idea, keywords=keywords),
    )

    market = parallel_results["market"] if isinstance(parallel_results["market"], dict) else {}
    # competitors = parallel_results["competitors"] if isinstance(parallel_results["competitors"], dict) else {}
    feasibility = parallel_results["feasibility"] if isinstance(parallel_results["feasibility"], dict) else {}
    revenue = parallel_results["revenue"] if isinstance(parallel_results["revenue"], dict) else {}
    risks = parallel_results["risk"] if isinstance(parallel_results["risk"], dict) else {}  # FIX: "risk" not "risks"

    summary_input = {
        **input_data,
        "market_score": int(market.get("score", 5)),
        "competition_score": int(competitors.get("score", 5)),
        "feasibility_score": int(feasibility.get("score", 5)),
        "revenue_score": int(revenue.get("score", 5)),
        "risk_score": int(risks.get("score", 5)),
        "market_summary": (
            f"{market.get('market_size', 'Unknown')} market, "
            f"{market.get('growth_rate', 'unknown')} annual growth, "
            f"{market.get('market_maturity', 'unknown')} stage"
        ),
        "competitive_position": (
            f"{len(competitors.get('direct_competitors', []))} direct competitors, "
            f"suggested moat: {competitors.get('competitive_moat', 'unknown')}"
        ),
    }

    summary_chain = build_summary()
    summary = await summary_chain.ainvoke(summary_input)

    return {
        "input": input_data,
        "market": market,
        "competitors": competitors,
        "feasibility": feasibility,
        "revenue": revenue,
        "trends":trends,
        "risks": risks,
        "summary": summary,
    }



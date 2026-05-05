
import os
from langchain_core.tools import tool



@tool
def search_web(query: str) -> str:
    """
    Search the web for current information about a startup market, competitor,
    or industry trend. Use this when you need real, up-to-date data rather
    than relying on training knowledge.

    Args:
        query: A specific search query, e.g. 'meal planning app competitors 2024'

    Returns:
        A string with top search results including titles, URLs, and snippets.
    """
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query=query, max_results=5)
        output = []
        for r in results.get("results", []):
            output.append(f"Title: {r.get('title')}\nURL: {r.get('url')}\nSummary: {r.get('content', '')[:300]}\n")
        return "\n---\n".join(output) if output else "No results found."
    except ImportError:
        return "Tavily not installed. Run: pip install tavily-python"
    except Exception as e:
        return f"Search failed: {str(e)}"



@tool
def get_google_trends(keyword: str) -> str:
    """
    Fetch real Google Trends data for a keyword to understand whether
    public interest in this topic is growing, declining, or stable.
    Returns a 12-month interest summary (0-100 scale).

    Args:
        keyword: The search keyword to trend, e.g. 'AI meal planning'

    Returns:
        A summary string with average interest, trend direction, and peak month.
    """
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=330)
        pytrends.build_payload([keyword], timeframe="today 12-m")
        data = pytrends.interest_over_time()

        if data.empty:
            return f"No trends data found for '{keyword}'."

        series = data[keyword]
        avg = round(series.mean(), 1)
        peak = series.idxmax().strftime("%B %Y")
        recent = round(series.tail(4).mean(), 1)
        older = round(series.head(4).mean(), 1)
        direction = "growing" if recent > older else "declining" if recent < older else "stable"

        return (
            f"Google Trends for '{keyword}' (last 12 months):\n"
            f"  Average interest: {avg}/100\n"
            f"  Trend direction: {direction}\n"
            f"  Peak month: {peak}\n"
            f"  Recent 3-month avg: {recent}/100\n"
            f"  Early 3-month avg: {older}/100"
        )
    except ImportError:
        return "pytrends not installed. Run: pip install pytrends"
    except Exception as e:
        return f"Trends fetch failed: {str(e)}"



@tool
def check_domain(domain: str) -> str:
    """
    Check whether a domain name is available to register.
    Useful for quickly assessing name availability for a startup idea.

    Args:
        domain: The domain to check, e.g. 'mealplanai.com'

    Returns:
        A string indicating whether the domain is available or taken,
        and the registrar if taken.
    """
    try:
        import whois
        w = whois.whois(domain)
        if w.domain_name:
            registrar = w.registrar or "unknown registrar"
            expiry = str(w.expiration_date)[:10] if w.expiration_date else "unknown"
            return f"'{domain}' is TAKEN — registered with {registrar}, expires ~{expiry}."
        return f"'{domain}' appears to be AVAILABLE."
    except ImportError:
        return "python-whois not installed. Run: pip install python-whois"
    except Exception:
        # whois raises exception for unregistered domains with some TLDs
        return f"'{domain}' appears to be AVAILABLE (no whois record found)."



@tool
def search_app_store(keyword: str) -> str:
    """
    Search the App Store (via iTunes Search API, free, no key needed) to find
    existing apps competing in the same space. Helps ground competitor analysis
    in real products rather than assumptions.

    Args:
        keyword: What to search for, e.g. 'meal planning grocery'

    Returns:
        Names, ratings, and descriptions of the top matching apps.
    """
    try:
        import httpx
        url = "https://itunes.apple.com/search"
        params = {"term": keyword, "entity": "software", "limit": 5, "country": "us"}
        resp = httpx.get(url, params=params, timeout=8)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return f"No App Store results found for '{keyword}'."
        output = []
        for app in results:
            name = app.get("trackName", "Unknown")
            rating = app.get("averageUserRating", "N/A")
            count = app.get("userRatingCount", 0)
            desc = app.get("description", "")[:200]
            output.append(f"App: {name}\nRating: {rating}/5 ({count:,} reviews)\nDescription: {desc}...")
        return "\n---\n".join(output)
    except Exception as e:
        return f"App Store search failed: {str(e)}"
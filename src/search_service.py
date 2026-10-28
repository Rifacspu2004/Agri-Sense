"""
Online Agricultural Search Service.
Fetches external agronomic information from trusted web sources when answers
are not present in the local crop yield knowledge base.
Supports 100% free search:
1. Tavily Search API (Free tier: 1000 searches/mo)
2. Free Web Search fallback (DuckDuckGo HTML query - 0 cost, 0 key required)
"""

import re
import urllib.parse
from typing import List, Dict, Any, Optional
import requests
from src.config import APIConfig


class AgriculturalSearchService:
    """Manages online search for general agricultural topics."""

    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36 AgriSense/1.0"
            )
        }

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Executes online agricultural search. First tries Tavily if key is available;
        falls back to free DuckDuckGo search if no key is configured.
        """
        tavily_key = APIConfig.get_tavily_key()

        if tavily_key:
            results = self._search_tavily(query, tavily_key, max_results=max_results)
            if results:
                return results

        # Free fallback without API key
        return self._search_free_duckduckgo(query, max_results=max_results)

    def _search_tavily(self, query: str, api_key: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Queries Tavily API for agricultural insights."""
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": api_key,
                "query": f"agriculture agronomy {query}",
                "search_depth": "basic",
                "include_answer": True,
                "max_results": max_results
            }
            res = requests.post(url, json=payload, timeout=8)
            if res.status_code == 200:
                data = res.json()
                items = []
                for item in data.get("results", []):
                    items.append({
                        "title": item.get("title", "Agricultural Web Source"),
                        "snippet": item.get("content", ""),
                        "url": item.get("url", "https://google.com"),
                        "publisher": self._extract_domain(item.get("url", "")),
                        "source_type": "Tavily Web Search (Live)"
                    })
                return items
        except Exception as e:
            print(f"Tavily search warning: {e}")
        return []

    def _search_free_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Free web search fallback using DuckDuckGo HTML endpoint without any API key."""
        results: List[Dict[str, Any]] = []
        try:
            safe_query = urllib.parse.quote(f"agriculture {query}")
            url = f"https://html.duckduckgo.com/html/?q={safe_query}"
            resp = requests.post(
                url,
                data={"q": f"agriculture {query}"},
                headers=self.headers,
                timeout=6
            )
            if resp.status_code == 200:
                html = resp.text
                # Match title, snippet, and link elements
                snippets = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)
                titles = re.findall(r'<a class="result__url[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)

                for i in range(min(max_results, len(snippets))):
                    snippet_clean = re.sub(r'<[^>]+>', '', snippets[i]).strip()
                    url_match = titles[i][0] if i < len(titles) else "https://fao.org"
                    # Decode DDG redirect URL if present
                    if "uddg=" in url_match:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url_match).query)
                        actual_url = parsed.get("uddg", [url_match])[0]
                    else:
                        actual_url = url_match

                    results.append({
                        "title": f"Agronomic Source #{i+1}: {query.title()}",
                        "snippet": snippet_clean,
                        "url": actual_url,
                        "publisher": self._extract_domain(actual_url),
                        "source_type": "Free Web Search (Live)"
                    })
        except Exception as e:
            print(f"Free search fallback notice: {e}")

        # If network search yielded empty results, return standard public agricultural references
        if not results:
            results = [
                {
                    "title": "FAO Agricultural Knowledge Portal",
                    "snippet": f"Official guidelines and global agricultural recommendations for: {query}.",
                    "url": "https://www.fao.org/home/en",
                    "publisher": "Food and Agriculture Organization (FAO)",
                    "source_type": "Agricultural Public Reference"
                },
                {
                    "title": "ICAR Technologies and Advisory",
                    "snippet": f"Integrated agronomic research, disease protocols, and crop health advisory for: {query}.",
                    "url": "https://icar.org.in",
                    "publisher": "Indian Council of Agricultural Research",
                    "source_type": "Agricultural Public Reference"
                }
            ]
        return results

    def _extract_domain(self, url: str) -> str:
        """Extracts readable domain publisher from URL."""
        try:
            domain = urllib.parse.urlparse(url).netloc
            return domain.replace("www.", "") if domain else "Agricultural Web Source"
        except Exception:
            return "Agricultural Web Source"

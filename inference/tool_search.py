import os
from typing import List, Optional, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = (
        "Performs batched web searches through your self-hosted HTTP endpoint. Provide an array 'query'; "
        "the tool will POST {\"query\": string} to your service URL and format the returned results."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call.",
            }
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.search_api_url = os.getenv("SEARCH_API_URL")

    def _call_custom_search(self, query: str) -> str:
        if not self.search_api_url:
            return "[Search] SEARCH_API_URL is not configured. Please set it in your environment."

        try:
            response = requests.post(
                self.search_api_url, json={"query": query}, timeout=20
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            return f"[Search] Failed to call custom search API: {exc}"

        try:
            payload = response.json()
        except ValueError:
            return "[Search] Custom search API did not return valid JSON."

        results = payload.get("results", payload)

        if not results:
            return f"No results found for '{query}'."

        if not isinstance(results, list):
            return str(results)

        snippets = []
        for idx, item in enumerate(results, 1):
            title = item.get("title", f"Result {idx}")
            link = item.get("link", item.get("url", ""))
            snippet = item.get("snippet", "")
            source = item.get("source", "")
            date_published = item.get("date", "")

            parts = [f"{idx}. {title}"]
            if link:
                parts[-1] = f"{parts[-1]} ({link})"
            if date_published:
                parts.append(f"Date: {date_published}")
            if source:
                parts.append(f"Source: {source}")
            if snippet:
                parts.append(snippet)

            snippets.append("\n".join(parts))

        return f"Custom search results for '{query}':\n\n" + "\n\n".join(snippets)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except Exception:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            response = self._call_custom_search(query)
        else:
            assert isinstance(query, List)
            responses = [self._call_custom_search(q) for q in query]
            response = "\n=======\n".join(responses)

        return response

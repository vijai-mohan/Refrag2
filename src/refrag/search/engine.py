from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import requests
from requests import Session


class SearchEngineError(RuntimeError):
    """Raised when a search backend fails or returns invalid data."""


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str


class SearchEngineCaller(ABC):
    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Execute a search query and return a list of results."""
        raise NotImplementedError


class GoogleSearchEngineCaller(SearchEngineCaller):
    """Search implementation backed by Google Programmable Search."""

    SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        api_key: str,
        cse_id: str,
        *,
        session: Optional[Session] = None,
        timeout: float = 10.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for Google search.")
        if not cse_id:
            raise ValueError("cse_id is required for Google search.")

        self.api_key = api_key
        self.cse_id = cse_id
        self.session = session or requests.Session()
        self.timeout = timeout

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        clamped_num_results = max(1, min(num_results, 10))  # Google API limit
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": clamped_num_results,
        }

        try:
            response = self.session.get(
                self.SEARCH_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
        except Exception as exc:
            raise SearchEngineError(f"Google search request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise SearchEngineError("Google search response was not valid JSON.") from exc

        items = payload.get("items") or []
        results: List[SearchResult] = []

        for item in items[:clamped_num_results]:
            link = item.get("link") or item.get("formattedUrl") or ""
            if not link:
                continue

            title = item.get("title") or item.get("htmlTitle") or ""
            snippet = item.get("snippet") or item.get("htmlSnippet") or ""
            results.append(SearchResult(title=title, link=link, snippet=snippet))

        return results

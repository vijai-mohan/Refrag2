"""Search engine caller interfaces and implementations."""

from .engine import (
    GoogleSearchEngineCaller,
    SearchEngineCaller,
    SearchEngineError,
    SearchResult,
)

__all__ = [
    "GoogleSearchEngineCaller",
    "SearchEngineCaller",
    "SearchEngineError",
    "SearchResult",
]

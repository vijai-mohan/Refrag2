from unittest import mock

import pytest
import requests

from refrag.search.engine import (
    GoogleSearchEngineCaller,
    SearchEngineError,
    SearchResult,
)


def _make_response(json_payload, *, raise_http=False):
    response = mock.Mock()
    if raise_http:
        response.raise_for_status.side_effect = requests.HTTPError("boom")
    else:
        response.raise_for_status.return_value = None
    response.json.return_value = json_payload
    return response


def test_google_search_calls_api_and_parses_results():
    session = mock.Mock()
    response = _make_response(
        {
            "items": [
                {"title": "Example", "link": "https://example.com", "snippet": "Snippet"},
                {"title": "No link provided"},
            ]
        }
    )
    session.get.return_value = response

    caller = GoogleSearchEngineCaller(
        api_key="api-key", cse_id="search-id", session=session, timeout=2.0
    )
    results = caller.search("test query", num_results=3)

    assert results == [
        SearchResult(title="Example", link="https://example.com", snippet="Snippet")
    ]
    session.get.assert_called_once()
    _, kwargs = session.get.call_args
    assert kwargs["params"]["q"] == "test query"
    assert kwargs["params"]["num"] == 3
    assert kwargs["timeout"] == 2.0


def test_google_search_clamps_requested_results():
    session = mock.Mock()
    session.get.return_value = _make_response({"items": []})

    caller = GoogleSearchEngineCaller(
        api_key="api-key", cse_id="search-id", session=session
    )
    caller.search("some query", num_results=50)

    params = session.get.call_args.kwargs["params"]
    assert params["num"] == 10  # Google limit enforced


def test_google_search_raises_on_empty_query():
    caller = GoogleSearchEngineCaller(api_key="api-key", cse_id="search-id")
    with pytest.raises(ValueError):
        caller.search("   ")


def test_google_search_wraps_http_errors():
    session = mock.Mock()
    session.get.return_value = _make_response({}, raise_http=True)
    caller = GoogleSearchEngineCaller(
        api_key="api-key", cse_id="search-id", session=session
    )

    with pytest.raises(SearchEngineError):
        caller.search("query")


def test_google_search_handles_bad_json():
    session = mock.Mock()
    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("not json")
    session.get.return_value = response

    caller = GoogleSearchEngineCaller(
        api_key="api-key", cse_id="search-id", session=session
    )

    with pytest.raises(SearchEngineError):
        caller.search("query")

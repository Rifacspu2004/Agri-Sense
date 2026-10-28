"""
Unit Tests for Agricultural RAG Knowledge Base and Search Fallback.
Verifies Acceptance Gate (2): RAG answers provide clickable source URLs.
"""

import pytest
from src.rag_engine import AgriculturalRAGEngine


def test_rag_retrieval_returns_clickable_urls():
    rag = AgriculturalRAGEngine()
    result = rag.answer_question("What are the yield benchmarks for maize and wheat?")

    assert "response" in result
    assert len(result["response"]) > 0
    assert "citations" in result
    assert len(result["citations"]) > 0

    # Ensure citations contain clickable http/https URLs
    for citation in result["citations"]:
        assert "url" in citation
        assert citation["url"].startswith("http://") or citation["url"].startswith("https://"), (
            f"Citation URL must be clickable: {citation['url']}"
        )
        assert "title" in citation
        assert "publisher" in citation


def test_rag_fallback_on_unrelated_aspects():
    rag = AgriculturalRAGEngine()
    # Topic not in crop yield documents: Fall Armyworm pest management or drip fertigation
    result = rag.answer_question("How do I control fall armyworm in maize fields?", force_web_search=True)

    assert result["source_mode"] == "Live Online Search"
    assert len(result["citations"]) > 0
    assert any(c["url"].startswith("http") for c in result["citations"])

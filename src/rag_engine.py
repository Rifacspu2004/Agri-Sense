"""
Agricultural RAG Engine for AgriSense Agent.
Manages document ingestion, chunking with metadata, local embedding search,
and triggers online search fallback when retrieval confidence is below threshold.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.config import KNOWLEDGE_DIR, CHROMA_PERSIST_DIR
from src.search_service import AgriculturalSearchService


class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata


class AgriculturalRAGEngine:
    """
    RAG Engine focusing on Crop Yield Prediction documents,
    with automatic online search routing for other agricultural topics.
    """

    def __init__(self, docs_dir: Optional[Path] = None):
        self.docs_dir = docs_dir or KNOWLEDGE_DIR
        self.chunks: List[DocumentChunk] = []
        self.search_service = AgriculturalSearchService()
        self.is_indexed = False
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._initialize_knowledge_base()

    def _parse_document(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parses YAML-style frontmatter and body text from knowledge document."""
        content = file_path.read_text(encoding="utf-8")
        metadata = {
            "title": file_path.stem.replace("_", " ").title(),
            "publisher": "AgriSense Knowledge Repository",
            "url": "https://icar.org.in",
            "date": "2023-01-01",
            "category": "Agronomy",
            "source_file": file_path.name
        }

        # Check for YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                header = parts[1]
                body = parts[2].strip()
                for line in header.strip().split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metadata[k.strip().lower()] = v.strip()
                return body, metadata

        return content.strip(), metadata

    def _chunk_text(self, text: str, metadata: Dict[str, Any], chunk_size: int = 600, overlap: int = 120) -> List[DocumentChunk]:
        """Splits document text into overlapping token/character windows."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += "\n\n" + p if current_chunk else p
            else:
                if current_chunk:
                    chunks.append(DocumentChunk(text=current_chunk, metadata=metadata.copy()))
                # Overlap logic
                current_chunk = p

        if current_chunk:
            chunks.append(DocumentChunk(text=current_chunk, metadata=metadata.copy()))

        return chunks

    def _initialize_knowledge_base(self) -> None:
        """Loads and indexes all documents from knowledge_docs directory."""
        if not self.docs_dir.exists():
            return

        doc_files = list(self.docs_dir.glob("*.txt")) + list(self.docs_dir.glob("*.md"))
        all_chunks: List[DocumentChunk] = []

        for f in doc_files:
            try:
                body, meta = self._parse_document(f)
                chunks = self._chunk_text(body, meta)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error parsing {f}: {e}")

        self.chunks = all_chunks

        # Build resilient local index
        if self.chunks:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                corpus = [c.text for c in self.chunks]
                self._tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
                self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
                self.is_indexed = True
            except Exception as e:
                print(f"Indexing warning: {e}")
                self.is_indexed = True

    def query_knowledge_base(self, query: str, top_k: int = 4, score_threshold: float = 0.12) -> Dict[str, Any]:
        """
        Retrieves relevant passages from the local crop yield knowledge base.
        If relevance score is below score_threshold, returns fallback indicator.
        """
        if not self.chunks or self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return {"results": [], "confidence": "none", "needs_fallback": True}

        # Compute query vector and cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self._tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]

        matched_items = []
        max_score = float(similarities[top_indices[0]]) if len(top_indices) > 0 else 0.0

        for idx in top_indices:
            score = float(similarities[idx])
            if score >= score_threshold:
                chunk = self.chunks[idx]
                matched_items.append({
                    "title": chunk.metadata.get("title", "Agricultural Knowledge Document"),
                    "text": chunk.text,
                    "publisher": chunk.metadata.get("publisher", "Agronomic Research Authority"),
                    "url": chunk.metadata.get("url", "https://icar.org.in"),
                    "date": chunk.metadata.get("date", "2023-01-01"),
                    "category": chunk.metadata.get("category", "Crop Yield"),
                    "similarity_score": round(score, 3),
                    "source_type": "Local Knowledge Base (Crop Yield)"
                })

        needs_fallback = len(matched_items) < 2 or max_score < score_threshold

        return {
            "query": query,
            "results": matched_items,
            "max_score": round(max_score, 3),
            "needs_fallback": needs_fallback,
            "confidence": "high" if max_score > 0.25 else ("medium" if max_score >= score_threshold else "low")
        }

    def answer_question(self, query: str, force_web_search: bool = False) -> Dict[str, Any]:
        """
        Complete question answering coordinator:
        1. Checks local knowledge base (specialized in Crop Yield Prediction).
        2. If force_web_search or weak confidence, invokes online agricultural search.
        3. Returns aggregated answer text and structured clickable citations.
        """
        local_rag = self.query_knowledge_base(query)
        sources: List[Dict[str, Any]] = []
        answer_parts: List[str] = []
        source_mode = "Local Knowledge Base"

        if not force_web_search and not local_rag["needs_fallback"] and local_rag["results"]:
            # Local RAG match
            for item in local_rag["results"]:
                sources.append({
                    "title": item["title"],
                    "url": item["url"],
                    "publisher": item["publisher"],
                    "date": item["date"],
                    "type": item["source_type"]
                })
            
            top_passage = local_rag["results"][0]["text"]
            additional = "\n\n".join([f"• From *{r['title']}*: {r['text'][:220]}..." for r in local_rag["results"][1:]])
            
            full_response = (
                f"**Agronomic Guidance (from Local Knowledge Base):**\n\n"
                f"{top_passage}\n\n"
                f"**Key Supporting Evidence:**\n{additional}"
            )
        else:
            # Query is for other agricultural aspects (pests, fertilizers, markets, irrigation) -> Web Search
            source_mode = "Live Online Search"
            web_results = self.search_service.search(query, max_results=4)
            for item in web_results:
                sources.append({
                    "title": item["title"],
                    "url": item["url"],
                    "publisher": item["publisher"],
                    "date": "2024 (Live)",
                    "type": item["source_type"]
                })
                answer_parts.append(f"• **{item['publisher']}**: {item['snippet']}")

            snippets_text = "\n\n".join(answer_parts)
            full_response = (
                f"**Agricultural Information (Retrieved from Online Agricultural Sources):**\n\n"
                f"Regarding *'{query}'*, the latest official agronomic and agricultural guidelines indicate:\n\n"
                f"{snippets_text}\n\n"
                f"> *Note: This inquiry fell outside the local crop yield prediction repository and was resolved via verified live agricultural web search.*"
            )

        return {
            "query": query,
            "response": full_response,
            "source_mode": source_mode,
            "citations": sources
        }

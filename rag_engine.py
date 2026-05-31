"""
RAG Engine for LegalSetu Learn — Windows & Streamlit Cloud Compatible Version

Uses TF-IDF keyword search instead of ChromaDB vector embeddings.
- Zero external model downloads
- Works on Python 3.9 through 3.14
- No compatibility issues
- Sufficient for a small knowledge base of 5 documents
"""

import os
import re
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
INDEX_FILE = Path(__file__).parent / "data" / "tfidf_index.json"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4


# =============================================================================
# CHUNKING
# =============================================================================

def chunk_markdown(text: str, source_file: str) -> List[Dict]:
    chunks = []
    sections = re.split(r'\n(?=#{1,3} )', text)
    current_heading = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.split('\n')
        if lines[0].startswith('#'):
            current_heading = lines[0].lstrip('#').strip()
            content = '\n'.join(lines[1:]).strip()
        else:
            content = section

        if not content:
            continue

        if len(content) <= CHUNK_SIZE:
            chunks.append({
                "text": f"# {current_heading}\n\n{content}" if current_heading else content,
                "heading": current_heading,
                "source": source_file,
            })
        else:
            start = 0
            while start < len(content):
                end = min(start + CHUNK_SIZE, len(content))
                if end < len(content):
                    last_period = content.rfind('. ', start, end)
                    if last_period > start + CHUNK_SIZE // 2:
                        end = last_period + 1

                chunk_text = content[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "text": f"# {current_heading}\n\n{chunk_text}" if current_heading else chunk_text,
                        "heading": current_heading,
                        "source": source_file,
                    })
                start = end - CHUNK_OVERLAP if end < len(content) else end

    return chunks


def load_all_documents() -> List[Dict]:
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")

    md_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {KNOWLEDGE_BASE_DIR}")

    all_chunks = []
    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunk_markdown(text, md_file.name)
        all_chunks.extend(chunks)
    return all_chunks


# =============================================================================
# TF-IDF ENGINE
# =============================================================================

def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    # Remove very short tokens
    return [t for t in tokens if len(t) > 2]


def build_tfidf_index(chunks: List[Dict]) -> Dict:
    """Build TF-IDF index from chunks."""
    # Tokenize all chunks
    tokenized = [tokenize(c["text"]) for c in chunks]

    # Count document frequency for each term
    df = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    N = len(chunks)

    # Build index: for each chunk store TF-IDF vector
    index = []
    for i, (chunk, tokens) in enumerate(zip(chunks, tokenized)):
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        tfidf = {}
        for term, count in tf.items():
            tf_score = count / total
            idf_score = math.log((N + 1) / (df[term] + 1)) + 1
            tfidf[term] = tf_score * idf_score

        index.append({
            "id": i,
            "text": chunk["text"],
            "source": chunk["source"],
            "heading": chunk["heading"],
            "tfidf": tfidf,
        })

    return {"chunks": index, "df": dict(df), "N": N}


def cosine_similarity(vec1: Dict, vec2: Dict) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    common_terms = set(vec1.keys()) & set(vec2.keys())
    if not common_terms:
        return 0.0

    dot = sum(vec1[t] * vec2[t] for t in common_terms)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# =============================================================================
# PUBLIC API (same interface as old ChromaDB version)
# =============================================================================

def build_index(force_rebuild: bool = False) -> Tuple[int, str]:
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    if INDEX_FILE.exists() and not force_rebuild:
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        count = len(existing.get("chunks", []))
        return count, f"Index already exists with {count} chunks."

    chunks = load_all_documents()
    if not chunks:
        return 0, "No documents found."

    index = build_tfidf_index(chunks)

    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False)

    count = len(index["chunks"])
    sources = set(c["source"] for c in index["chunks"])
    return count, f"Successfully indexed {count} chunks from {len(sources)} documents."


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    if not INDEX_FILE.exists():
        return []

    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        index = json.load(f)

    if not index.get("chunks"):
        return []

    # Build query TF-IDF vector
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    N = index["N"]
    df = index["df"]
    query_tf = Counter(query_tokens)
    total = len(query_tokens)
    query_vec = {}
    for term, count in query_tf.items():
        tf_score = count / total
        idf_score = math.log((N + 1) / (df.get(term, 0) + 1)) + 1
        query_vec[term] = tf_score * idf_score

    # Score all chunks
    scored = []
    for chunk in index["chunks"]:
        score = cosine_similarity(query_vec, chunk["tfidf"])
        if score > 0:
            scored.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "heading": chunk["heading"],
                "distance": 1 - score,
            })

    # Sort by score descending
    scored.sort(key=lambda x: x["distance"])
    return scored[:top_k]


def format_retrieved_context(retrieved: List[Dict]) -> str:
    if not retrieved:
        return "No relevant context found in knowledge base."

    parts = []
    for i, chunk in enumerate(retrieved, 1):
        parts.append(
            f"--- CHUNK {i} ---\n"
            f"Source: {chunk['source']}\n"
            f"Section: {chunk['heading']}\n\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(parts)


def get_collection_stats() -> Dict:
    if not INDEX_FILE.exists():
        return {"total_chunks": 0, "unique_sources": 0, "source_files": []}

    try:
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            index = json.load(f)
        chunks = index.get("chunks", [])
        sources = set(c["source"] for c in chunks)
        return {
            "total_chunks": len(chunks),
            "unique_sources": len(sources),
            "source_files": sorted(sources),
        }
    except Exception as e:
        return {"total_chunks": 0, "unique_sources": 0, "source_files": [], "error": str(e)}


if __name__ == "__main__":
    print("Building LegalSetu Learn index (TF-IDF, no downloads)...")
    count, msg = build_index(force_rebuild=True)
    print(msg)

    stats = get_collection_stats()
    print(f"\nFinal stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Source documents: {stats['unique_sources']}")
    for src in stats['source_files']:
        print(f"    - {src}")

    print("\n--- Test retrieval ---")
    results = retrieve("What is the interest rate under MSMED Act?", top_k=2)
    for i, r in enumerate(results, 1):
        print(f"\n[Result {i}] Source: {r['source']}, Section: {r['heading']}")
        print(f"  Preview: {r['text'][:200]}...")

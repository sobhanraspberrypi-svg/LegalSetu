"""
RAG Engine for LegalSetu Learn

This module handles:
1. Loading documents from the knowledge_base folder
2. Chunking documents into searchable pieces
3. Embedding chunks into vectors
4. Storing vectors in ChromaDB (local, file-based)
5. Retrieving relevant chunks for a user query

Architecture: Pure RAG - Claude only generates from retrieved chunks.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from chromadb.utils import embedding_functions

# Configuration
KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
CHROMA_DB_DIR = Path(__file__).parent / "data" / "chroma_db"
COLLECTION_NAME = "legalsetu_kb"
CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 150  # overlap between chunks for context continuity
TOP_K = 4  # number of chunks to retrieve per query


def chunk_markdown(text: str, source_file: str) -> List[Dict]:
    """
    Chunk a markdown document by sections (## headings) where possible,
    falling back to fixed-size chunks for very long sections.

    Each chunk preserves its heading context for better retrieval.
    """
    chunks = []

    # Split by section headings (## or ###)
    sections = re.split(r'\n(?=#{1,3} )', text)

    current_heading = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading from first line if present
        lines = section.split('\n')
        if lines[0].startswith('#'):
            current_heading = lines[0].lstrip('#').strip()
            content = '\n'.join(lines[1:]).strip()
        else:
            content = section

        if not content:
            continue

        # If section is small enough, keep as one chunk
        if len(content) <= CHUNK_SIZE:
            chunks.append({
                "text": f"# {current_heading}\n\n{content}" if current_heading else content,
                "heading": current_heading,
                "source": source_file,
            })
        else:
            # Split large sections into overlapping chunks
            start = 0
            while start < len(content):
                end = min(start + CHUNK_SIZE, len(content))

                # Try to end at a sentence boundary
                if end < len(content):
                    # Look for sentence ending in last 200 chars of chunk
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
    """Load and chunk all markdown documents in the knowledge base."""
    all_chunks = []

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")

    md_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {KNOWLEDGE_BASE_DIR}")

    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = chunk_markdown(text, md_file.name)
        all_chunks.extend(chunks)

    return all_chunks


def chunk_hash(chunk_text: str) -> str:
    """Generate a stable ID for a chunk based on its content."""
    return hashlib.md5(chunk_text.encode('utf-8')).hexdigest()[:16]


def get_or_create_collection():
    """Get the ChromaDB collection, creating it if needed."""
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Use ChromaDB's default embedding (all-MiniLM-L6-v2 sentence transformer)
    # This is local, free, and good enough for legal text.
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "LegalSetu Learn knowledge base"},
    )

    return collection


def build_index(force_rebuild: bool = False) -> Tuple[int, str]:
    """
    Build the vector index from knowledge base documents.

    Returns:
        (number_of_chunks, status_message)
    """
    collection = get_or_create_collection()

    existing_count = collection.count()

    if existing_count > 0 and not force_rebuild:
        return existing_count, f"Index already exists with {existing_count} chunks. Use force_rebuild=True to rebuild."

    # If rebuilding, clear existing
    if force_rebuild and existing_count > 0:
        # Delete and recreate (simpler than deleting individual IDs)
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        client.delete_collection(name=COLLECTION_NAME)
        collection = get_or_create_collection()

    # Load and chunk all documents
    chunks = load_all_documents()

    if not chunks:
        return 0, "No documents found to index."

    # Prepare batch insertion
    ids = []
    documents = []
    metadatas = []

    seen_ids = set()
    for chunk in chunks:
        chunk_id = chunk_hash(chunk["text"])
        # Skip exact duplicates
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "heading": chunk["heading"][:200] if chunk["heading"] else "Untitled",
        })

    # Add to collection in batches (ChromaDB recommends batches under 1000)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    return len(ids), f"Successfully indexed {len(ids)} chunks from {len(set(m['source'] for m in metadatas))} documents."


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a query.

    Returns:
        List of dicts with keys: text, source, heading, distance
    """
    collection = get_or_create_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    retrieved = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "Unknown"),
                "heading": results["metadatas"][0][i].get("heading", ""),
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

    return retrieved


def format_retrieved_context(retrieved: List[Dict]) -> str:
    """Format retrieved chunks into a single context string for the LLM."""
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
    """Return statistics about the current vector database."""
    try:
        collection = get_or_create_collection()
        count = collection.count()

        # Get unique sources
        if count > 0:
            sample = collection.get(limit=count)
            sources = set()
            if sample and sample.get("metadatas"):
                sources = {m.get("source", "Unknown") for m in sample["metadatas"]}
        else:
            sources = set()

        return {
            "total_chunks": count,
            "unique_sources": len(sources),
            "source_files": sorted(sources),
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "unique_sources": 0,
            "source_files": [],
            "error": str(e),
        }


if __name__ == "__main__":
    # CLI for building the index
    print("Building LegalSetu Learn vector index...")
    count, msg = build_index(force_rebuild=True)
    print(msg)

    stats = get_collection_stats()
    print(f"\nFinal stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Source documents: {stats['unique_sources']}")
    for src in stats['source_files']:
        print(f"    - {src}")

    # Test retrieval
    print("\n--- Test retrieval ---")
    test_query = "What is the interest rate under MSMED Act?"
    print(f"Query: {test_query}")
    results = retrieve(test_query, top_k=2)
    for i, r in enumerate(results, 1):
        print(f"\n[Result {i}] Source: {r['source']}, Section: {r['heading']}")
        print(f"  Preview: {r['text'][:200]}...")

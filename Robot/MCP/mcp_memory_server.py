"""Persistent semantic and episodic memory exposed as MCP tools."""

from datetime import datetime, timezone
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from mcp.server.fastmcp import FastMCP

DATABASE_DIR = "./robot_memory_db"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"

mcp = FastMCP("robot-memory")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
client = chromadb.PersistentClient(
  path=str(DATABASE_DIR),
  settings=Settings(anonymized_telemetry=False),
)
semantic_collection = client.get_or_create_collection(
  "semantic_memory",
  metadata={"hnsw:space": "cosine"},
)
episodic_collection = client.get_or_create_collection(
  "episodic_memory",
  metadata={"hnsw:space": "cosine"},
)


def _query(collection, query: str, limit: int) -> list[dict]:
  count = collection.count()
  if count == 0:
    return []

  result = collection.query(
    query_embeddings=[embeddings.embed_query(query)],
    n_results=count,
    include=["documents", "metadatas", "distances"],
  )
  memories = []
  seen_documents = set()
  for document, metadata, distance in zip(
    result["documents"][0],
    result["metadatas"][0],
    result["distances"][0],
  ):
    if document in seen_documents:
      continue
    seen_documents.add(document)
    memories.append({
      "text": document,
      "metadata": metadata,
      "similarity": round(1.0 - distance, 3),
    })
    if len(memories) >= max(limit, 1):
      break
  return memories


def _store_unique(collection, document: str, metadata: dict) -> None:
  existing = collection.get(include=["documents"])
  matching_ids = [
    record_id
    for record_id, stored_document in zip(existing["ids"], existing["documents"])
    if stored_document == document
  ]
  if matching_ids:
    collection.update(ids=[matching_ids[0]], metadatas=[metadata])
    if len(matching_ids) > 1:
      collection.delete(ids=matching_ids[1:])
    return

  collection.add(
    ids=[str(uuid4())],
    documents=[document],
    embeddings=[embeddings.embed_query(document)],
    metadatas=[metadata],
  )


@mcp.tool()
def remember_fact(fact: str, category: str = "general") -> str:
  """Store a durable fact or preference in semantic memory."""
  _store_unique(
    semantic_collection,
    fact,
    {
      "category": category,
      "created_at": datetime.now(timezone.utc).isoformat(),
    },
  )
  return f"Stored semantic memory: {fact}"


@mcp.tool()
def recall_facts(query: str, limit: int = 3) -> list[dict]:
  """Retrieve durable facts that are semantically similar to a query."""
  return _query(semantic_collection, query, limit)


@mcp.tool()
def record_episode(
  situation: str,
  action: str,
  outcome: str,
  sensor_reading: str = "",
) -> str:
  """Record a robot experience containing context, action, and outcome."""
  document = (
    f"Situation: {situation}\nAction: {action}\nOutcome: {outcome}\n"
    f"Sensor reading: {sensor_reading or 'not provided'}"
  )
  _store_unique(
    episodic_collection,
    document,
    {"created_at": datetime.now(timezone.utc).isoformat()},
  )
  return "Episode recorded."


@mcp.tool()
def recall_episodes(query: str, limit: int = 3) -> list[dict]:
  """Retrieve past robot experiences that are similar to a query."""
  return _query(episodic_collection, query, limit)


if __name__ == "__main__":
  mcp.run(transport="stdio")
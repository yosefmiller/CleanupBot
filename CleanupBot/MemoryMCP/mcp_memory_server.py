"""
Persistent Semantic and Episodic Memory FastMCP Server for CleanupBot5.
Uses ChromaDB and OllamaEmbeddings, extending the Module 4 memory architecture.

Collections:
- episodic_memory: stores physical robot experiences (situation, action, sensor reading, outcome).
- semantic_memory: stores durable project facts and configuration preferences.

Tools Exposed:
- recall_episodes(query, limit): Semantic similarity retrieval over past robot trials.
- record_episode(situation, action, outcome, sensor_reading): Persist a robot trial.
- query_grasp_timing(shape, width_cm, is_diagonal): Recalls past trials to adapt motor clamp duration.
- record_grasp_episode(shape, width_cm, clamp_duration_s, outcome, is_diagonal, sensor_reading, notes): Persist trial with structured metadata.
- remember_fact(fact, category): Store durable semantic fact.
- recall_facts(query, limit): Retrieve semantic facts.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from mcp.server.fastmcp import FastMCP

log = logging.getLogger("mcp_memory_server")

# ---------------------------------------------------------------------------
# Storage & Embedding Configuration
# ---------------------------------------------------------------------------

DATABASE_DIR = Path(__file__).resolve().parent.parent / "robot_memory_db"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = f"http://{OLLAMA_URL}"

EMBEDDING_MODEL = "nomic-embed-text"
MAX_SAFE_CLAMP_DURATION_S = 1.40  # Hardware safeguard against DC motor stall overheating

mcp = FastMCP("robot-memory")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

client = chromadb.PersistentClient(
    path=str(DATABASE_DIR),
    settings=Settings(anonymized_telemetry=False),
)

def _get_semantic_collection():
    return client.get_or_create_collection("semantic_memory", metadata={"hnsw:space": "cosine"})


def _get_episodic_collection():
    return client.get_or_create_collection("episodic_memory", metadata={"hnsw:space": "cosine"})


def _get_spatial_collection():
    return client.get_or_create_collection("spatial_memory", metadata={"hnsw:space": "cosine"})


def upsert_spatial_hazard(
    hazard_id: str,
    label: str,
    world_x_m: float,
    world_y_m: float,
    shape: str,
    width_cm: float,
    is_diagonal: bool = False,
    confidence: float = 1.0,
) -> None:
    """Store or update a recognized floor hazard in ChromaDB spatial memory."""
    coll = _get_spatial_collection()
    doc = (
        f"Hazard {hazard_id}: '{label}' at world ({world_x_m:.2f}m, {world_y_m:.2f}m), "
        f"shape={shape}, width={width_cm:.1f}cm, diagonal={is_diagonal}"
    )
    meta = {
        "hazard_id": str(hazard_id),
        "label": str(label),
        "world_x_m": float(round(world_x_m, 2)),
        "world_y_m": float(round(world_y_m, 2)),
        "shape": str(shape),
        "width_cm": float(round(width_cm, 1)),
        "is_diagonal": bool(is_diagonal),
        "confidence": float(round(confidence, 2)),
        "last_seen": datetime.now(timezone.utc).isoformat(),
    }
    existing = coll.get(include=["metadatas"])
    found_id = None
    for rec_id, m in zip(existing["ids"], existing["metadatas"]):
        if m.get("hazard_id") == str(hazard_id) or (
            m.get("label") == str(label)
            and abs(float(m.get("world_x_m", 0.0)) - world_x_m) < 0.15
            and abs(float(m.get("world_y_m", 0.0)) - world_y_m) < 0.15
        ):
            found_id = rec_id
            break

    if found_id:
        coll.update(ids=[found_id], documents=[doc], embeddings=[_embed(doc)], metadatas=[meta])
    else:
        coll.add(ids=[str(uuid4())], documents=[doc], embeddings=[_embed(doc)], metadatas=[meta])


# ---------------------------------------------------------------------------
# Internal ChromaDB Helpers
# ---------------------------------------------------------------------------

def _embed(text: str) -> list[float]:
    """Generate text embedding via Ollama nomic-embed-text with deterministic fallback."""
    try:
        return embeddings.embed_query(text)
    except Exception:
        import hashlib
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b / 255.0) - 0.5 for b in (digest * 24)[:768]]


def _query(collection, query: str, limit: int) -> list[dict[str, Any]]:
    """Retrieve memories ordered by semantic cosine similarity."""
    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[_embed(query)],
        n_results=min(count, max(limit, 1)),
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


def _store_unique(collection, document: str, metadata: dict[str, Any]) -> None:
    """Store or update a unique document in ChromaDB."""
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
        embeddings=[_embed(document)],
        metadatas=[metadata],
    )


# ---------------------------------------------------------------------------
# Semantic Memory Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def remember_fact(fact: str, category: str = "general") -> str:
    """Store a durable fact or preference in semantic memory."""
    _store_unique(
        _get_semantic_collection(),
        fact,
        {
            "category": category,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return f"Stored semantic memory: {fact}"


@mcp.tool()
def recall_facts(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Retrieve durable facts that are semantically similar to a query."""
    return _query(_get_semantic_collection(), query, limit)


# ---------------------------------------------------------------------------
# Spatial Memory Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def query_spatial_map(query: str = "") -> dict[str, Any]:
    """
    Retrieve remembered floor hazards and their approximate world coordinates from ChromaDB.
    Enables planning navigation, checking remembered obstacle locations, and tracking odometry drift.

    Args:
        query: Optional semantic filter (e.g. 'cylinder', 'toy', or empty string for all known hazards).

    Returns:
        dict with:
            - known_hazards: list of remembered hazards with world_x_m, world_y_m, shape, width_cm, last_seen.
            - count: total count of remembered hazards on the floor.
            - summary: natural language overview of the spatial map.
    """
    coll = _get_spatial_collection()
    count = coll.count()
    if count == 0:
        return {
            "known_hazards": [],
            "count": 0,
            "summary": "Spatial memory is currently empty. No hazards remembered yet.",
        }

    existing = coll.get(include=["metadatas"])
    hazards = []
    for m in existing.get("metadatas", []):
        hazards.append({
            "hazard_id": m.get("hazard_id", "unknown"),
            "label": m.get("label", "hazard"),
            "world_x_m": float(m.get("world_x_m", 0.0)),
            "world_y_m": float(m.get("world_y_m", 0.0)),
            "shape": m.get("shape", "cylindrical"),
            "width_cm": float(m.get("width_cm", 2.0)),
            "is_diagonal": bool(m.get("is_diagonal", False)),
            "confidence": float(m.get("confidence", 1.0)),
            "last_seen": m.get("last_seen", ""),
        })

    summary_items = [
        f"'{h['label']}' at world (X={h['world_x_m']:.2f}m, Y={h['world_y_m']:.2f}m, shape={h['shape']})"
        for h in hazards
    ]
    return {
        "known_hazards": hazards,
        "count": len(hazards),
        "summary": f"Spatial map contains {len(hazards)} remembered hazard(s): " + "; ".join(summary_items),
    }


# ---------------------------------------------------------------------------
# Episodic Memory Tools
# ---------------------------------------------------------------------------


def _is_shape_compatible(s1: str, s2: str) -> bool:
    """Check if two shape or label strings are semantically compatible for memory retrieval."""
    a, b = s1.lower().strip(), s2.lower().strip()
    if not a or not b or a == b or a in ("object", "hazard", "item") or b in ("object", "hazard", "item"):
        return True
    cyl_set = {"cylinder", "cylindrical", "bottle", "can", "battery", "candle"}
    if a in cyl_set and b in cyl_set:
        return True
    box_set = {"box", "cube", "toy block", "block", "container"}
    if a in box_set and b in box_set:
        return True
    return False


@mcp.tool()
def record_grasp_episode(
    shape: str = "object",
    width_cm: float = 2.8,
    clamp_duration_s: float = 0.60,
    outcome: str = "SUCCESS",
    is_diagonal: bool = False,
    sensor_reading: str = "",
    notes: str = "",
    label: str = "",
) -> str:
    """
    Record a physical grasp trial in ChromaDB episodic memory.

    Args:
        shape: Geometric or object category ('cylindrical', 'box', 'flat', 'battery', etc.).
        width_cm: Head-on grasp dimension in centimetres.
        clamp_duration_s: Close-motor drive duration in seconds.
        outcome: 'SUCCESS', 'SLIP', or 'GRAVITY_SLIP'.
        is_diagonal: True if object was resting diagonally relative to jaws.
        sensor_reading: Pre/post-lift visual audit notes or sensor observations.
        notes: Context, failure reason, or observation.
        label: Optional specific semantic label (e.g. 'orange item').
    """
    norm_outcome = outcome.upper().strip()
    safe_duration = min(float(clamp_duration_s), MAX_SAFE_CLAMP_DURATION_S)
    effective_shape = label.lower().strip() if (shape in ("object", "") and label) else shape.lower().strip()

    document = (
        f"Situation: shape={effective_shape}, width={width_cm:.1f}cm, diagonal={is_diagonal}\n"
        f"Action: clamp_duration_s={safe_duration:.2f}s\n"
        f"Sensor reading: {sensor_reading or 'visual verification'}\n"
        f"Outcome: {norm_outcome}\n"
        f"Notes: {notes or 'none'}"
    )

    metadata: dict[str, Any] = {
        "shape": effective_shape,
        "width_cm": float(round(width_cm, 2)),
        "clamp_duration_s": float(round(safe_duration, 2)),
        "outcome": norm_outcome,
        "is_diagonal": bool(is_diagonal),
        "sensor_reading": sensor_reading or "visual verification",
        "notes": notes or "none",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    _store_unique(_get_episodic_collection(), document, metadata)
    return (
        f"Grasp episode recorded in ChromaDB: {effective_shape} ({width_cm}cm, diagonal={is_diagonal}) "
        f"-> {norm_outcome} using {safe_duration:.2f}s clamp."
    )


@mcp.tool()
def query_grasp_timing(
    shape: str = "object",
    width_cm: float = 2.8,
    is_diagonal: bool = False,
    label: str = "",
) -> dict[str, Any]:
    """
    Query ChromaDB episodic memory to recommend optimal clamp motor duration for an object.
    Grounds recommendations in physical width (cm) and diagonal rotational state (is_diagonal).

    Args:
        shape: Geometric or object category ('cylindrical', 'box', 'battery', 'toy', etc.).
        width_cm: Head-on lateral width in centimetres.
        is_diagonal: True if object is perceived diagonally / rotated relative to gripper jaws.
        label: Optional specific semantic label (e.g. 'orange item').

    Returns:
        Dict with recommended_duration_s, baseline_duration_s, adapted_from_memory,
        rationale, past_attempts, past_slips, and safety_safeguard_active.
    """
    # 1. Baseline default duration from size bins
    if width_cm < 2.0:
        base_duration = 0.40
    elif width_cm <= 3.5:
        base_duration = 0.60
    else:
        base_duration = 0.80

    if is_diagonal:
        base_duration += 0.10

    # Query ChromaDB episodic records
    existing = _get_episodic_collection().get(include=["metadatas"])
    metadatas = existing.get("metadatas", []) or []

    # Semantic similarity query on ChromaDB
    query_text = f"shape={shape}, width={width_cm:.1f}cm, diagonal={is_diagonal}"
    retrieved_raw = _query(_get_episodic_collection(), query_text, limit=2)
    retrieved_episodes = [
        {
            "situation": f"{d.get('metadata', {}).get('shape', '')} ({d.get('metadata', {}).get('width_cm', '')}cm, diagonal={d.get('metadata', {}).get('is_diagonal', False)})",
            "action": f"clamp {d.get('metadata', {}).get('clamp_duration_s', 0.0)}s",
            "outcome": d.get("metadata", {}).get("outcome", ""),
            "sensor_reading": d.get("metadata", {}).get("sensor_reading", ""),
            "similarity": d.get("similarity", 0.0),
        }
        for d in retrieved_raw
    ]

    if not metadatas:
        return {
            "baseline_duration_s": round(base_duration, 2),
            "recommended_duration_s": round(base_duration, 2),
            "adapted_from_memory": False,
            "rationale": f"No past memory in ChromaDB; using baseline sizing rule ({base_duration:.2f}s)",
            "retrieved_episodes": retrieved_episodes,
            "past_attempts": 0,
            "past_slips": 0,
            "past_successes": 0,
            "safety_safeguard_active": False,
        }

    relevant_successes: list[tuple[float, bool]] = []
    relevant_slips: list[tuple[float, bool]] = []

    for m in metadatas:
        m_shape = str(m.get("shape", "")).lower()
        m_width = float(m.get("width_cm", 0.0))
        m_duration = float(m.get("clamp_duration_s", 0.0))
        m_outcome = str(m.get("outcome", "")).upper()
        m_diag = bool(m.get("is_diagonal", False))

        shape_compat = _is_shape_compatible(m_shape, shape) or (label and _is_shape_compatible(m_shape, label))
        if shape_compat and abs(m_width - width_cm) <= 1.2:
            if "SUCCESS" in m_outcome:
                relevant_successes.append((m_duration, m_diag))
            elif "SLIP" in m_outcome:
                relevant_slips.append((m_duration, m_diag))

    past_attempts = len(relevant_successes) + len(relevant_slips)
    past_slips = len(relevant_slips)
    past_successes = len(relevant_successes)

    if relevant_successes:
        diag_successes = [dur for dur, d in relevant_successes if d == is_diagonal]
        target_dur = max(diag_successes) if diag_successes else max(dur for dur, _ in relevant_successes)
        recommended = min(max(target_dur, base_duration), MAX_SAFE_CLAMP_DURATION_S)
        return {
            "baseline_duration_s": round(base_duration, 2),
            "recommended_duration_s": round(recommended, 2),
            "adapted_from_memory": True,
            "past_failure_context": (
                f"Past slip recorded in ChromaDB on {shape} at lower clamp duration."
                if past_slips > 0
                else "No prior slips recorded."
            ),
            "rationale": (
                f"ChromaDB memory adaptation: Grounded in {past_successes} past success(es) on {shape} "
                f"({width_cm:.1f}cm, diagonal={is_diagonal}); recommended {recommended:.2f}s "
                f"to overcome rotational torque."
            ),
            "past_attempts": past_attempts,
            "past_slips": past_slips,
            "past_successes": past_successes,
            "safety_safeguard_active": recommended >= MAX_SAFE_CLAMP_DURATION_S,
            "retrieved_episodes": retrieved_episodes,
        }

    elif relevant_slips:
        max_slipped_dur = max(dur for dur, _ in relevant_slips)
        adapted = max(max_slipped_dur + 0.60, base_duration + 0.50)
        recommended = min(adapted, MAX_SAFE_CLAMP_DURATION_S)
        return {
            "baseline_duration_s": round(base_duration, 2),
            "recommended_duration_s": round(recommended, 2),
            "adapted_from_memory": True,
            "past_failure_context": (
                f"Past {max_slipped_dur:.2f}s clamp SLIPPED on {shape}: parallel jaws caused "
                f"diagonal object to twist and escape grip before full closure."
            ),
            "rationale": (
                f"ChromaDB memory adaptation: Prior clamp of {max_slipped_dur:.2f}s failed with SLIP. "
                f"Extending duration from baseline {base_duration:.2f}s to {recommended:.2f}s "
                f"to complete travel after straightening."
            ),
            "past_attempts": past_attempts,
            "past_slips": past_slips,
            "past_successes": 0,
            "safety_safeguard_active": recommended >= MAX_SAFE_CLAMP_DURATION_S,
            "retrieved_episodes": retrieved_episodes,
        }

    return {
        "baseline_duration_s": round(base_duration, 2),
        "recommended_duration_s": round(base_duration, 2),
        "adapted_from_memory": False,
        "past_failure_context": "No matching past episodes in ChromaDB for this geometry.",
        "rationale": f"Standard sizing rule ({base_duration:.2f}s) based on width {width_cm:.1f}cm.",
        "past_attempts": 0,
        "past_slips": 0,
        "past_successes": 0,
        "safety_safeguard_active": False,
        "retrieved_episodes": retrieved_episodes,
    }


if __name__ == "__main__":
    log.info("Starting ChromaDB Episodic Memory FastMCP Server on stdio...")
    mcp.run(transport="stdio")

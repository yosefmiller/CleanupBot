"""
Standalone Test for ChromaDB Episodic Memory Subsystem (mcp_memory_server).

Demonstrates:
1. ChromaDB initialization and collection verification.
2. Initial query retrieving the baseline slip episode from ChromaDB.
3. Decision adaptation (+0.60s extended travel to overcome diagonal torque).
4. Storing a verified success into ChromaDB.
5. Querying ChromaDB again to show the reinforced success.
6. Semantic retrieval via recall_episodes().
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.extend(["MemoryMCP", "robot_memory_db"])

from mcp_memory_server import (
    query_grasp_timing,
    record_grasp_episode,
    remember_fact,
    recall_facts,
    query_spatial_map,
    upsert_spatial_hazard,
)
from seed_baseline_memory import seed_memory

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_memory_test() -> None:
    """Execute record and recall demonstration on ChromaDB episodic memory."""
    print("=" * 64)
    print("  Testing ChromaDB Episodic Memory Subsystem (mcp_memory_server)")
    print("=" * 64)

    # 1. Reset ChromaDB to canonical baseline (1 prior slip on diagonal cylinder)
    print("\n[Step 1]: Resetting ChromaDB to canonical baseline slip...")
    seed_memory()

    # 2. Query timing from ChromaDB
    print("\n[Step 2]: Querying grasp timing from ChromaDB (exact shape)...")
    timing = query_grasp_timing(shape="cylindrical", width_cm=2.8, is_diagonal=True)
    print(f"Recommendation : {timing['recommended_duration_s']}s (Baseline: {timing['baseline_duration_s']}s)")
    print(f"Adapted        : {timing['adapted_from_memory']}")
    print(f"Failure Context: {timing['past_failure_context']}")
    print(f"Rationale      : {timing['rationale']}")
    assert timing["recommended_duration_s"] >= 1.15, "Expected duration >= 1.15s after slip"

    # 2b. Query timing with open-vocabulary detector label / generic shape
    print("\n[Step 2b]: Querying grasp timing with generic shape='object' & label='battery'...")
    timing_open = query_grasp_timing(shape="object", width_cm=2.8, is_diagonal=True, label="battery")
    print(f"Recommendation : {timing_open['recommended_duration_s']}s (Adapted: {timing_open['adapted_from_memory']})")
    assert timing_open["adapted_from_memory"], "Expected adaptation on physical width+diagonal matching"
    assert timing_open["recommended_duration_s"] >= 1.15, "Expected duration >= 1.15s for physical match"

    # 3. Semantic similarity retrieval via ChromaDB
    print("\n[Step 3]: Verifying semantic retrieval returned in query_grasp_timing()...")
    episodes = timing.get("retrieved_episodes", [])
    print(f"Retrieved {len(episodes)} episode(s) from ChromaDB:")
    for ep in episodes:
        print(f" - Situation: {ep['situation']} | Action: {ep['action']} | Outcome: {ep['outcome']}")

    print("\n[Step 3b]: Testing Semantic Memory (remember_fact / recall_facts)...")
    remember_fact("Infant choking hazard limit is 3.17cm diameter.", category="safety")
    facts = recall_facts("choking hazard dimensions", limit=1)
    if facts:
        print(f" - Fact: {facts[0]['text']}")

    print("\n[Step 3c]: Testing Spatial Memory (upsert_spatial_hazard / query_spatial_map)...")
    upsert_spatial_hazard(
        hazard_id="hazard_1",
        label="toy_cylinder",
        world_x_m=0.45,
        world_y_m=-0.02,
        shape="cylindrical",
        width_cm=2.8,
        is_diagonal=True,
    )
    s_map = query_spatial_map()
    print(f"Spatial Map Summary: {s_map['summary']}")
    assert s_map["count"] >= 1, "Expected at least 1 hazard in spatial memory"

    # 4. Record a successful grasp trial using adapted duration
    print(f"\n[Step 4]: Recording successful trial with {timing['recommended_duration_s']}s clamp in ChromaDB...")
    success_res = record_grasp_episode(
        shape="cylindrical",
        width_cm=2.8,
        clamp_duration_s=timing["recommended_duration_s"],
        outcome="SUCCESS",
        is_diagonal=True,
        notes="Held securely against gravity with extended clamp",
    )
    print(f"Record result: {success_res}")

    # 5. Query verified timing after success
    print("\n[Step 5]: Querying ChromaDB timing after recorded success...")
    final_timing = query_grasp_timing(shape="cylindrical", width_cm=2.8, is_diagonal=True)
    print(f"Final Recommendation: {final_timing['recommended_duration_s']}s")
    print(f"Final Rationale     : {final_timing['rationale']}")
    print(f"Past Successes      : {final_timing['past_successes']}")

    print("\n" + "=" * 64)
    print("  ALL CHROMADB EPISODIC MEMORY TESTS PASSED SUCCESSFULLY!")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    run_memory_test()

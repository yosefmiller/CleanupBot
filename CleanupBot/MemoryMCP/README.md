# MemoryMCP: ChromaDB Robot Episodic & Semantic Memory Server

FastMCP server managing persistent ChromaDB collections and dynamic motor timing adaptation for **CleanupBot5** (Embodied Agentic AI - Module 4).

## Overview
Built with **ChromaDB** (`chromadb.PersistentClient`) and **OllamaEmbeddings** (`nomic-embed-text`), directly extending the course's Module 4 memory architecture (`Robot/MCP/mcp_memory_server.py`).

Because the parallel jaw gripper operates without direct torque sensors, empirical motor timing must be adapted to avoid drop and slip failures caused by object geometry (particularly rotational torque on elongated and cylindrical infant hazards resting diagonally). `MemoryMCP` provides persistent vector storage in `robot_memory_db/chroma.sqlite3`.

## ChromaDB Collections
1. **`episodic_memory`** (Cosine similarity):
   - Stores physical robot experiences: Situation, Action, Sensor Reading, and Outcome.
   - Used for empirical motor clamp adaptation and situation retrieval.
2. **`semantic_memory`** (Cosine similarity):
   - Stores durable facts, project constraints, and user preferences.

## Exposed Tools

### 1. `recall_episodes(query, limit=3)`
- **Purpose**: Semantic similarity retrieval over past robot trials from ChromaDB `episodic_memory`.
- **Returns**: Top matching experiences with cosine similarity scores and trial metadata.

### 2. `record_episode(situation, action, outcome, sensor_reading)`
- **Purpose**: Persists a new experience document and metadata into ChromaDB `episodic_memory`.

### 3. `query_grasp_timing(shape, width_cm, is_diagonal)`
- **Purpose**: Queries ChromaDB episodic memory to compute optimal motor clamp duration for target geometry.
- **Decision Logic**:
  - Baseline hardware rule: $0.40\text{s}$ ($<2.0\text{cm}$), $0.60\text{s}$ ($2.0-3.5\text{cm}$), $0.80\text{s}$ ($>3.5\text{cm}$).
  - Diagonal offset: $+0.10\text{s}$ initial clearance.
  - Memory adaptation: Recalls past episodes in ChromaDB matching shape and size. If a prior attempt slipped, extends clamp duration to $1.20\text{s}$ to overcome rotational torque.
  - Safety ceiling: Capped at $1.40\text{s}$ to prevent DC motor stall overheating.

### 4. `record_grasp_episode(shape, width_cm, clamp_duration_s, outcome, is_diagonal, sensor_reading, notes)`
- **Purpose**: Stores structured grasp trials into ChromaDB with metadata for fast indexing.

### 5. `remember_fact(fact, category)` & `recall_facts(query, limit=3)`
- **Purpose**: Manage project facts in ChromaDB `semantic_memory`.

## Usage & Verification
- **Reset ChromaDB to canonical demonstration baseline**:
  ```bash
  python robot_memory_db/seed_baseline_memory.py
  ```
- **Run automated ChromaDB test suite**:
  ```bash
  python tests/test_memory.py
  ```
- **Launch server over stdio**:
  ```bash
  python MemoryMCP/mcp_memory_server.py
  ```

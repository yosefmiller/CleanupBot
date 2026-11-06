"""Interactive ROSA agent with MCP tools and multiple forms of memory."""

import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from rosa import ROSA, RobotSystemPrompts

from memory_manager import ConversationMemory
from robot_tools import (
  emergency_stop,
  get_robot_status,
  move_forward,
  read_imu,
  read_tof_distance,
  rotate,
)

LOCAL_TOOLS = [read_imu, move_forward, read_tof_distance, rotate, emergency_stop, get_robot_status]
MCP_SERVERS = {
  "datasheets": {
    "command": "python3",
    "args": ["MCP/mcp_datasheet_server.py"],
    "transport": "stdio",
  },
  "weather": {
    "command": "python3",
    "args": ["MCP/mcp_weather_server.py"],
    "transport": "stdio",
  },
  "memory": {
    "command": "python3",
    "args": ["MCP/mcp_memory_server.py"],
    "transport": "stdio",
  },
}


async def build_agent(llm: ChatOllama) -> tuple[ROSA, dict]:
  mcp_tools = await MultiServerMCPClient(MCP_SERVERS).get_tools()
  print(f"Loaded {len(mcp_tools)} MCP tool(s): {[tool.name for tool in mcp_tools]}")

  prompts = RobotSystemPrompts(
    embodiment_and_persona=(
      "You are a memory-aware differential-drive robot assistant. Use the supplied "
      "conversation summary and recent-message window for follow-up questions. When "
      "the user explicitly asks you to remember a durable fact, call remember_fact. "
      "Use recall_facts when prior stored knowledge may help. Use record_episode for "
      "past robot experiences that include a situation, action, and outcome, and use "
      "recall_episodes to retrieve relevant experiences. Do not claim that information "
      "was stored unless the corresponding memory tool succeeded. Treat statements "
      "such as assignments, names, and times as conversational context: acknowledge "
      "them briefly without inventing ROS commands, parameters, services, or tools. "
      "Always check for obstacles before moving, and call emergency_stop when safety is uncertain."
    )
  )
  agent = ROSA(
    ros_version=2,
    llm=llm,
    tools=LOCAL_TOOLS + mcp_tools,
    prompts=prompts,
    verbose=True,
    streaming=True,
  )
  return agent, {tool.name: tool for tool in mcp_tools}


async def run_memory_command(command: str, mcp_tools: dict) -> str | None:
  if command.startswith("/remember "):
    arguments = command.removeprefix("/remember ").split("|", maxsplit=1)
    if len(arguments) != 2:
      return "Usage: /remember category | fact"
    category, fact = (argument.strip() for argument in arguments)
    print("Invoking memory tool: remember_fact")
    return await mcp_tools["remember_fact"].ainvoke({"fact": fact, "category": category})

  if command.startswith("/recall "):
    query = command.removeprefix("/recall ").strip()
    print("Invoking memory tool: recall_facts")
    return str(await mcp_tools["recall_facts"].ainvoke({"query": query, "limit": 3}))

  if command.startswith("/episode "):
    arguments = command.removeprefix("/episode ").split("|", maxsplit=3)
    if len(arguments) != 4:
      return "Usage: /episode situation | action | outcome | sensor reading"
    situation, action, outcome, sensor_reading = (
      argument.strip() for argument in arguments
    )
    print("Invoking memory tool: record_episode")
    return await mcp_tools["record_episode"].ainvoke(
      {
        "situation": situation,
        "action": action,
        "outcome": outcome,
        "sensor_reading": sensor_reading,
      }
    )

  if command.startswith("/episodes "):
    query = command.removeprefix("/episodes ").strip()
    print("Invoking memory tool: recall_episodes")
    return str(await mcp_tools["recall_episodes"].ainvoke({"query": query, "limit": 3}))

  return None


async def run_short_term_command(
  command: str,
  llm: ChatOllama,
  memory: ConversationMemory,
) -> str | None:
  if command.startswith("/note "):
    fact = command.removeprefix("/note ").strip()
    if not fact:
      return "Usage: /note fact"
    answer = f"Noted for this session: {fact}"
    await memory.add_turn(fact, answer)
    return answer

  if command.startswith("/ask "):
    question = command.removeprefix("/ask ").strip()
    if not question:
      return "Usage: /ask question"
    response = await llm.ainvoke(
      [
        SystemMessage(
          content=(
            "Answer only from the supplied memory context. Be concise. "
            "Do not invent commands, tools, parameters, services, or facts. "
            "Do not include User or Assistant role labels. If the answer is not "
            "in the context, say that it is not available in short-term memory."
          )
        ),
        HumanMessage(
          content=f"{memory.render_context()}\n\nQuestion:\n{question}"
        ),
      ]
    )
    answer = str(response.content).strip()
    await memory.add_turn(question, answer)
    return answer

  return None


async def run_grounded_memory_query(
  command: str,
  llm: ChatOllama,
  memory: ConversationMemory,
  mcp_tools: dict,
) -> str | None:
  if not command.startswith("/ask-memory "):
    return None

  question = command.removeprefix("/ask-memory ").strip()
  if not question:
    return "Usage: /ask-memory question"

  print("Invoking memory tool: recall_facts")
  facts = await mcp_tools["recall_facts"].ainvoke({"query": question, "limit": 3})
  print("Invoking memory tool: recall_episodes")
  episodes = await mcp_tools["recall_episodes"].ainvoke({"query": question, "limit": 3})
  fact_evidence = format_memory_evidence(facts, "Semantic Fact")
  episode_evidence = format_memory_evidence(episodes, "Episode")
  response = await llm.ainvoke(
    [
      SystemMessage(
        content=(
          "Answer the question using only the supplied short-term context and "
          "retrieved long-term memories. Distinguish stored safety facts from past "
          "episodes. Preserve measurements, actions, outcomes, and negation exactly "
          "as written in the evidence. Never reverse the meaning of an outcome such "
          "as 'collision avoided.' Cite the evidence labels in the answer. Do not "
          "invent commands, measurements, or events. If the memories do not contain "
          "the answer, say so."
        )
      ),
      HumanMessage(
        content=(
          f"{memory.render_context()}\n\n"
          f"Retrieved semantic facts:\n{fact_evidence}\n\n"
          f"Retrieved episodes:\n{episode_evidence}\n\n"
          f"Question:\n{question}"
        )
      ),
    ]
  )
  answer = str(response.content).strip()
  await memory.add_turn(question, answer)
  return answer


def format_memory_evidence(records, label: str) -> str:
  if not isinstance(records, list):
    records = [records]

  parsed_records = []
  seen_text = set()
  for record in records:
    if isinstance(record, str):
      try:
        record = json.loads(record)
      except json.JSONDecodeError:
        record = {"text": record}
    if not isinstance(record, dict):
      record = {"text": str(record)}

    text = str(record.get("text", "")).strip()
    if not text or text in seen_text:
      continue
    seen_text.add(text)
    parsed_records.append(record)

  if not parsed_records:
    return "(none retrieved)"

  evidence_blocks = []
  for index, record in enumerate(parsed_records, start=1):
    metadata = record.get("metadata", {})
    evidence_blocks.append(
      f"[{label} {index}]\n"
      f"Text (verbatim):\n{record['text']}\n"
      f"Metadata: {json.dumps(metadata, sort_keys=True)}"
    )
  return "\n\n".join(evidence_blocks)


async def run_query(agent: ROSA, memory: ConversationMemory, query: str) -> str:
  agent.clear_chat()
  memory_context = memory.render_context()
  augmented_query = f"{memory_context}\n\nCurrent user request:\n{query}"

  async for event in agent.astream(augmented_query):
    if event["type"] == "final":
      answer = event["content"]
      await memory.add_turn(query, answer)
      agent.clear_chat()
      return answer
    if event["type"] == "error":
      raise RuntimeError(event["content"])

  raise RuntimeError("ROSA completed without returning a final response.")


async def main() -> None:
  llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    base_url="http://localhost:11434",
  )
  memory = ConversationMemory(llm=llm, window_messages=6)
  agent, mcp_tools = await build_agent(llm)

  print("\nMemory-aware ROSA is ready.")
  print("Commands: /note, /ask, /ask-memory, /memory, /clear, /remember, /recall, /episode, /episodes, /quit")
  print("Semantic and episodic memories persist in the memory_db folder.\n")

  while True:
    query = (await asyncio.to_thread(input, "You> ")).strip()
    if not query:
      continue
    if query == "/quit":
      break
    if query == "/memory":
      print(memory.status())
      continue
    if query == "/clear":
      memory.clear()
      agent.clear_chat()
      print("Short-term window and summary memory cleared.")
      continue

    try:
      grounded_result = await run_grounded_memory_query(query, llm, memory, mcp_tools)
      if grounded_result is not None:
        print(f"Memory-grounded answer> {grounded_result}\n")
        continue
      short_term_result = await run_short_term_command(query, llm, memory)
      if short_term_result is not None:
        print(f"Short-term memory> {short_term_result}\n")
        continue
      memory_result = await run_memory_command(query, mcp_tools)
      if memory_result is not None:
        print(f"Memory> {memory_result}\n")
        continue
      print(f"Agent> {await run_query(agent, memory, query)}\n")
    except Exception as error:
      print(f"Agent error: {error}\n")


if __name__ == "__main__":
  asyncio.run(main())
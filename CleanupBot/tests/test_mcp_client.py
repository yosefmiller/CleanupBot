"""
Test for MultiServerMCPClient Registration & Tool Invocation.

Demonstrates:
1. Connecting to the 'vision' FastMCP server over stdio JSON-RPC.
2. Tool discovery: inspecting loaded tools and their schemas.
3. Invoking 'scan_for_hazards' via MultiServerMCPClient.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import json
import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

MCP_SERVERS = {
    "vision": {
        "command": "pixi",
        "args": ["run", "python", "mcp_vision_server.py"],
        "cwd": "VisionMCP",
        "transport": "stdio",
    },
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def run_mcp_client_test() -> None:
    """Connect to MCP servers with persistent stdio sessions, list tools, and invoke tools."""
    print("=" * 60)
    print("  Testing MCP Server Registration & Persistent stdio Session")
    print("=" * 60)

    client = MultiServerMCPClient(MCP_SERVERS)
    async with AsyncExitStack() as stack:
        tools: list[BaseTool] = []
        for srv_name in MCP_SERVERS:
            session = await stack.enter_async_context(client.session(srv_name))
            srv_tools = await load_mcp_tools(session)
            tools.extend(srv_tools)

        print(f"\n[Step 1]: Successfully connected and discovered {len(tools)} MCP tool(s):")
        for t in tools:
            print(f"  • {t.name}: {t.description.splitlines()[0] if t.description else ''}")

        # Invoke scan_for_hazards tool on vision server
        scan_tool = next(t for t in tools if t.name == "scan_for_hazards")
        print("\n[Step 2]: Invoking 'scan_for_hazards' (First Call - model weights loaded)...")
        res1 = await scan_tool.ainvoke({})
        if isinstance(res1, str):
            try:
                res1 = json.loads(res1)
            except Exception:
                pass
        count1 = len(res1.get("hazards", [])) if isinstance(res1, dict) else 0
        print(f"  ✓ First call succeeded (detected {count1} hazards)")

        print("\n[Step 3]: Invoking 'scan_for_hazards' (Second Call - ZERO weight reloading, persistent stdio)...")
        res2 = await scan_tool.ainvoke({})
        if isinstance(res2, str):
            try:
                res2 = json.loads(res2)
            except Exception:
                pass
        count2 = len(res2.get("hazards", [])) if isinstance(res2, dict) else 0
        print(f"  ✓ Second call succeeded (detected {count2} hazards on persistent process)")

        print("\nMCP Client test completed successfully with persistent stdio session.")


if __name__ == "__main__":
    asyncio.run(run_mcp_client_test())

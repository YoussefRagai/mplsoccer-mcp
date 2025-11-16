#!/usr/bin/env python3
import asyncio
from mplsoccer_viz_server import mcp

async def main():
    # Get tools using the list_tools method
    tools = await mcp.list_tools()
    print(f"Number of tools: {len(tools)}")
    print("\nTools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:100] if tool.description else 'No description'}")

asyncio.run(main())

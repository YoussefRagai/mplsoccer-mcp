#!/usr/bin/env python3
"""Debug script to inspect MCP object."""

from mplsoccer_viz_server import mcp
import inspect

print("=== MCP Object Inspection ===")
print(f"Type: {type(mcp)}")
print(f"Dir: {[x for x in dir(mcp) if not x.startswith('_')]}")
print()

# Check for tools attribute
if hasattr(mcp, '_tools'):
    print("Has _tools:", list(mcp._tools.keys()))
elif hasattr(mcp, 'tools'):
    print("Has tools:", list(mcp.tools.keys()))
elif hasattr(mcp, 'router'):
    print("Has router")
    if hasattr(mcp.router, 'tools'):
        print("Router tools:", [tool.name for tool in mcp.router.tools])

# List all methods
print("\nAll attributes:")
for attr in dir(mcp):
    if not attr.startswith('_'):
        obj = getattr(mcp, attr)
        print(f"  {attr}: {type(obj)}")

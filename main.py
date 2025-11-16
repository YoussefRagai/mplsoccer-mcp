#!/usr/bin/env python3
"""Entry point for the mplsoccer MCP server using the built-in streamable HTTP transport."""

from __future__ import annotations

import os

import os

import uvicorn
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route

from mplsoccer_viz_server import mcp


app = mcp.streamable_http_app()

OUTPUT_DIR = os.environ.get("MPLSOCCER_OUTPUT_DIR", "/app/output/mplsoccer")


async def health(_request):
    return JSONResponse({"status": "ok", "service": "mplsoccer_viz"})


async def serve_file(request):
    filename = request.path_params["filename"]
    root = os.path.abspath(OUTPUT_DIR)
    target = os.path.abspath(os.path.join(root, filename))
    if not target.startswith(root) or not os.path.exists(target):
        return JSONResponse({"error": "Not Found"}, status_code=404)
    return FileResponse(target)


app.router.routes.append(Route("/health", endpoint=health))
app.router.routes.append(Route("/files/{filename:path}", endpoint=serve_file))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

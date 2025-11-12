# MPLSoccer Viz MCP Server

A Model Context Protocol (MCP) server that produces rich soccer analytics visualizations using the `mplsoccer` plotting toolkit.

## Purpose

This MCP server provides a secure interface for AI assistants to render tactical boards, heatmaps, and radar charts using mplsoccer's drawing utilities.

## Features

### Current Implementation

- **`draw_pitch`** - Renders customizable pitches with scatter, lines, arrows, polygons, annotations, and image overlays.
- **`heatmap_bin_statistic`** - Builds binned heatmaps with optional statistic labels.
- **`hexbin_density`** - Creates hexbin density maps across the field.
- **`pass_sonar_map`** - Generates pass sonar or sonar grid visuals.
- **`create_radar_chart`** - Produces radar charts with optional comparison overlay and custom fonts.
- **`create_pizza_chart`** - Builds pizza (Nightingale) charts for player profiles.
- **`create_bumpy_chart`** - Renders bumpy charts showing ranking over time.
- **`compose_tactical_freeze`** - Creates freeze-frame tactical boards with players, ball, arrows, and annotations.
- **`animate_frames`** - Generates animated GIF sequences from frame-by-frame data.
- **`goal_angle_and_voronoi`** - Visualizes goal angles and Voronoi regions.
- **`create_grid_layout`** - Composes multi-pitch grids for dashboards.
- **`load_statsbomb_data`** - Pulls StatsBomb open data (competitions, matches, events sample).
- **`standardize_coordinates`** - Converts coordinates between provider systems.
- **`render_shot_map`** - Builds shot maps with automatic player colors and outcome markers.

## Prerequisites

- Docker Desktop with MCP Toolkit enabled
- Docker MCP CLI plugin (`docker mcp`)
- Optional: Volume mount for `/app/output` if you want figures outside the container
- Recommended: Mount `/mcp/output` (default output path) to a host directory via the MCP gateway Docker run command.

## Installation

See the step-by-step instructions provided with the files.

## Usage Examples

In Claude Desktop, you can ask:

- "Render a StatsBomb pitch with comet pass trails, numbered annotations, and a club crest."
- "Create a shot heatmap using bin statistics and label the average xG for each zone."
- "Generate a pass sonar grid for these pass angles and lengths."
- "Draw a radar comparing Player A and Player B across these metrics with custom fonts."
- "Build a pizza chart for this player profile with title and subtitle styling."
- "Produce a bumpy chart showing weekly league positions for three teams."
- "Animate these tracking frames into a GIF with a 200ms interval."
- "Standardize these Opta coordinates into StatsBomb space."
- "Fetch StatsBomb open data for competition 43 season 3 and show a sample of match events."
- "Render a shot map with player colors and outcome shapes from this JSON payload."

## Architecture

Claude Desktop → MCP Gateway → MPLSoccer Viz MCP Server → mplsoccer, Matplotlib, StatsBomb open data
↓
Docker Desktop Volume (optional)
(Figures saved inside `/app/output`)

## Development

### Local Testing

```bash
# Optional: customize output directory (defaults to /mcp/output/mplsoccer)
export MPLSOCCER_OUTPUT_DIR="./output"

# Run directly
python mplsoccer_viz_server.py

# Test MCP protocol
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python mplsoccer_viz_server.py
```

### Adding New Tools

1. Add the function to `mplsoccer_viz_server.py`
2. Decorate with `@mcp.tool()`
3. Update the catalog entry with the new tool name
4. Rebuild the Docker image

### Troubleshooting

**Tools Not Appearing**
- Verify Docker image built successfully
- Check catalog and registry files
- Ensure Claude Desktop config includes custom catalog
- Restart Claude Desktop

**Rendering Issues**
- Confirm JSON payloads are well-formed
- Adjust metric ranges for radar charts to avoid clipping
- Ensure mplsoccer functions requested are supported in the installed version

### Security Considerations

- No secrets required; all data supplied per request
- Runs as non-root user in container
- Figures written to container-local directory unless volume-mounted

### License

MIT License

# MPLSoccer Viz MCP Server Notes

## Overview
- **Service name:** MPLSoccer Viz
- **Entry script:** `mplsoccer_viz_server.py`
- **Purpose:** Produce soccer pitches, heatmaps, radars, and freeze frames via mplsoccer.

## Configuration
- Env vars:
  - `MPLSOCCER_OUTPUT_DIR` (defaults to `/mcp/output/mplsoccer`)
  - `MPLSOCCER_DEFAULT_CMAP` (defaults to `viridis`)
  - `MPLSOCCER_TIMEOUT_SECONDS` is unused; plotting is local.
- Figures saved to `/mcp/output/mplsoccer`; ensure the MCP gateway mounts `/mcp/output` to the host if you need access.

## Tools
- `draw_pitch(layout="", primitives="", title="", save_as="")`
  - Layout JSON controls pitch orientation, colors, padding.
  - Primitives JSON supports `scatter`, `lines` (with `comet`/`transparent`), `arrows`, `polygons`, `annotations`, `images`.
- `heatmap_bin_statistic(data="", statistic="", cmap="", title="", save_as="", bins="")`
  - Uses `bin_statistic` + `heatmap` + `label_heatmap`.
- `hexbin_density(data="", gridsize="", cmap="", title="", save_as="")`
  - Calls `Pitch.hexbin` with configurable grid.
- `pass_sonar_map(data="", title="", sonar_type="", save_as="")`
  - Accepts angle/length/weight arrays; supports standard or grid sonar.
- `create_radar_chart(metrics="", compare="", title="", save_as="", fonts="")`
  - Optional comparison dataset and font URLs via `FontManager`.
- `create_pizza_chart(metrics="", styling="", title="", subtitle="", save_as="", fonts="")`
  - Renders PyPizza with custom slice colors and text blocks.
- `create_bumpy_chart(data="", title="", subtitle="", save_as="", theme="")`
  - Utilizes `Bumpy` for ranking timelines.
- `compose_tactical_freeze(frame="", layout="", title="", save_as="")`
  - Tactical snapshot with players, opponents, ball, directional arrows.
- `animate_frames(frames="", layout="", title="", save_as="", interval="")`
  - Builds GIF via `FuncAnimation` + Pillow writer.
- `goal_angle_and_voronoi(data="", layout="", save_as="")`
  - Visualizes goal angles and Voronoi regions.
- `create_grid_layout(layouts="", config="", save_as="")`
  - Multi-pitch dashboards using Matplotlib grids.
- `load_statsbomb_data(competition="", season="", match="", limit="")`
  - Wraps `Sbopen` to fetch competitions/matches/events (sample).
- `standardize_coordinates(data="", provider="", target="")`
  - Uses `Standardizer` for coordinate transforms.
- `render_shot_map(shots="", title="", subtitle="", save_as="", pitch_type="", orientation="", background="", line_color="")`
  - Auto-styles shot maps with player colors and outcome-specific markers, returns base64 image.

## Error Handling
- JSON parsing errors returned with friendly ❌ messages.
- Missing coordinates or mismatched metric lengths reported to user.
- Heatmap/hexbin/sonar gracefully handle invalid parameters.
- StatsBomb loader reports initialization/network issues.

## Logging
- INFO-level logging to stderr via `logging.basicConfig`.
- Key actions logged with context; exceptions captured with stack traces.

## Docker Runtime
- Based on `python:3.11-slim`.
- Dependencies: `mplsoccer`, `matplotlib`, `numpy`, `pandas`, `scipy`, `Pillow`.
- Non-root user `mcpuser`.

## Testing
```bash
export MPLSOCCER_OUTPUT_DIR="./output"
python mplsoccer_viz_server.py
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python mplsoccer_viz_server.py
```
- To validate StatsBomb loader: `python -c "from mplsoccer_viz_server import load_statsbomb_data; ..."` (ensure network).

## Maintenance
- Keep docstrings single-line.
- Close Matplotlib figures after saving to avoid memory growth.
- Reuse helpers (`_apply_primitives`, `_load_json`) when adding visuals.
- Animation relies on Pillow writer; ensure new formats include fallback handling.

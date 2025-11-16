#!/usr/bin/env python3
"""Simple MPLSoccer Viz MCP Server - Generates advanced soccer visuals and data utilities."""

import os
import sys
import json
import logging
from io import BytesIO
from datetime import datetime, timezone
import base64
import mimetypes

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
from PIL import Image
import httpx

from mplsoccer import (
    Pitch,
    VerticalPitch,
    Radar,
    PyPizza,
    Bumpy,
    FontManager,
    Sbopen,
    Standardizer,
)
from mplsoccer.utils import add_image, inset_image
from matplotlib.lines import Line2D

from mcp.server.fastmcp import FastMCP


LOG_LEVEL = os.environ.get("MCP_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mplsoccer_viz-server")

mcp = FastMCP("mplsoccer_viz", stateless_http=True)

OUTPUT_DIR = os.environ.get("MPLSOCCER_OUTPUT_DIR", "/app/output/mplsoccer")
FALLBACK_OUTPUT_DIR = os.environ.get("MPLSOCCER_FALLBACK_OUTPUT_DIR", "/tmp/mplsoccer")
DEFAULT_CMAP = os.environ.get("MPLSOCCER_DEFAULT_CMAP", "viridis")


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_output_dir() -> str:
    for path in (OUTPUT_DIR, FALLBACK_OUTPUT_DIR):
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except Exception as exc:
            logger.error("Failed to ensure output directory %s: %s", path, exc)
    return FALLBACK_OUTPUT_DIR


def _generate_filename(prefix: str, extension: str) -> str:
    safe_prefix = prefix.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_prefix}_{stamp}.{extension}"


def _save_figure(fig, prefix: str, provided_name: str, extension: str = "png") -> str:
    directory = _ensure_output_dir()
    if provided_name.strip():
        filename = provided_name.strip()
        if "." not in filename:
            filename = f"{filename}.{extension}"
    else:
        filename = _generate_filename(prefix, extension)
    path = os.path.join(directory, filename)
    if os.path.isabs(filename):
        path = filename
    output_dir = os.path.dirname(path)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create output directory %s: %s", output_dir, exc)
            raise
    try:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    except Exception as exc:
        logger.error("Error saving figure: %s", exc)
        raise
    finally:
        plt.close(fig)
    return path


def _save_animation(anim, prefix: str, provided_name: str) -> str:
    directory = _ensure_output_dir()
    filename = provided_name.strip() if provided_name.strip() else _generate_filename(prefix, "gif")
    if not filename.endswith(".gif"):
        filename = f"{filename}.gif"
    path = os.path.join(directory, filename)
    if os.path.isabs(filename):
        path = filename
    output_dir = os.path.dirname(path)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create output directory %s: %s", output_dir, exc)
            raise
    try:
        writer = animation.PillowWriter(fps=anim.save_count if anim.save_count > 0 else 10)
        anim.save(path, writer=writer)
    except Exception as exc:
        logger.error("Animation save failed: %s", exc)
        raise
    return path


def _encode_file(path: str) -> tuple:
    try:
        with open(path, "rb") as handle:
            data = handle.read()
        encoded = base64.b64encode(data).decode("ascii")
        mime, _ = mimetypes.guess_type(path)
        return encoded, mime or "application/octet-stream"
    except Exception as exc:
        logger.error("Failed to read output file %s: %s", path, exc)
        return "", "application/octet-stream"


def _load_json(raw: str, description: str):
    if not raw.strip():
        return False, f"❌ Error: {description} JSON is required."
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error for %s: %s", description, exc)
        return False, f"❌ Error: Invalid {description} JSON ({exc})."
    return True, data


async def _fetch_image(url: str):
    if not url.strip():
        return False, "❌ Error: Image URL is empty."
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url.strip())
            response.raise_for_status()
        return True, Image.open(BytesIO(response.content)).convert("RGBA")
    except Exception as exc:
        logger.error("Image fetch failed for %s: %s", url, exc)
        return False, f"❌ Error: Failed to fetch image ({exc})."


def _load_font_resource(resource: str):
    if not resource or not resource.strip():
        return None
    try:
        fm = FontManager(resource.strip())
        return fm.path
    except Exception as exc:
        logger.warning("Font load failed for %s: %s", resource, exc)
        return None


def _format_success(summary: str, path: str, include_base64: bool = True) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    base64_block = "(base64 omitted; set include_base64=true to embed)"
    if include_base64:
        encoded, mime = _encode_file(path)
        base64_block = encoded if encoded else "(file read failed)"
    download_hint = "(serve file manually)"
    for base in (OUTPUT_DIR, FALLBACK_OUTPUT_DIR):
        try:
            root = os.path.abspath(base)
            target = os.path.abspath(path)
            if target.startswith(root):
                rel = os.path.relpath(target, root).replace("\\", "/")
                download_hint = f"/files/{rel}"
                break
        except Exception:
            continue
    return f"""✅ Success:
- File saved: {path}
- Download URL: {download_hint}
- MIME type: {mime}
- Base64 payload:
{base64_block}

Note: If file mounts are blocked, decode the base64 payload directly; no additional setup is required.

Summary: {summary} | Generated at {_iso_timestamp()}"""


def _format_json_output(obj, label: str) -> str:
    try:
        text = json.dumps(obj, indent=2)
    except TypeError:
        text = json.dumps(str(obj))
    limit = 4000
    if len(text) > limit:
        text = text[:limit] + "\n... (truncated)"
    return f"""📊 {label}:
{text}

Summary: Generated at {_iso_timestamp()}"""


SUPPORTED_PITCH_ALIASES = {
    "statsbomb": "statsbomb",
    "opta": "opta",
    "wyscout": "wyscout",
    "tracab": "tracab",
    "uefa": "uefa",
    "metricasports": "metricasports",
    "skillcorner": "skillcorner",
    "secondspectrum": "secondspectrum",
    "impect": "impect",
    "custom": "custom",
}


def _build_pitch(layout: dict):
    orientation = layout.get("orientation", "horizontal")
    pitch_key = (layout.get("pitch_type") or "statsbomb").lower()
    if pitch_key in {"korastats", "kora"}:
        pitch_type = "custom"
        default_length = 100.0
        default_width = 100.0
    else:
        pitch_type = SUPPORTED_PITCH_ALIASES.get(pitch_key, "statsbomb")
        default_length = None
        default_width = None
    pitch_color = layout.get("pitch_color", "#22312b")
    line_color = layout.get("line_color", "#c7d5cc")
    pad_bottom = float(layout.get("pad_bottom", 0.02))
    pad_top = float(layout.get("pad_top", 0.02))
    pad_left = float(layout.get("pad_left", 0.02))
    pad_right = float(layout.get("pad_right", 0.02))
    goal_type = layout.get("goal_type", "box")
    kwargs = {
        "pitch_type": pitch_type,
        "pitch_color": pitch_color,
        "line_color": line_color,
        "goal_type": goal_type,
        "pad_bottom": pad_bottom,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }
    pitch_length = layout.get("pitch_length", default_length)
    pitch_width = layout.get("pitch_width", default_width)
    if pitch_length:
        kwargs["pitch_length"] = float(pitch_length)
    if pitch_width:
        kwargs["pitch_width"] = float(pitch_width)

    if orientation.lower() == "vertical":
        return VerticalPitch(**kwargs)
    return Pitch(**kwargs)


COLOR_PALETTE = [
    "#ff6b6b",
    "#4ecdc4",
    "#1a535c",
    "#ffa36c",
    "#c77dff",
    "#ffd93d",
    "#6bcf99",
    "#b08ea2",
    "#00bcd4",
    "#ff8f00",
    "#3a86ff",
    "#ffbe0b",
    "#8338ec",
    "#ff006e",
]

OUTCOME_MARKERS = {
    "goal": ("*", 16),
    "ontarget": ("o", 12),
    "offtarget": ("x", 12),
    "blockbydefense": ("s", 12),
    "bars": ("d", 14),
}


def _get_player_color(index: int) -> str:
    return COLOR_PALETTE[index % len(COLOR_PALETTE)]


def _marker_for_outcome(outcome: str) -> tuple:
    if not outcome:
        return ("^", 12)
    key = outcome.lower()
    return OUTCOME_MARKERS.get(key, ("^", 12))


def _normalize_primitives(elements):
    if isinstance(elements, list):
        buckets = {}
        for entry in elements:
            entry_type = (entry.get("type") or "").lower()
            buckets.setdefault(entry_type, []).append(entry)
        return buckets
    return elements or {}


def _collect_images(*sources):
    images = []
    for source in sources:
        if not source:
            continue
        if isinstance(source, dict):
            for key in ("images", "image"):
                value = source.get(key)
                if isinstance(value, list):
                    images.extend(value)
        elif isinstance(source, list):
            normalized = _normalize_primitives(source)
            for key in ("images", "image"):
                value = normalized.get(key)
                if isinstance(value, list):
                    images.extend(value)
        else:
            normalized = _normalize_primitives(source)
            if isinstance(normalized, dict):
                for key in ("images", "image"):
                    value = normalized.get(key)
                    if isinstance(value, list):
                        images.extend(value)
    return images


def _coerce_marker(marker: str) -> str:
    if not marker:
        return "o"
    marker_lower = marker.lower()
    if marker_lower == "star":
        return "*"
    if marker_lower in {"x", "+", "o", "s", "d", "^", "v", "<", ">", "p", "h"}:
        return marker_lower
    return marker


def _apply_primitives(pitch, ax, elements):
    normalized = _normalize_primitives(elements)
    scatter_items = normalized.get("scatter", [])
    for item in scatter_items:
        try:
            label = item.get("label")
            if "data" in item and isinstance(item["data"], list):
                xs = [float(pt[0]) for pt in item["data"]]
                ys = [float(pt[1]) for pt in item["data"]]
            else:
                x = item.get("x")
                y = item.get("y")
                if isinstance(x, list) and isinstance(y, list):
                    xs = [float(v) for v in x]
                    ys = [float(v) for v in y]
                else:
                    xs = [float(x)]
                    ys = [float(y)]
            kwargs = {
                "ax": ax,
                "s": float(item.get("s", item.get("size", 120))),
                "c": item.get("color", "#f94144"),
                "marker": _coerce_marker(item.get("marker")),
                "linewidth": float(item.get("linewidth", item.get("linewidths", 1.0))),
                "alpha": float(item.get("alpha", 0.9)),
                "zorder": float(item.get("zorder", 6)),
            }
            edge = item.get("edgecolor", item.get("edgecolors"))
            if edge:
                kwargs["edgecolors"] = edge
            if label:
                kwargs["label"] = label
            pitch.scatter(
                xs,
                ys,
                **kwargs,
            )
        except Exception as exc:
            logger.warning("Scatter rendering skipped: %s", exc)

    line_items = normalized.get("lines", [])
    for item in line_items:
        try:
            points = item.get("points", [])
            if len(points) < 2:
                continue
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            pitch.lines(
                xs,
                ys,
                ax=ax,
                lw=float(item.get("linewidth", 3)),
                color=item.get("color", "#90be6d"),
                alpha=float(item.get("alpha", 0.8)),
                zorder=float(item.get("zorder", 5)),
                comet=bool(item.get("comet", False)),
                transparent=bool(item.get("transparent", False)),
            )
        except Exception as exc:
            logger.warning("Line rendering skipped: %s", exc)

    arrow_items = normalized.get("arrows", [])
    for item in arrow_items:
        try:
            pitch.arrows(
                float(item.get("start_x", 0)),
                float(item.get("start_y", 0)),
                float(item.get("end_x", 0)),
                float(item.get("end_y", 0)),
                ax=ax,
                width=float(item.get("width", 2)),
                headwidth=float(item.get("headwidth", 3)),
                color=item.get("color", "#577590"),
                alpha=float(item.get("alpha", 0.9)),
                zorder=float(item.get("zorder", 6)),
            )
        except Exception as exc:
            logger.warning("Arrow rendering skipped: %s", exc)

    polygon_items = normalized.get("polygons", [])
    for item in polygon_items:
        try:
            coords = item.get("points", [])
            if len(coords) < 3:
                continue
            pts = [(float(p[0]), float(p[1])) for p in coords]
            pitch.polygon(
                pts,
                ax=ax,
                facecolor=item.get("facecolor", "#f9c74f"),
                edgecolor=item.get("edgecolor", "#f9844a"),
                alpha=float(item.get("alpha", 0.3)),
                lw=float(item.get("linewidth", 1.5)),
                zorder=float(item.get("zorder", 4)),
            )
        except Exception as exc:
            logger.warning("Polygon rendering skipped: %s", exc)

    text_items = normalized.get("annotations", [])
    for item in text_items:
        try:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            ax.text(
                float(item.get("x", 0)),
                float(item.get("y", 0)),
                text,
                color=item.get("color", "#ffffff"),
                fontsize=float(item.get("size", 12)),
                ha=item.get("ha", "center"),
                va=item.get("va", "center"),
                rotation=float(item.get("rotation", 0)),
                zorder=float(item.get("zorder", 10)),
            )
        except Exception as exc:
            logger.warning("Text annotation skipped: %s", exc)


async def _apply_images(fig, ax, images):
    for item in images:
        url = item.get("url", "").strip()
        if not url:
            continue
        ok, resource = await _fetch_image(url)
        if not ok:
            logger.warning("Image skipped: %s", resource)
            continue
        zoom = float(item.get("zoom", 0.2))
        coords = item.get("coords", [0, 0])
        location = item.get("location", "pitch")
        try:
            if location == "inset":
                inset_image(
                    resource,
                    ax=ax,
                    zoom=zoom,
                    xy=(float(coords[0]), float(coords[1])),
                )
            else:
                add_image(
                    resource,
                    fig=fig,
                    left=float(item.get("left", 0.5)),
                    bottom=float(item.get("bottom", 0.5)),
                    width=float(item.get("width", 0.1)),
                )
        except Exception as exc:
            logger.warning("Failed to place image: %s", exc)


@mcp.tool()
async def render_shot_map(
    shots: str = "",
    title: str = "",
    subtitle: str = "",
    save_as: str = "",
    pitch_type: str = "",
    orientation: str = "",
    background: str = "",
    line_color: str = "",
    include_base64: str = "false",
    split_by_team: str = "",
    left_team: str = "",
    right_team: str = "",
    left_team_color: str = "",
    right_team_color: str = "",
) -> str:
    """Render a shot map; supports optional team-based split with automatic colors."""
    ok, shot_rows = _load_json(shots, "shots")
    if not ok:
        return shot_rows
    if not isinstance(shot_rows, list) or not shot_rows:
        return "❌ Error: shots must be a non-empty JSON array."

    cleaned_rows = []
    for idx, item in enumerate(shot_rows):
        if not isinstance(item, dict):
            continue
        try:
            x_value = float(item.get("x", item.get("start_x")))
            y_value = float(item.get("y", item.get("start_y")))
            if x_value is None or y_value is None:
                continue
            player_name = (
                str(
                    item.get("player_name")
                    or item.get("player")
                    or item.get("name")
                    or f"Player {idx+1}"
                )
            ).strip()
            shot_outcome = (
                str(
                    item.get("shot_outcome")
                    or item.get("result")
                    or item.get("outcome")
                    or "Shot"
                )
            ).strip()
            team_value = (
                str(
                    item.get("team_name")
                    or item.get("team")
                    or item.get("club")
                    or ""
                )
            ).strip()
            cleaned_rows.append(
                {
                    "player_name": player_name or f"Player {idx+1}",
                    "x": float(x_value),
                    "y": float(y_value),
                    "shot_outcome": shot_outcome or "Shot",
                    "team_name": team_value,
                    "raw": item,
                }
            )
        except (TypeError, ValueError):
            continue

    if not cleaned_rows:
        return "❌ Error: Unable to parse any valid shot rows (require x/y coordinates at minimum)."

    teams_detected = []
    for row in cleaned_rows:
        team = row.get("team_name", "")
        if team and team not in teams_detected:
            teams_detected.append(team)

    split_flag = split_by_team.strip().lower() == "true" or bool(left_team.strip() or right_team.strip())
    left_candidates = [name.strip() for name in left_team.split(",") if name.strip()]
    right_candidates = [name.strip() for name in right_team.split(",") if name.strip()]

    if split_flag:
        if not left_candidates and teams_detected:
            left_candidates = [teams_detected[0]]
        if not right_candidates and len(teams_detected) > 1:
            right_candidates = [team for team in teams_detected if team not in left_candidates][:1]

    left_lookup = {name.lower() for name in left_candidates}
    right_lookup = {name.lower() for name in right_candidates}
    default_right_team = None

    team_color_map = {}
    if split_flag and left_candidates:
        team_color_map[left_candidates[0]] = left_team_color.strip() or "#DC143C"
    if split_flag and right_candidates:
        team_color_map[right_candidates[0]] = right_team_color.strip() or "#4169E1"

    layout = {
        "pitch_type": pitch_type.strip() or "korastats",
        "orientation": orientation.strip() or "horizontal",
    }
    if layout["pitch_type"].lower() in {"korastats", "kora", "custom"}:
        layout["pitch_type"] = layout["pitch_type"]
        layout["pitch_length"] = 100
        layout["pitch_width"] = 100
    if background.strip():
        layout["pitch_color"] = background.strip()
    if line_color.strip():
        layout["line_color"] = line_color.strip()

    pitch = _build_pitch(layout)
    figsize = (12, 8)
    fig, ax = pitch.draw(figsize=figsize)

    if title.strip():
        fig.suptitle(title.strip(), color="#ffffff", fontsize=20)
    if subtitle.strip():
        fig.text(0.5, 0.93, subtitle.strip(), ha="center", color="#dddddd", fontsize=12)

    pitch_length = layout.get("pitch_length", 100 if layout["pitch_type"] == "korastats" else 120)

    outcome_labels_shown = set()
    teams_present = set()
    for row in cleaned_rows:
        player = row["player_name"]
        outcome = row["shot_outcome"]
        team = row.get("team_name", "")
        teams_present.add(team)
        x_coord = row["x"]
        if split_flag and team:
            team_lower = team.lower()
            if left_lookup:
                if team_lower in left_lookup:
                    x_coord = float(pitch_length) - x_coord
            elif right_lookup:
                if team_lower not in right_lookup:
                    x_coord = float(pitch_length) - x_coord
            else:
                if default_right_team is None:
                    default_right_team = team_lower
                if team_lower != default_right_team:
                    x_coord = float(pitch_length) - x_coord
        marker, size = _marker_for_outcome(outcome)
        color = team_color_map.get(team)
        if not color:
            next_index = len(team_color_map)
            color = _get_player_color(next_index)
            if team:
                team_color_map.setdefault(team, color)
        ax.scatter(
            [x_coord],
            [row["y"]],
            c=color,
            marker=marker,
            s=size * 6,
            edgecolors="#ffffff",
            linewidths=1.2,
            alpha=0.9,
            zorder=6,
        )
        outcome_labels_shown.add(outcome)

    team_handles = []
    if split_flag and team_color_map:
        for team, color in team_color_map.items():
            if not team:
                continue
            team_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=10,
                    label=team,
                    linestyle="None",
                )
            )

    outcome_handles = []
    for outcome in sorted(outcome_labels_shown):
        marker, size = _marker_for_outcome(outcome)
        outcome_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="#333333" if marker != "x" else "#ffffff",
                markerfacecolor="#333333" if marker not in {"x", "*"} else "none",
                markeredgecolor="#333333",
                markersize=max(7, size / 1.5),
                label=outcome,
                linestyle="None",
                markeredgewidth=1.5,
            )
        )

    first_legend = ax.legend(
        handles=team_handles if team_handles else outcome_handles,
        title="Teams" if team_handles else "Shot Outcome",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )
    if team_handles:
        ax.add_artist(first_legend)
        ax.legend(
            handles=outcome_handles,
            title="Shot Outcome",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.55),
            frameon=False,
        )
    else:
        ax.add_artist(first_legend)

    if split_flag and left_candidates and right_candidates:
        ax.text(
            5,
            layout.get("pitch_width", 100) + 3,
            f"{left_candidates[0]} attacking →",
            color=team_color_map.get(left_candidates[0], "#ffffff"),
            fontsize=11,
            ha="left",
        )
        ax.text(
            layout.get("pitch_length", 100) - 5,
            -3,
            f"← {right_candidates[0]} attacking",
            color=team_color_map.get(right_candidates[0], "#ffffff"),
            fontsize=11,
            ha="right",
        )

    fig.tight_layout(rect=[0, 0, 0.82, 1])

    path = _save_figure(fig, "shot_map", save_as, "png")
    include_b64 = include_base64.strip().lower() == "true"
    return _format_success("Shot map rendered.", path, include_b64)


@mcp.tool()
async def draw_pitch(layout: str = "", primitives: str = "", title: str = "", save_as: str = "", include_base64: str = "false") -> str:
    """Render a soccer pitch with advanced primitives."""
    layout_data = {}
    if layout.strip():
        ok, parsed = _load_json(layout, "layout")
        if not ok:
            return parsed
        layout_data = parsed

    primitives_source = {}
    if primitives.strip():
        ok, parsed = _load_json(primitives, "primitives")
        if not ok:
            return parsed
        primitives_source = parsed

    try:
        pitch = _build_pitch(layout_data)
        figsize = layout_data.get("figsize", [12, 8])
        fig, ax = pitch.draw(figsize=tuple(figsize))
        if title.strip():
            fig.suptitle(
                title.strip(),
                color=layout_data.get("title_color", "#ffffff"),
                fontsize=layout_data.get("title_size", 18),
            )
        normalized_primitives = _normalize_primitives(primitives_source)
        _apply_primitives(pitch, ax, normalized_primitives)
        image_entries = _collect_images(primitives_source, normalized_primitives)
        if image_entries:
            await _apply_images(fig, ax, image_entries)
        path = _save_figure(fig, "pitch", save_as, "png")
        include_b64 = include_base64.strip().lower() == "true"
        return _format_success("Pitch with primitives rendered.", path, include_b64)
    except Exception as exc:
        logger.error("draw_pitch failed: %s", exc, exc_info=True)
        return f"❌ Error: {str(exc)}"


@mcp.tool()
async def heatmap_bin_statistic(data: str = "", statistic: str = "", cmap: str = "", title: str = "", save_as: str = "", bins: str = "") -> str:
    """Generate a binned heatmap with optional labels."""
    ok, events = _load_json(data, "event data")
    if not ok:
        return events

    if not isinstance(events, list) or not events:
        return "❌ Error: event data must be a non-empty JSON array."

    xs = [float(item.get("x", item.get("start_x", 0))) for item in events]
    ys = [float(item.get("y", item.get("start_y", 0))) for item in events]
    values = [float(item.get("value", 1)) for item in events]

    pitch = Pitch(pitch_color="#1b263b", line_color="#ffffff")
    fig, ax = pitch.draw(figsize=(12, 8))
    try:
        bins_value = int(bins.strip()) if bins.strip() else 30
    except ValueError:
        return f"❌ Error: Invalid bins value: {bins}"

    try:
        bin_stat = pitch.bin_statistic(
            xs,
            ys,
            values=values if statistic.strip() else None,
            statistic=statistic.strip() if statistic.strip() else "count",
            bins=(bins_value, bins_value),
        )
        heat = pitch.heatmap(bin_stat, ax=ax, cmap=cmap.strip() or DEFAULT_CMAP)
        pitch.label_heatmap(bin_stat, ax=ax, color="#222222", str_format="{:.1f}")
    except Exception as exc:
        logger.error("Heatmap generation failed: %s", exc)
        plt.close(fig)
        return f"❌ Error: Failed to generate heatmap ({exc})."

    if title.strip():
        fig.suptitle(title.strip(), color="#ffffff", fontsize=18)

    try:
        path = _save_figure(fig, "heatmap", save_as, "png")
    except Exception as exc:
        return f"❌ Error: {str(exc)}"

    return _format_success("Binned heatmap generated.", path)


@mcp.tool()
async def hexbin_density(data: str = "", gridsize: str = "", cmap: str = "", title: str = "", save_as: str = "") -> str:
    """Create a hexbin density map on the pitch."""
    ok, events = _load_json(data, "event data")
    if not ok:
        return events

    if not isinstance(events, list) or not events:
        return "❌ Error: event data must be a non-empty JSON array."

    xs = [float(item.get("x", item.get("start_x", 0))) for item in events]
    ys = [float(item.get("y", item.get("start_y", 0))) for item in events]
    pitch = Pitch(pitch_color="#0b132b", line_color="#f0f3bd")
    fig, ax = pitch.draw(figsize=(12, 8))
    try:
        gs = int(gridsize.strip()) if gridsize.strip() else 20
    except ValueError:
        return f"❌ Error: Invalid gridsize value: {gridsize}"

    try:
        pitch.hexbin(
            xs,
            ys,
            ax=ax,
            gridsize=gs,
            cmap=cmap.strip() or "inferno",
            linewidths=0.5,
            edgecolors="#f0f3bd",
        )
    except Exception as exc:
        logger.error("Hexbin failed: %s", exc)
        plt.close(fig)
        return f"❌ Error: Failed to generate hexbin ({exc})."

    if title.strip():
        fig.suptitle(title.strip(), color="#ffffff", fontsize=18)

    try:
        path = _save_figure(fig, "hexbin", save_as, "png")
    except Exception as exc:
        return f"❌ Error: {str(exc)}"

    return _format_success("Hexbin density map generated.", path)


@mcp.tool()
async def pass_sonar_map(data: str = "", title: str = "", sonar_type: str = "", save_as: str = "") -> str:
    """Generate a pass sonar visualization."""
    ok, events = _load_json(data, "event data")
    if not ok:
        return events

    if not isinstance(events, list) or not events:
        return "❌ Error: event data must be a non-empty JSON array."

    angles = []
    lengths = []
    weights = []
    for item in events:
        angles.append(float(item.get("angle", 0)))
        lengths.append(float(item.get("length", item.get("distance", 5))))
        weights.append(float(item.get("weight", 1)))

    pitch = Pitch(pitch_color="#121420", line_color="#eeeeee")
    fig, ax = pitch.draw(figsize=(10, 10))
    try:
        if sonar_type.strip().lower() == "grid":
            pitch.sonar_grid(
                angles,
                lengths,
                values=weights,
                ax=ax,
                cmap="coolwarm",
            )
        else:
            pitch.sonar(
                angles,
                lengths,
                values=weights,
                ax=ax,
                cmap="coolwarm",
                colorbar=True,
            )
    except Exception as exc:
        logger.error("Sonar creation failed: %s", exc)
        plt.close(fig)
        return f"❌ Error: Failed to generate pass sonar ({exc})."

    if title.strip():
        fig.suptitle(title.strip(), color="#ffffff", fontsize=18)

    try:
        path = _save_figure(fig, "sonar", save_as, "png")
    except Exception as exc:
        return f"❌ Error: {str(exc)}"

    return _format_success("Pass sonar visual generated.", path)


@mcp.tool()
async def create_radar_chart(metrics: str = "", compare: str = "", title: str = "", save_as: str = "", fonts: str = "") -> str:
    """Render a radar chart for player or team metrics."""
    ok, metrics_data = _load_json(metrics, "metrics")
    if not ok:
        return metrics_data

    labels = metrics_data.get("labels")
    values = metrics_data.get("values")
    min_range = metrics_data.get("min_range", [0] * len(labels))
    max_range = metrics_data.get("max_range", [1] * len(labels))

    if not isinstance(labels, list) or not isinstance(values, list):
        return "❌ Error: metrics labels and values must be lists."

    if len(labels) != len(values):
        return "❌ Error: metrics labels and values length mismatch."

    comparison = None
    if compare.strip():
        ok, compare_data = _load_json(compare, "compare")
        if not ok:
            return compare_data
        comparison = compare_data.get("values")
        if comparison and len(comparison) != len(labels):
            return "❌ Error: compare values must match label count."

    font_paths = {}
    if fonts.strip():
        ok, font_data = _load_json(fonts, "fonts")
        if ok and isinstance(font_data, dict):
            for key, val in font_data.items():
                font_paths[key] = _load_font_resource(val)

    try:
        radar = Radar(labels, min_range, max_range)
        fig, ax = radar.setup_chart(figsize=(8, 8))
        radar.draw_circles(ax=ax)
        radar.draw_radar(values, ax=ax, kwargs_radar={"color": "#4cc9f0", "alpha": 0.6}, kwargs_rings={"color": "#4cc9f0", "alpha": 0.25})
        if comparison:
            radar.draw_radar(comparison, ax=ax, kwargs_radar={"color": "#f72585", "alpha": 0.5}, kwargs_rings={"color": "#f72585", "alpha": 0.2})
        radar.draw_param_labels(ax=ax, fontsize=metrics_data.get("label_fontsize", 12), fontfamily=font_paths.get("labels"))
        radar.draw_range_labels(ax=ax, fontsize=metrics_data.get("range_fontsize", 10), fontfamily=font_paths.get("ranges"))
        if title.strip():
            fig.suptitle(title.strip(), color="#ffffff", fontsize=metrics_data.get("title_size", 18), fontfamily=font_paths.get("title"))
        background = metrics_data.get("background_color", "#000000")
        fig.set_facecolor(background)
        ax.set_facecolor(background)
        path = _save_figure(fig, "radar", save_as, "png")
        return _format_success("Radar chart created.", path)
    except Exception as exc:
        logger.error("create_radar_chart failed: %s", exc, exc_info=True)
        return f"❌ Error: {str(exc)}"


@mcp.tool()
async def create_pizza_chart(metrics: str = "", styling: str = "", title: str = "", subtitle: str = "", save_as: str = "", fonts: str = "") -> str:
    """Render a pizza chart for player profiling."""
    ok, metrics_data = _load_json(metrics, "metrics")
    if not ok:
        return metrics_data

    params = metrics_data.get("labels") or metrics_data.get("params")
    values = metrics_data.get("values")
    min_range = metrics_data.get("min_range", [0] * len(params))
    max_range = metrics_data.get("max_range", [1] * len(params))

    if not isinstance(params, list) or not isinstance(values, list):
        return "❌ Error: pizza labels and values must be lists."

    if len(params) != len(values):
        return "❌ Error: pizza labels and values length mismatch."

    font_paths = {}
    if fonts.strip():
        ok, font_data = _load_json(fonts, "fonts")
        if ok and isinstance(font_data, dict):
            for key, val in font_data.items():
                font_paths[key] = _load_font_resource(val)

    try:
        pizza = PyPizza(
            params=params,
            background_color=metrics_data.get("background_color", "#1c1c1c"),
            straight_line_color=metrics_data.get("line_color", "#FFFFFF"),
            straight_line_lw=metrics_data.get("line_width", 1),
            last_circle_color=metrics_data.get("last_circle_color", "#FFFFFF"),
            last_circle_lw=metrics_data.get("last_circle_width", 2),
            other_circle_lw=metrics_data.get("other_circle_width", 1),
            min_range=min_range,
            max_range=max_range,
        )
        fig, ax = pizza.make_pizza(
            values,
            figsize=metrics_data.get("figsize", (8, 8)),
            color_blank_space=metrics_data.get("blank_color", "#f5f5f5"),
            slice_colors=metrics_data.get("slice_colors", ["#08F7FE"] * len(params)),
            value_colors=metrics_data.get("value_colors", ["#000000"] * len(params)),
            value_bck_colors=metrics_data.get("value_background_colors", ["#FFFFFF"] * len(params)),
            blank_alpha=metrics_data.get("blank_alpha", 0.4),
        )
        if title.strip():
            fig.text(
                0.5,
                0.98,
                title.strip(),
                ha="center",
                va="center",
                fontsize=metrics_data.get("title_size", 20),
                color=metrics_data.get("title_color", "#FFFFFF"),
                fontfamily=font_paths.get("title"),
            )
        if subtitle.strip():
            fig.text(
                0.5,
                0.94,
                subtitle.strip(),
                ha="center",
                va="center",
                fontsize=metrics_data.get("subtitle_size", 14),
                color=metrics_data.get("subtitle_color", "#FFFFFF"),
                fontfamily=font_paths.get("subtitle"),
            )
        if styling.strip():
            ok, style = _load_json(styling, "styling")
            if ok and isinstance(style, dict):
                for key, value in style.items():
                    try:
                        fig.text(
                            float(value.get("x", 0.5)),
                            float(value.get("y", 0.05)),
                            value.get("text", ""),
                            ha=value.get("ha", "center"),
                            va=value.get("va", "center"),
                            fontsize=float(value.get("fontsize", 10)),
                            color=value.get("color", "#FFFFFF"),
                            fontfamily=font_paths.get(key),
                        )
                    except Exception as exc:
                        logger.warning("Additional styling skipped: %s", exc)
        path = _save_figure(fig, "pizza", save_as, "png")
        return _format_success("Pizza chart created.", path)
    except Exception as exc:
        logger.error("create_pizza_chart failed: %s", exc, exc_info=True)
        return f"❌ Error: {str(exc)}"


@mcp.tool()
async def create_bumpy_chart(data: str = "", title: str = "", subtitle: str = "", save_as: str = "", theme: str = "") -> str:
    """Render a bumpy chart showing ranking across time."""
    ok, dataset = _load_json(data, "bumpy data")
    if not ok:
        return dataset

    x_labels = dataset.get("x_labels", [])
    y_values = dataset.get("y_values", [])
    highlight = dataset.get("highlight", [])

    if not isinstance(x_labels, list) or not isinstance(y_values, list):
        return "❌ Error: x_labels and y_values must be lists."

    if not y_values:
        return "❌ Error: y_values cannot be empty."

    try:
        bumpy = Bumpy(
            height=dataset.get("height", 600),
            width=dataset.get("width", 800),
            title_height=dataset.get("title_height", 50),
            plot_separation=dataset.get("plot_separation", 30),
            rounding_precision=dataset.get("rounding_precision", 0),
        )
        fig, ax = bumpy.plot(
            x_list=x_labels,
            y_lists=y_values,
            highlight_dict=highlight,
            figsize=dataset.get("figsize", (10, 6)),
            colors=dataset.get("colors", ["#f94144", "#277da1"]),
            background_color=dataset.get("background_color", "#0f172a"),
            line_width=dataset.get("line_width", 3),
        )
        if title.strip():
            fig.suptitle(title.strip(), color="#ffffff", fontsize=dataset.get("title_size", 18))
        if subtitle.strip():
            fig.text(0.5, 0.94, subtitle.strip(), ha="center", color="#b5e48c", fontsize=dataset.get("subtitle_size", 12))
        if theme.strip():
            ok, theme_data = _load_json(theme, "theme")
            if ok and isinstance(theme_data, dict):
                fig.patch.set_facecolor(theme_data.get("background", fig.get_facecolor()))
                ax.set_facecolor(theme_data.get("axes_background", ax.get_facecolor()))
        path = _save_figure(fig, "bumpy", save_as, "png")
        return _format_success("Bumpy chart created.", path)
    except Exception as exc:
        logger.error("create_bumpy_chart failed: %s", exc, exc_info=True)
        return f"❌ Error: {str(exc)}"


@mcp.tool()
async def load_statsbomb_data(competition: str = "", season: str = "", match: str = "", limit: str = "") -> str:
    """Load StatsBomb open data (competitions, matches, events)."""
    try:
        sb = Sbopen()
    except Exception as exc:
        logger.error("Sbopen initialization failed: %s", exc)
        return f"❌ Error: Failed to initialize Sbopen ({exc})."

    try:
        comps = sb.competitions()
        result = {"competitions": comps[0].to_dict(orient="records")}
        if competition.strip() and season.strip():
            matches = sb.matches(int(competition.strip()), int(season.strip()))
            result["matches"] = matches[0].to_dict(orient="records")
            if match.strip():
                events = sb.events(int(match.strip()))
                events_df = events[0]
                lim = int(limit.strip()) if limit.strip() else 25
                result["events_sample"] = events_df.head(lim).to_dict(orient="records")
        return _format_json_output(result, "StatsBomb data")
    except Exception as exc:
        logger.error("StatsBomb download failed: %s", exc)
        return f"❌ Error: Unable to load StatsBomb data ({exc})."


@mcp.tool()
async def standardize_coordinates(data: str = "", provider: str = "", target: str = "") -> str:
    """Convert coordinates between provider systems."""
    ok, payload = _load_json(data, "coordinate data")
    if not ok:
        return payload

    if not isinstance(payload, list) or not payload:
        return "❌ Error: coordinate data must be a non-empty JSON array."

    provider_name = provider.strip() or "statsbomb"
    target_name = target.strip() or "statsbomb"

    try:
        standardizer = Standardizer(provider=provider_name, pitch_length=120, pitch_width=80)
        coords = []
        for item in payload:
            x = float(item.get("x", 0))
            y = float(item.get("y", 0))
            coordinates = standardizer.transform(
                x,
                y,
                target_system=target_name,
                pitch_length=float(item.get("pitch_length", 120)),
                pitch_width=float(item.get("pitch_width", 80)),
            )
            coords.append({"original": [x, y], "converted": [coordinates[0], coordinates[1]]})
        return _format_json_output({"provider": provider_name, "target": target_name, "coordinates": coords}, "Standardized coordinates")
    except Exception as exc:
        logger.error("Standardizer failed: %s", exc)
        return f"❌ Error: Coordinate standardization failed ({exc})."


@mcp.tool()
async def animate_frames(frames: str = "", layout: str = "", title: str = "", save_as: str = "", interval: str = "") -> str:
    """Create an animated sequence of frames."""
    ok, frames_data = _load_json(frames, "frames")
    if not ok:
        return frames_data

    if not isinstance(frames_data, list) or not frames_data:
        return "❌ Error: frames must be a non-empty JSON array."

    layout_data = {}
    if layout.strip():
        ok, parsed = _load_json(layout, "layout")
        if not ok:
            return parsed
        layout_data = parsed

    try:
        pitch = _build_pitch(layout_data)
        figsize = layout_data.get("figsize", [12, 8])
        fig, ax = pitch.draw(figsize=tuple(figsize))
        if title.strip():
            fig.suptitle(title.strip(), color=layout_data.get("title_color", "#ffffff"), fontsize=layout_data.get("title_size", 18))

        players_scatter = pitch.scatter([], [], ax=ax, s=200, c="#4cc9f0", zorder=6)
        opponents_scatter = pitch.scatter([], [], ax=ax, s=200, c="#f72585", zorder=6)
        ball_scatter = pitch.scatter([], [], ax=ax, s=120, c="#ffb703", zorder=8)

        def init():
            players_scatter.set_offsets([])
            opponents_scatter.set_offsets([])
            ball_scatter.set_offsets([])
            return players_scatter, opponents_scatter, ball_scatter

        def update(frame):
            players = frame.get("players", [])
            opponents = frame.get("opponents", [])
            ball = frame.get("ball", {})
            if players:
                players_scatter.set_offsets([[float(p.get("x", 0)), float(p.get("y", 0))] for p in players])
            else:
                players_scatter.set_offsets([])
            if opponents:
                opponents_scatter.set_offsets([[float(p.get("x", 0)), float(p.get("y", 0))] for p in opponents])
            else:
                opponents_scatter.set_offsets([])
            if ball:
                ball_scatter.set_offsets([[float(ball.get("x", 0)), float(ball.get("y", 0))]])
            else:
                ball_scatter.set_offsets([])
            return players_scatter, opponents_scatter, ball_scatter

        try:
            interval_ms = float(interval.strip()) if interval.strip() else 400
        except ValueError:
            return f"❌ Error: Invalid interval value: {interval}"

        anim = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=frames_data,
            interval=interval_ms,
            blit=True,
            repeat=True,
        )
        path = _save_animation(anim, "animation", save_as)
        plt.close(fig)
        return _format_success("Animation generated.", path)
    except Exception as exc:
        logger.error("animate_frames failed: %s", exc, exc_info=True)
        return f"❌ Error: {str(exc)}"


@mcp.tool()
async def goal_angle_and_voronoi(data: str = "", layout: str = "", save_as: str = "") -> str:
    """Plot goal angles and voronoi regions for locations."""
    ok, dataset = _load_json(data, "event data")
    if not ok:
        return dataset

    if not isinstance(dataset, list) or not dataset:
        return "❌ Error: event data must be a non-empty JSON array."

    xs = [float(item.get("x", 0)) for item in dataset]
    ys = [float(item.get("y", 0)) for item in dataset]

    layout_data = {}
    if layout.strip():
        ok, parsed = _load_json(layout, "layout")
        if not ok:
            return parsed
        layout_data = parsed

    try:
        pitch = _build_pitch(layout_data)
        figsize = layout_data.get("figsize", [12, 8])
        fig, ax = pitch.draw(figsize=tuple(figsize))
        goal_angles = pitch.goal_angle(xs, ys, goal='right')
        pitch.scatter(xs, ys, ax=ax, c=goal_angles, cmap="coolwarm", s=150, edgecolors="#000000", zorder=6)
        if len(xs) > 2:
            pitch.voronoi(xs, ys, ax=ax, colors="winter")
        path = _save_figure(fig, "goal_angle_voronoi", save_as, "png")
        return _format_success("Goal angle and Voronoi visual generated.", path)
    except Exception as exc:
        logger.error("goal_angle_and_voronoi failed: %s", exc, exc_info=True)
        return f"❌ Error: {str(exc)}"


@mcp.tool()
async def create_grid_layout(layouts: str = "", config: str = "", save_as: str = "") -> str:
    """Compose a multi-pitch grid layout."""
    ok, layout_data = _load_json(layouts, "layouts")
    if not ok:
        return layout_data

    if not isinstance(layout_data, list) or not layout_data:
        return "❌ Error: layouts must be a non-empty JSON array."

    config_data = {}
    if config.strip():
        ok, parsed = _load_json(config, "config")
        if not ok:
            return parsed
        config_data = parsed

    try:
        fig, axs = plt.subplots(
            nrows=config_data.get("rows", 2),
            ncols=config_data.get("cols", 2),
            figsize=tuple(config_data.get("figsize", [14, 10])),
        )
        axs = np.array(axs).reshape(-1)
        for index, item in enumerate(layout_data):
            if index >= len(axs):
                break
            pitch = _build_pitch(item)
            pitch.draw(ax=axs[index])
            primitives = item.get("primitives", {})
            _apply_primitives(pitch, axs[index], primitives)
            if item.get("title"):
                axs[index].set_title(item.get("title"), color=item.get("title_color", "#ffffff"))
        fig.tight_layout()
        path = _save_figure(fig, "grid_layout", save_as, "png")
        return _format_success("Grid layout rendered.", path)
    except Exception as exc:
        logger.error("create_grid_layout failed: %s", exc, exc_info=True)
        return f"❌ Error: {str(exc)}"


if __name__ == "__main__":
    logger.info("Starting MPLSoccer Viz MCP server...")
    try:
        mcp.run(transport="stdio")
    except Exception as exc:
        logger.error("Server error: %s", exc, exc_info=True)
        sys.exit(1)

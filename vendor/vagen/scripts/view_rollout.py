#!/usr/bin/env python3
"""
Render a .jsonl rollout file as an interactive HTML chat viewer.

Usage:
    python scripts/view_rollout.py <path_to.jsonl> [--line N] [--out chat.html]

Each line in the jsonl is one episode.  --line selects which (1-indexed, default 1).
Opens the HTML in the default browser automatically.
"""

import argparse
import html
import json
import re
import webbrowser
from pathlib import Path


def parse_conversation(raw: str):
    """Split a chatml string into list of (role, content) tuples."""
    turns = []
    for block in raw.split("<|im_start|>"):
        block = block.strip()
        if not block:
            continue
        block = block.replace("<|im_end|>", "").strip()
        newline = block.find("\n")
        if newline == -1:
            continue
        role = block[:newline].strip()
        content = block[newline + 1:].strip()
        turns.append((role, content))
    return turns


def render_html(turns, meta, title="Rollout Viewer"):
    escaped = []
    for role, content in turns:
        escaped.append((role, html.escape(content)))

    turn_divs = []
    for i, (role, content) in enumerate(escaped):
        css_class = role
        label = role.upper()
        if role == "assistant":
            # try to extract action from JSON
            try:
                raw = turns[i][1]
                obj = json.loads(raw)
                action = obj.get("action", "")
                if action:
                    label += f' &mdash; <span class="action">{html.escape(action)}</span>'
            except Exception:
                pass
        turn_divs.append(
            f'<div class="turn {css_class}" id="turn-{i}">'
            f'<div class="role">{label}</div>'
            f'<pre class="content">{content}</pre>'
            f'</div>'
        )

    meta_html = " &nbsp;|&nbsp; ".join(
        f"<b>{html.escape(k)}</b>: {html.escape(str(v))}" for k, v in meta.items()
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #1a1a2e; color: #e0e0e0; padding: 0; }}
.header {{ position: sticky; top: 0; z-index: 10; background: #16213e;
           padding: 12px 24px; border-bottom: 1px solid #0f3460;
           display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
.header h2 {{ color: #e94560; font-size: 1.1em; }}
.meta {{ font-size: 0.85em; color: #a0a0c0; }}
.nav {{ display: flex; gap: 6px; align-items: center; }}
.nav button {{ background: #0f3460; color: #e0e0e0; border: 1px solid #e94560;
              border-radius: 4px; padding: 4px 10px; cursor: pointer; font-size: 0.85em; }}
.nav button:hover {{ background: #e94560; }}
.nav select {{ background: #0f3460; color: #e0e0e0; border: 1px solid #555;
              border-radius: 4px; padding: 4px 8px; font-size: 0.85em; }}
.container {{ max-width: 900px; margin: 0 auto; padding: 16px; }}
.turn {{ margin: 12px 0; border-radius: 8px; padding: 12px 16px; }}
.turn.system {{ background: #16213e; border-left: 3px solid #0f3460; }}
.turn.user {{ background: #1a1a3e; border-left: 3px solid #4a90d9; }}
.turn.assistant {{ background: #1e2a1e; border-left: 3px solid #4caf50; }}
.role {{ font-weight: 700; font-size: 0.8em; text-transform: uppercase; margin-bottom: 6px;
         letter-spacing: 0.05em; }}
.system .role {{ color: #0f3460; }}
.user .role {{ color: #4a90d9; }}
.assistant .role {{ color: #4caf50; }}
.action {{ color: #e94560; font-family: monospace; font-weight: 400; text-transform: none; letter-spacing: 0; }}
.content {{ white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em;
           line-height: 1.5; font-family: "SF Mono", "Fira Code", monospace; }}
.turn.collapsed .content {{ display: none; }}
.turn.collapsed {{ opacity: 0.6; cursor: pointer; }}
</style></head><body>
<div class="header">
  <div><h2>{html.escape(title)}</h2><div class="meta">{meta_html}</div></div>
  <div class="nav">
    <button onclick="toggleAll(true)">Collapse All</button>
    <button onclick="toggleAll(false)">Expand All</button>
    <select onchange="jumpTo(this.value)">{
      ''.join(f'<option value="turn-{i}">{turns[i][0]} #{i}</option>' for i in range(len(turns)))
    }</select>
  </div>
</div>
<div class="container">{''.join(turn_divs)}</div>
<script>
document.querySelectorAll('.turn').forEach(el => {{
  el.querySelector('.role').addEventListener('click', () => el.classList.toggle('collapsed'));
}});
// collapse system by default
document.querySelectorAll('.turn.system').forEach(el => el.classList.add('collapsed'));
function toggleAll(collapse) {{
  document.querySelectorAll('.turn').forEach(el => {{
    if (collapse) el.classList.add('collapsed'); else el.classList.remove('collapsed');
  }});
}}
function jumpTo(id) {{ document.getElementById(id)?.scrollIntoView({{behavior:'smooth',block:'start'}}); }}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", help="Path to .jsonl rollout file")
    ap.add_argument("--line", type=int, default=1, help="Line number (1-indexed)")
    ap.add_argument("--out", default=None, help="Output HTML path (default: auto)")
    args = ap.parse_args()

    path = Path(args.jsonl)
    with open(path) as f:
        for i, raw_line in enumerate(f, 1):
            if i == args.line:
                data = json.loads(raw_line)
                break
        else:
            raise ValueError(f"File has fewer than {args.line} lines")

    full_text = data.get("input", "") + data.get("output", "")
    turns = parse_conversation(full_text)

    meta = {}
    for k in ("score", "reward", "traj_success", "step"):
        if k in data:
            meta[k] = data[k]

    title = f"{path.stem} line {args.line}"
    page = render_html(turns, meta, title=title)

    out_path = Path(args.out) if args.out else path.with_suffix(f".line{args.line}.html")
    out_path.write_text(page)
    print(f"Written to {out_path}")
    webbrowser.open(f"file://{out_path.resolve()}")


if __name__ == "__main__":
    main()

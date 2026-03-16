"""Minimal frontend shell for the phase-0 web surface."""

from __future__ import annotations

from bracket_sim import __version__


def build_frontend_shell() -> str:
    """Return the static HTML shell served by the local FastAPI app."""

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Bracket Pool Simulator</title>
    <style>
      :root {{
        --ink: #13233a;
        --ink-soft: #47617d;
        --paper: #f5f1e8;
        --panel: rgba(255, 255, 255, 0.88);
        --accent: #d95d39;
        --accent-soft: #f1b775;
        --line: rgba(19, 35, 58, 0.12);
        --shadow: 0 18px 50px rgba(19, 35, 58, 0.12);
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        min-height: 100vh;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(217, 93, 57, 0.2), transparent 32%),
          radial-gradient(circle at top right, rgba(241, 183, 117, 0.35), transparent 28%),
          linear-gradient(180deg, #fcfaf5 0%, var(--paper) 100%);
      }}

      .page {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 32px 20px 48px;
      }}

      .hero {{
        padding: 28px;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(255, 244, 227, 0.92));
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
      }}

      .eyebrow {{
        display: inline-flex;
        gap: 10px;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(19, 35, 58, 0.06);
        font-family: "Menlo", "SFMono-Regular", Consolas, monospace;
        font-size: 13px;
      }}

      h1 {{
        margin: 18px 0 10px;
        font-size: clamp(2.4rem, 5vw, 4.5rem);
        line-height: 0.95;
        letter-spacing: -0.05em;
      }}

      .lede {{
        max-width: 54rem;
        margin: 0;
        color: var(--ink-soft);
        font-size: 1.05rem;
        line-height: 1.6;
      }}

      .status-row,
      .grid {{
        display: grid;
        gap: 16px;
      }}

      .status-row {{
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        margin-top: 20px;
      }}

      .grid {{
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        margin-top: 20px;
      }}

      .panel {{
        padding: 22px;
        border-radius: 24px;
        background: var(--panel);
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
      }}

      .panel h2 {{
        margin: 0 0 14px;
        font-size: 1.2rem;
      }}

      .metric {{
        font-family: "Menlo", "SFMono-Regular", Consolas, monospace;
        font-size: 0.92rem;
      }}

      ul {{
        margin: 0;
        padding: 0;
        list-style: none;
      }}

      li + li {{
        margin-top: 14px;
      }}

      .item {{
        display: grid;
        gap: 6px;
        padding-top: 14px;
        border-top: 1px solid var(--line);
      }}

      .item:first-child {{
        padding-top: 0;
        border-top: 0;
      }}

      .item-title {{
        display: flex;
        gap: 10px;
        align-items: center;
        justify-content: space-between;
        font-weight: 700;
      }}

      .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-family: "Menlo", "SFMono-Regular", Consolas, monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}

      .badge-live {{
        background: rgba(31, 122, 90, 0.12);
        color: #1f7a5a;
      }}

      .badge-planned {{
        background: rgba(217, 93, 57, 0.12);
        color: #a3482d;
      }}

      .muted {{
        color: var(--ink-soft);
      }}

      .code {{
        font-family: "Menlo", "SFMono-Regular", Consolas, monospace;
        font-size: 0.84rem;
      }}

      @media (max-width: 720px) {{
        .page {{
          padding-inline: 14px;
        }}

        .hero,
        .panel {{
          border-radius: 22px;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <div class="eyebrow">
          <span>Local Web/API Surface</span>
          <span id="version">v{__version__}</span>
        </div>
        <h1>Bracket tools, staged for the browser.</h1>
        <p class="lede">
          Phase 0 wires the existing Python engine into a small web surface, exposes shared
          typed product contracts, and documents how reusable analysis and optimization caches
          will be keyed.
        </p>
        <div class="status-row">
          <article class="panel">
            <h2>API Health</h2>
            <div id="health-status" class="metric">Loading...</div>
          </article>
          <article class="panel">
            <h2>Roadmap Phase</h2>
            <div id="phase-name" class="metric">Loading...</div>
          </article>
          <article class="panel">
            <h2>Cache Kinds</h2>
            <div id="cache-kinds" class="metric">Loading...</div>
          </article>
        </div>
      </section>

      <section class="grid">
        <article class="panel">
          <h2>Scoring Systems</h2>
          <ul id="scoring-systems"></ul>
        </article>
        <article class="panel">
          <h2>Completion Modes</h2>
          <ul id="completion-modes"></ul>
        </article>
        <article class="panel">
          <h2>Cache Strategy</h2>
          <p id="dataset-rule" class="muted">Loading...</p>
          <p id="cache-rule" class="muted">Loading...</p>
          <div class="code" id="cache-preview"></div>
        </article>
      </section>
    </main>

    <script>
      function badge(implemented) {{
        return implemented
          ? '<span class="badge badge-live">live</span>'
          : '<span class="badge badge-planned">planned</span>';
      }}

      function renderScoringSystems(items) {{
        return items.map((item) => `
          <li class="item">
            <div class="item-title">
              <span>${{item.label}}</span>
              ${{badge(item.implemented)}}
            </div>
            <div class="muted">${{item.description}}</div>
            <div class="code">${{item.key}} -> ${{item.round_values.join(", ")}}</div>
          </li>
        `).join("");
      }}

      function renderCompletionModes(items) {{
        return items.map((item) => `
          <li class="item">
            <div class="item-title">
              <span>${{item.label}}</span>
              ${{badge(item.implemented)}}
            </div>
            <div class="muted">${{item.description}}</div>
            <div class="code">${{item.mode}}</div>
          </li>
        `).join("");
      }}

      async function boot() {{
        const [healthResponse, foundationResponse] = await Promise.all([
          fetch("/api/health"),
          fetch("/api/foundation"),
        ]);

        const health = await healthResponse.json();
        const foundation = await foundationResponse.json();

        document.getElementById("health-status").textContent =
          `${{health.status}} / v${{health.version}}`;
        document.getElementById("phase-name").textContent = foundation.roadmap_phase;
        document.getElementById("cache-kinds").textContent =
          foundation.cache_policy.artifact_kinds.join(", ");
        document.getElementById("dataset-rule").textContent =
          foundation.cache_policy.dataset_hash_rule;
        document.getElementById("cache-rule").textContent = foundation.cache_policy.cache_key_rule;
        document.getElementById("scoring-systems").innerHTML =
          renderScoringSystems(foundation.scoring_systems);
        document.getElementById("completion-modes").innerHTML =
          renderCompletionModes(foundation.completion_modes);
        document.getElementById("cache-preview").textContent =
          "example: analysis-<sha256-prefix> from dataset hash + pool settings + completion mode";
      }}

      boot().catch((error) => {{
        document.getElementById("health-status").textContent = "error";
        document.getElementById("phase-name").textContent = "bootstrap failed";
        document.getElementById("dataset-rule").textContent = String(error);
      }});
    </script>
  </body>
</html>
"""

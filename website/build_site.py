#!/usr/bin/env python3
"""Render BLOG.md into a dependency-free HTML page for GitHub Pages."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLOG = ROOT / "BLOG.md"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "index.html"
REPO_BLOB = "https://github.com/lin826/Online-SFT-Demo/blob/main/"
REPO_TREE = "https://github.com/lin826/Online-SFT-Demo/tree/main/"

try:
    import markdown
except ImportError as exc:  # pragma: no cover - install guidance for contributors
    raise SystemExit(
        "The 'markdown' package is required. Install with: python3 -m pip install markdown"
    ) from exc


REPO_PATH_RE = re.compile(
    r'href="((?:online_sdft|docs|figures)/[^"]+|online_sdft_bandit_demo\.ipynb)"'
)
MERMAID_RE = re.compile(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>',
    re.DOTALL,
)


def rewrite_repo_links(content_html: str) -> str:
    def replace(match: re.Match[str]) -> str:
        path = match.group(1)
        if path.endswith("/"):
            return f'href="{REPO_TREE}{path}"'
        return f'href="{REPO_BLOB}{path}"'

    return REPO_PATH_RE.sub(replace, content_html)


def promote_mermaid_blocks(content_html: str) -> str:
    def replace(match: re.Match[str]) -> str:
        source = html.unescape(match.group(1)).strip()
        return f'<pre class="mermaid">{html.escape(source)}</pre>'

    return MERMAID_RE.sub(replace, content_html)


def render_markdown(source: str) -> str:
    converter = markdown.Markdown(
        extensions=[
            "fenced_code",
            "tables",
            "sane_lists",
            "smarty",
        ]
    )
    content_html = converter.convert(source)
    content_html = promote_mermaid_blocks(content_html)
    content_html = rewrite_repo_links(content_html)
    return content_html


def page_shell(body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f7f4ee">
  <meta name="description" content="Learning from the notification you did—or did not—send: online personalization under action-dependent feedback.">
  <link rel="canonical" href="https://lin826.github.io/Online-SFT-Demo/">
  <meta property="og:type" content="article">
  <meta property="og:title" content="Learning From the Notification You Did—or Didn’t—Send">
  <meta property="og:description" content="An online learning demo for continual personalization under action-dependent feedback.">
  <meta property="og:url" content="https://lin826.github.io/Online-SFT-Demo/">
  <meta property="og:image" content="https://lin826.github.io/Online-SFT-Demo/assets/og.png">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="Learning From the Notification You Did—or Didn’t—Send">
  <meta name="twitter:description" content="An online learning demo for continual personalization under action-dependent feedback.">
  <meta name="twitter:image" content="https://lin826.github.io/Online-SFT-Demo/assets/og.png">
  <title>Learning From the Notification You Did—or Didn’t—Send</title>
  <link rel="stylesheet" href="styles.css">
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, theme: "neutral", securityLevel: "strict" }});
  </script>
</head>
<body>
  <header class="site-header">
    <div class="site-header-inner">
      <a class="site-title" href="./">Online-SDFT</a>
      <nav aria-label="Repository links">
        <a href="https://colab.research.google.com/github/lin826/Online-SFT-Demo/blob/main/online_sdft_bandit_demo.ipynb">Colab</a>
        <a href="https://github.com/lin826/Online-SFT-Demo">Code</a>
      </nav>
    </div>
  </header>
  <main class="article">
{body_html}
  </main>
  <footer class="site-footer">
    <p>Source: <a href="https://github.com/lin826/Online-SFT-Demo/blob/main/BLOG.md">BLOG.md</a></p>
  </footer>
</body>
</html>
"""


def build(blog_path: Path, output_path: Path) -> None:
    source = blog_path.read_text(encoding="utf-8")
    body_html = render_markdown(source)
    output_path.write_text(page_shell(body_html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blog", type=Path, default=DEFAULT_BLOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.blog.resolve(), args.output.resolve())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render BLOG.md into a static research-article page for GitHub Pages.

Math is protected from the Markdown converter and rendered client-side by
KaTeX. Standalone images followed by an italic line become wide figures with
captions.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLOG = ROOT / "BLOG.md"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "index.html"
REPO = "https://github.com/lin826/Online-SFT-Demo"
REPO_BLOB = f"{REPO}/blob/main/"
REPO_TREE = f"{REPO}/tree/main/"
COLAB = (
    "https://colab.research.google.com/github/lin826/Online-SFT-Demo/blob/main/"
    "online_sdft_bandit_demo.ipynb"
)

try:
    import markdown
except ImportError as exc:  # pragma: no cover - install guidance for contributors
    raise SystemExit(
        "The 'markdown' package is required. Install with: python3 -m pip install markdown"
    ) from exc


CODE_SPAN_RE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]+`", re.DOTALL)
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)|\$(?!\s)([^$\n]+?)(?<!\s)\$")
REPO_PATH_RE = re.compile(
    r'href="((?:online_sdft|docs|figures)/[^"]+|online_sdft_bandit_demo\.ipynb)"'
)
MERMAID_RE = re.compile(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>', re.DOTALL
)
FIGURE_WITH_CAPTION_RE = re.compile(
    r"<p>(<img\b[^>]*>)</p>\s*<p><em>(.*?)</em></p>", re.DOTALL
)
BARE_FIGURE_RE = re.compile(r"<p>(<img\b[^>]*>)</p>")
H1_RE = re.compile(r"<h1>(.*?)</h1>", re.DOTALL)
HEADING_ID_RE = re.compile(r"<h([23])>(.*?)</h\1>", re.DOTALL)


def protected_spans(source: str) -> list[tuple[int, int]]:
    return [match.span() for match in CODE_SPAN_RE.finditer(source)]


def outside_code(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return not any(begin <= start and end <= finish for begin, finish in spans)


def extract_math(source: str) -> tuple[str, dict[str, tuple[str, bool]]]:
    """Replace math with inert tokens so Markdown cannot mangle it."""
    store: dict[str, tuple[str, bool]] = {}
    counter = 0

    def substitute(pattern: re.Pattern[str], text: str, display: bool) -> str:
        nonlocal counter
        spans = protected_spans(text)
        pieces: list[str] = []
        cursor = 0
        for match in pattern.finditer(text):
            if not outside_code(spans, *match.span()):
                continue
            body = next(group for group in match.groups() if group is not None)
            token = f"MATHTOKEN{counter}ZZ"
            counter += 1
            store[token] = (body.strip(), display)
            pieces.append(text[cursor : match.start()])
            pieces.append(token)
            cursor = match.end()
        pieces.append(text[cursor:])
        return "".join(pieces)

    source = substitute(DISPLAY_MATH_RE, source, True)
    source = substitute(INLINE_MATH_RE, source, False)
    return source, store


def restore_math(content_html: str, store: dict[str, tuple[str, bool]]) -> str:
    for token, (body, display) in store.items():
        escaped = html.escape(body, quote=False)
        if display:
            block = f'<div class="math-display">\\[{escaped}\\]</div>'
            content_html = content_html.replace(f"<p>{token}</p>", block)
            content_html = content_html.replace(token, block)
        else:
            content_html = content_html.replace(token, f"\\({escaped}\\)")
    return content_html


def rewrite_repo_links(content_html: str) -> str:
    def replace(match: re.Match[str]) -> str:
        path = match.group(1)
        base = REPO_TREE if path.endswith("/") else REPO_BLOB
        return f'href="{base}{path}"'

    return REPO_PATH_RE.sub(replace, content_html)


def promote_mermaid_blocks(content_html: str) -> str:
    def replace(match: re.Match[str]) -> str:
        source = html.unescape(match.group(1)).strip()
        return (
            '<figure class="figure figure-wide">'
            f'<pre class="mermaid">{html.escape(source)}</pre>'
            "</figure>"
        )

    return MERMAID_RE.sub(replace, content_html)


def wrap_figures(content_html: str) -> str:
    content_html = FIGURE_WITH_CAPTION_RE.sub(
        lambda m: (
            '<figure class="figure figure-wide">'
            f"{m.group(1)}<figcaption>{m.group(2).strip()}</figcaption></figure>"
        ),
        content_html,
    )
    return BARE_FIGURE_RE.sub(
        lambda m: f'<figure class="figure figure-wide">{m.group(1)}</figure>',
        content_html,
    )


def slugify(text: str) -> str:
    plain = re.sub(r"<[^>]+>", "", text)
    plain = html.unescape(plain).lower()
    plain = re.sub(r"[^a-z0-9]+", "-", plain).strip("-")
    return plain or "section"


def add_heading_ids(content_html: str) -> tuple[str, list[tuple[str, str]]]:
    outline: list[tuple[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        level, inner = match.group(1), match.group(2)
        anchor = slugify(inner)
        if level == "2":
            outline.append((anchor, re.sub(r"<[^>]+>", "", inner).strip()))
        return f'<h{level} id="{anchor}">{inner}</h{level}>'

    return HEADING_ID_RE.sub(replace, content_html), outline


def split_hero(content_html: str) -> tuple[str, str]:
    match = H1_RE.search(content_html)
    if not match:
        return "Online-SDFT", content_html
    title = match.group(1).strip()
    return title, content_html[: match.start()] + content_html[match.end() :]


def render_markdown(source: str) -> tuple[str, str, list[tuple[str, str]]]:
    source, math_store = extract_math(source)
    converter = markdown.Markdown(
        extensions=["fenced_code", "tables", "sane_lists", "smarty"]
    )
    content_html = converter.convert(source)
    content_html = promote_mermaid_blocks(content_html)
    content_html = wrap_figures(content_html)
    content_html = rewrite_repo_links(content_html)
    content_html = restore_math(content_html, math_store)
    title, content_html = split_hero(content_html)
    content_html, outline = add_heading_ids(content_html)
    return title, content_html.strip(), outline


def render_outline(outline: list[tuple[str, str]]) -> str:
    items = "\n".join(
        f'        <li><a href="#{anchor}">{html.escape(label)}</a></li>'
        for anchor, label in outline
    )
    return f"""      <nav class="contents" aria-label="Contents">
        <p>Contents</p>
        <ol>
{items}
        </ol>
      </nav>"""


def page_shell(title: str, body_html: str, outline_html: str) -> str:
    plain_title = re.sub(r"<[^>]+>", "", title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#ffffff">
  <meta name="description" content="Why continual personalization should be treated as online learning, and what changes when only the executed action produces feedback.">
  <link rel="canonical" href="https://lin826.github.io/Online-SFT-Demo/">
  <meta property="og:type" content="article">
  <meta property="og:title" content="{html.escape(plain_title)}">
  <meta property="og:description" content="Online continual personalization under action-dependent feedback.">
  <meta property="og:url" content="https://lin826.github.io/Online-SFT-Demo/">
  <meta property="og:image" content="https://lin826.github.io/Online-SFT-Demo/assets/og.png">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{html.escape(plain_title)}">
  <meta name="twitter:description" content="Online continual personalization under action-dependent feedback.">
  <meta name="twitter:image" content="https://lin826.github.io/Online-SFT-Demo/assets/og.png">
  <title>{html.escape(plain_title)}</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">
  <link rel="stylesheet" href="styles.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"
    onload="renderMathInElement(document.body, {{delimiters: [
      {{left: '\\\\[', right: '\\\\]', display: true}},
      {{left: '$$', right: '$$', display: true}},
      {{left: '\\\\(', right: '\\\\)', display: false}}
    ], throwOnError: false}});"></script>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, theme: "neutral", securityLevel: "strict" }});
  </script>
</head>
<body>
  <header class="masthead">
    <div class="masthead-inner">
      <a class="masthead-title" href="./">Online-SDFT</a>
      <nav aria-label="Project links">
        <a href="{COLAB}">Colab</a>
        <a href="{REPO}">Code</a>
      </nav>
    </div>
  </header>

  <main>
    <header class="hero">
      <h1>{title}</h1>
      <p class="hero-note">An online-learning study of continual personalization under action-dependent feedback</p>
      <div class="hero-links">
        <a href="{REPO}">Code</a>
        <a href="{COLAB}">Colab notebook</a>
        <a href="{REPO_BLOB}BLOG.md">Blog source</a>
        <a href="{REPO_BLOB}docs/evaluation.md">Evaluation details</a>
      </div>
    </header>

{outline_html}

    <article class="article">
{body_html}
    </article>
  </main>

  <footer class="site-footer">
    <p>Rendered from <a href="{REPO_BLOB}BLOG.md">BLOG.md</a>. Results are preliminary simulator measurements over three paired seeds.</p>
  </footer>
</body>
</html>
"""


def build(blog_path: Path, output_path: Path) -> None:
    source = blog_path.read_text(encoding="utf-8")
    title, body_html, outline = render_markdown(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        page_shell(title, body_html, render_outline(outline)), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blog", type=Path, default=DEFAULT_BLOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.blog.resolve(), args.output.resolve())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

"""
Assembles frontend/standalone_demo.html: one self-contained file with the
CSS, rendering JS, and trimmed dataset all inlined, and no fetch() calls
-- suitable for publishing as a live preview (a CSP-locked published page
can't fetch sibling local files). frontend/index.html remains the real,
fetch-based version for the actual project.

Body markup is extracted straight from index.html rather than duplicated
by hand, so the two can't drift out of sync with each other.
"""
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(ROOT, "frontend")


def extract_body(index_html: str) -> str:
    match = re.search(r"<body>(.*?)<script src=\"https://cdnjs", index_html, re.DOTALL)
    if not match:
        raise ValueError("Could not find body content in index.html")
    return match.group(1).strip()


def main():
    with open(os.path.join(FRONTEND, "index.html"), encoding="utf-8") as f:
        index_html = f.read()
    with open(os.path.join(FRONTEND, "style.css"), encoding="utf-8") as f:
        css = f.read()
    with open(os.path.join(FRONTEND, "app.js"), encoding="utf-8") as f:
        js = f.read()
    with open(os.path.join(FRONTEND, "embedded_data.json"), encoding="utf-8") as f:
        data_json = f.read()

    body = extract_body(index_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>SouqPulse — Fair-Value &amp; Trust Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
{css}
</style>
</head>
<body>
{body}

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const EMBEDDED_DATA = {data_json};
</script>
<script>
{js}
</script>
<script>
  SouqPulse.init(EMBEDDED_DATA);
</script>
</body>
</html>
"""

    out_path = os.path.join(FRONTEND, "standalone_demo.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB)")

    # GitHub Pages serves the repo's /docs folder; publish the same file there.
    pages_dir = os.path.join(ROOT, "docs")
    os.makedirs(pages_dir, exist_ok=True)
    with open(os.path.join(pages_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {os.path.join(pages_dir, 'index.html')} (GitHub Pages)")


if __name__ == "__main__":
    main()

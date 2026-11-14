"""Report CJK characters left in the English tree. Usage: python3 tools/check_cjk.py <paths...>

Allowed leftovers: link text/URLs of the author's Chinese blog posts and the language-switch labels.
"""
import json
import re
import sys
from pathlib import Path

CJK = re.compile(r"[　-〿㐀-鿿＀-￯]")
ALLOW = re.compile(r"https?://\S+|🌐 中文|繁體中文|\"中文\"")


def scan_text(label: str, text: str):
    for i, line in enumerate(text.splitlines(), 1):
        if CJK.search(ALLOW.sub("", line)):
            print(f"{label}:{i}: {line.strip()[:100]}")


def scan(path: Path):
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        for n, c in enumerate(nb["cells"]):
            scan_text(f"{path}#cell{n}", "".join(c["source"]))
            for o in c.get("outputs", []):
                t = o.get("text") or o.get("data", {}).get("text/plain") or []
                scan_text(f"{path}#cell{n}/out", "".join(t))
                h = o.get("data", {}).get("text/html")
                if h:
                    scan_text(f"{path}#cell{n}/html", "".join(h))
    elif path.suffix in {".py", ".qmd", ".md", ".yml", ".html"}:
        scan_text(str(path), path.read_text(encoding="utf-8"))


for arg in sys.argv[1:]:
    p = Path(arg)
    for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
        if f.is_file() and "_site" not in f.parts and ".quarto" not in f.parts:
            scan(f)

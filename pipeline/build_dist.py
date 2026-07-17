"""Assemble Nazar/dist, the only folder that should ever be deployed.

The app fetches ../../artifacts at dev time, which means the dev server has to
sit at the repo root and see pipeline/, runs/ and the datasets. None of that
belongs on a public site. This copies the page and only the files it actually
needs into dist/, rewrites the artifact paths, and refuses to include anything
from _stale.

    python build_dist.py

Then deploy dist/ and nothing else.
"""

import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
APP = ROOT / "web" / "app"
ART = ROOT / "artifacts"
DIST = ROOT / "dist"

ANALYTICS = (
    '<script defer>\n'
    '(function(){\n'
    "  var E='https://safeer-analytics.safeer-ali-mirani.workers.dev/collect';\n"
    "  function send(t,l){try{var b=JSON.stringify({type:t,label:l||null,path:location.pathname,referrer:document.referrer||null});\n"
    "    if(navigator.sendBeacon)navigator.sendBeacon(E,new Blob([b],{type:'application/json'}));\n"
    "    else fetch(E,{method:'POST',body:b,headers:{'Content-Type':'application/json'},keepalive:true});}catch(_){ }}\n"
    "  send('pageview');\n"
    "  document.addEventListener('click',function(e){\n"
    "    var a=e.target.closest?e.target.closest('a'):null; if(!a)return;\n"
    "    var h=a.getAttribute('href')||'';\n"
    "    if(a.hasAttribute('download')||/\\.pdf(\\?|$)/i.test(h))return send('cv_download','CV');\n"
    "    if(h.indexOf('mailto:')===0)return send('contact_click','email');\n"
    "    if(/github\\.com|linkedin\\.com|orcid\\.org|scholar\\.google/.test(h))return send('source_click',(h.match(/https?:\\/\\/([^\\/]+)/)||[])[1]||'link');\n"
    "  },true);\n"
    '})();\n'
    '</script>\n'
)


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "artifacts" / "queries").mkdir(parents=True)
    (DIST / "kernels").mkdir(parents=True)

    for f in ["index.html", "app.js", "live.js", "style.css"]:
        shutil.copy2(APP / f, DIST / f)
    shutil.copytree(APP / "data", DIST / "data")

    shutil.copy2(ROOT / "web" / "kernels" / "knn.wgsl", DIST / "kernels" / "knn.wgsl")

    # only the shipped banks. f32 is a dev convenience for the parity harness and
    # doubles the payload, so it stays out. nothing from _stale goes anywhere.
    kept = 0
    for p in sorted(ART.glob("*.bin")) + sorted(ART.glob("sidecar_*.json")):
        if "_stale" in str(p) or "_c10" in p.name or ".f32." in p.name:
            continue
        shutil.copy2(p, DIST / "artifacts" / p.name)
        kept += 1
    for p in sorted((ART / "queries").iterdir()):
        if p.is_file():
            shutil.copy2(p, DIST / "artifacts" / "queries" / p.name)

    # the page asks for ../../artifacts from web/app. in dist it sits alongside.
    for f in ["app.js", "live.js", "index.html"]:
        t = (DIST / f)
        s = t.read_text(encoding="utf-8")
        n = s.count("../../artifacts")
        s = s.replace("../../artifacts", "artifacts")
        t.write_text(s, encoding="utf-8", newline="")
        if n:
            print(f"{f}: rewrote {n} artifact paths")

    html = (DIST / "index.html").read_text(encoding="utf-8")
    if "safeer-analytics" not in html:
        html = html.replace("</body>", ANALYTICS + "</body>", 1)
        (DIST / "index.html").write_text(html, encoding="utf-8", newline="")
        print("index.html: analytics snippet added")

    # no source, no datasets, nothing stale. markdown that documents a data file
    # is fine and worth publishing, so it is allowed under data/.
    bad = [p for p in DIST.rglob("*") if p.is_file() and
           (p.suffix in {".py", ".ipynb"} or "_stale" in str(p) or "_c10" in p.name
            or (p.suffix == ".md" and p.parent.name != "data"))]
    if bad:
        raise SystemExit(f"dist contains files that must not ship: {[str(b) for b in bad[:5]]}")

    files = [p for p in DIST.rglob("*") if p.is_file()]
    mb = sum(p.stat().st_size for p in files) / 1e6
    big = sorted(files, key=lambda p: -p.stat().st_size)[:3]
    print(f"\ndist: {len(files)} files, {mb:.1f} MB")
    for p in big:
        print(f"  {p.stat().st_size/1e6:6.1f} MB  {p.relative_to(DIST)}")
    over = [p for p in files if p.stat().st_size > 25 * 1024 * 1024]
    if over:
        raise SystemExit(f"over the Cloudflare Pages 25 MiB per file cap: {[p.name for p in over]}")
    print("\nevery file is under the 25 MiB Pages cap. deploy dist and nothing else:")
    print('  wrangler pages deploy dist --project-name=nazar --branch=production')


if __name__ == "__main__":
    main()

"""Build the public GitHub Pages demo using an explicit set of verified artifacts."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
BENCHMARK = ROOT / "benchmarks" / "E1-bluequbit"
OUT = ROOT / "build" / "pages"


def main() -> None:
    """Check provenance and copy only website files and public benchmark artifacts."""
    result = json.loads((BENCHMARK / "result.json").read_text())
    if result["config"]["experiment"] != "E1" or len(result["remote_jobs"]) != 458:
        raise ValueError("Demo requires the verified E1 benchmark")
    if any(job["status"] != "complete" for job in result["remote_jobs"]):
        raise ValueError("Benchmark contains incomplete cloud jobs")
    for name, digest in json.loads((BENCHMARK / "checksums.json").read_text()).items():
        if hashlib.sha256((BENCHMARK / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Benchmark checksum mismatch: {name}")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)
    (OUT / "reports").mkdir()
    for name in ("index.html", "styles.css", "app.js", "favicon.svg"):
        shutil.copy2(SITE / name, OUT / name)
    for name in ("result.json", "cloud-verification.json"):
        shutil.copy2(BENCHMARK / name, OUT / "data" / name)
    for name in ("report.pdf", "report.html"):
        shutil.copy2(BENCHMARK / name, OUT / "reports" / name)
    (OUT / ".nojekyll").touch()
    print(f"Built public demo: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

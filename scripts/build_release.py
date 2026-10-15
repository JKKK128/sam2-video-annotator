#!/usr/bin/env python3
"""Build a clean, self-checked source release without changing source files."""

from __future__ import annotations

import argparse
import compileall
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PACKAGE = "sam2_video_annotator_tool"
STAGE = DIST / PACKAGE
ARCHIVE_BASE = DIST / PACKAGE
ALLOWED = ("README.md", "LICENSE", "pyproject.toml", "environment.yml", "pytest.ini", "configs", "docs", "scripts", "src", "tests", "examples")
FORBIDDEN_PARTS = {"data", "outputs", "snapshots", "archive", "checkpoints", ".git", ".venv", "__pycache__", ".pytest_cache__", ".pytest_cache"}
HELP_SCRIPTS = (
    "register_object.py", "create_tracking_segment.py", "track_registered_object.py",
    "detect_anomalies.py", "build_anomaly_intervals.py", "update_review_status.py",
    "create_handoff.py", "accept_handoff.py", "build_object_track.py",
    "merge_object_tracks.py", "render_multi_overlay.py", "validate_annotation_export.py",
    "finalize_video_annotation.py",
)


def _ignore(_: str, names: list[str]) -> set[str]:
    return {name for name in names if name in FORBIDDEN_PARTS or name.endswith((".pyc", ".pyo", ".bak"))}


def _copy_allowed() -> None:
    for item in ALLOWED:
        source = ROOT / item
        if not source.exists():
            continue
        destination = STAGE / item
        if source.is_dir():
            shutil.copytree(source, destination, ignore=_ignore)
        else:
            shutil.copy2(source, destination)


def _run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _scan_release() -> None:
    large: list[Path] = []
    violations: list[str] = []
    for path in STAGE.rglob("*"):
        relative = path.relative_to(STAGE)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            violations.append(f"forbidden path: {relative}")
        if path.is_file():
            if path.stat().st_size > 50 * 1024 * 1024:
                large.append(relative)
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".mp4", ".zip", ".gz", ".pt", ".pth"}:
                content = path.read_text(encoding="utf-8", errors="ignore")
                private_home = "/" + "home" + "/" + "lenovo"
                windows_drive_paths = (chr(67) + ":\\", chr(68) + ":\\")
                if private_home in content or any(prefix in content for prefix in windows_drive_paths):
                    violations.append(f"absolute/local path text: {relative}")
    if large:
        violations.extend(f"large file (>50 MiB): {path}" for path in large)
    if violations:
        raise RuntimeError("release scan failed:\n" + "\n".join(violations))


def _archives() -> tuple[Path, Path, Path]:
    tar_path = ARCHIVE_BASE.with_suffix(".tar.gz")
    zip_path = ARCHIVE_BASE.with_suffix(".zip")
    sha_path = DIST / f"{PACKAGE}.SHA256"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(STAGE, arcname=PACKAGE)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STAGE.rglob("*")):
            archive.write(path, path.relative_to(DIST))
    lines = []
    for path in (tar_path, zip_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    sha_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tar_path, zip_path, sha_path


def build_release() -> None:
    """Run preflight checks and create release artifacts under dist only."""
    print("[1/6] Run tests")
    _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    print("[2/6] Check CLI help")
    for name in HELP_SCRIPTS:
        _run([sys.executable, f"scripts/{name}", "--help"])
    print("[3/6] Rebuild dist staging directory")
    if DIST.exists():
        shutil.rmtree(DIST)
    STAGE.mkdir(parents=True)
    _copy_allowed()
    print("[4/6] Compile staged Python")
    if not compileall.compile_dir(STAGE / "src", quiet=1) or not compileall.compile_dir(STAGE / "scripts", quiet=1):
        raise RuntimeError("staged compileall failed")
    for cache in STAGE.rglob("__pycache__"):
        shutil.rmtree(cache)
    print("[5/6] Scan release contents")
    _scan_release()
    print("[6/6] Create archives and checksums")
    tar_path, zip_path, sha_path = _archives()
    manifest = sorted(str(path.relative_to(STAGE)) for path in STAGE.rglob("*") if path.is_file())
    (DIST / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"release directory: {STAGE}")
    print(f"tar.gz: {tar_path}")
    print(f"zip: {zip_path}")
    print(f"sha256: {sha_path}")
    print(f"release size: {sum(path.stat().st_size for path in STAGE.rglob('*') if path.is_file())} bytes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate a clean source release in dist/.")
    parser.parse_args()
    build_release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

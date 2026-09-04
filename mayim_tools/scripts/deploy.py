"""
Mayim Tools – Local Deploy Script
Copies the plugin to the QGIS user plugins folder for live testing.
Run this from the project root after making changes:
    python mayim_tools/scripts/deploy.py
"""

import shutil
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────── #

PLUGIN_NAME = "mayim_tools"

# Update this path to match your QGIS plugins folder:
QGIS_PLUGINS_DIR = Path(
    r"C:\Users\caets\AppData\Roaming\QGIS\QGIS4" r"\profiles\default\python\plugins"
)

# ── Source: the mayim_tools package folder ──────────────────────────────────  #
SOURCE_DIR = Path(__file__).resolve().parent.parent

# ── Destination: inside the QGIS plugins folder ─────────────────────────── #
DEST_DIR = QGIS_PLUGINS_DIR / PLUGIN_NAME

# ── Files and folders to exclude from deployment ────────────────────────── #
EXCLUDE = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "tests",
    "*.pyc",
    "*.pyo",
}


def should_exclude(path: Path) -> bool:
    """Check if a path should be excluded from deployment."""
    for pattern in EXCLUDE:
        if path.match(pattern) or path.name in EXCLUDE:
            return True
    return False


def deploy() -> None:
    """
    Copy the mayim_tools package to the QGIS plugins directory.
    Removes the existing deployment first for a clean install.
    """

    print(f"\n{'─' * 55}")
    print("  Mayim Tools — Local Deploy Script")
    print(f"{'─' * 55}")
    print(f"  Source : {SOURCE_DIR}")
    print(f"  Target : {DEST_DIR}")
    print(f"{'─' * 55}\n")

    # ── Check that the QGIS plugins folder exists ──
    if not QGIS_PLUGINS_DIR.exists():
        print(f"❌ ERROR: QGIS plugins folder not found:\n   {QGIS_PLUGINS_DIR}")
        print(
            "\n   Please update QGIS_PLUGINS_DIR in this script "
            "to match your system path."
        )
        sys.exit(1)

    # ── Remove existing deployment ──
    if DEST_DIR.exists():
        print("🗑️  Removing existing deployment...")
        shutil.rmtree(DEST_DIR)
        print("   Done.\n")

    # ── Copy plugin files to QGIS plugins folder ──
    print("📦 Copying plugin files...")

    copied = 0
    skipped = 0

    for item in SOURCE_DIR.rglob("*"):
        # Skip excluded files and folders
        if should_exclude(item):
            skipped += 1
            continue

        # Calculate relative path and destination
        relative = item.relative_to(SOURCE_DIR)
        destination = DEST_DIR / relative

        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            copied += 1

    print(f"   ✅ {copied} files copied.")
    print(f"   ⏭️  {skipped} files skipped.\n")

    # ── Also copy metadata.txt from project root if present ──
    root_metadata = SOURCE_DIR.parent / "metadata.txt"
    if root_metadata.exists():
        shutil.copy2(root_metadata, DEST_DIR / "metadata.txt")
        print("   📄 metadata.txt copied from project root.\n")

    print(f"{'─' * 55}")
    print("  ✅ Deploy complete!")
    print("  👉 Reload the plugin in QGIS:")
    print("     Plugins > Manage and Install Plugins >")
    print("     Installed > Mayim Tools > Reload")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    deploy()

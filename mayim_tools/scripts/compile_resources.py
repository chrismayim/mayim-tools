"""
Mayim Tools – Resource Compilation Script
Compiles the Qt resource file (resources.qrc) into resources_rc.py.
Run this whenever you add new icons or assets to resources.qrc:
    python mayim_tools/scripts/compile_resources.py
"""

import subprocess
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────── #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QRC_FILE = PROJECT_ROOT / "resources.qrc"
OUTPUT_FILE = PROJECT_ROOT / "resources_rc.py"

# ── pyrcc6 location options ────────────────────────────────────────────────── #
# pyrcc6 is the Qt6 resource compiler.
# On Windows with QGIS, it is typically available via the OSGeo4W shell.
# Update this path if pyrcc6 is installed elsewhere on your system.

PYRCC6_OPTIONS = [
    "pyrcc6",  # If on system PATH
    r"C:\Program Files\QGIS 4.x\apps\Python3xx\Scripts\pyrcc6.exe",
]


def find_pyrcc6() -> str:
    """
    Locate the pyrcc6 executable.
    Tries each known location in order.

    :returns: Path to pyrcc6 executable
    :raises FileNotFoundError: If pyrcc6 cannot be found
    """
    import shutil

    for option in PYRCC6_OPTIONS:
        if shutil.which(option) or Path(option).exists():
            return option

    raise FileNotFoundError(
        "\n❌ pyrcc6 not found. Please install it with:\n"
        "   pip install pyqt6\n"
        "   or ensure it is available in your OSGeo4W environment."
    )


def compile_resources() -> None:
    """
    Compile resources.qrc into resources_rc.py using pyrcc6.
    """

    print(f"\n{'─' * 55}")
    print("  Mayim Tools — Resource Compiler")
    print(f"{'─' * 55}")
    print(f"  Input  : {QRC_FILE}")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"{'─' * 55}\n")

    # ── Check that resources.qrc exists ──
    if not QRC_FILE.exists():
        print(f"❌ ERROR: resources.qrc not found at:\n   {QRC_FILE}")
        sys.exit(1)

    # ── Find pyrcc6 ──
    try:
        pyrcc6 = find_pyrcc6()
        print(f"✅ Found pyrcc6: {pyrcc6}\n")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # ── Run pyrcc6 ──
    print("⚙️  Compiling resources...")
    result = subprocess.run(
        [pyrcc6, str(QRC_FILE), "-o", str(OUTPUT_FILE)],
        capture_output=True,
        check=False,
        text=True,
    )

    if result.returncode == 0:
        print(f"   ✅ Compiled successfully → {OUTPUT_FILE.name}")
    else:
        print(f"   ❌ Compilation failed:\n{result.stderr}")
        sys.exit(1)

    print(f"\n{'─' * 55}")
    print("  ✅ Resource compilation complete!")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    compile_resources()

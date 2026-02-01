#!/usr/bin/env python3
"""Sync src/rpi_edgetpu/service.py into the embedded heredoc in install.sh."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICE_PY = REPO_ROOT / "src" / "rpi_edgetpu" / "service.py"
INSTALL_SH = REPO_ROOT / "install.sh"

START_MARKER = "cat > \"$dest\" << 'SERVICEEOF'"
END_MARKER = "SERVICEEOF"


def main():
    service_content = SERVICE_PY.read_text()
    install_lines = INSTALL_SH.read_text().splitlines(keepends=True)

    start_idx = None
    end_idx = None

    for i, line in enumerate(install_lines):
        if START_MARKER in line and start_idx is None:
            start_idx = i
        elif start_idx is not None and line.rstrip() == END_MARKER:
            end_idx = i
            break

    if start_idx is None:
        raise SystemExit(f"ERROR: start marker not found: {START_MARKER}")
    if end_idx is None:
        raise SystemExit(f"ERROR: end marker not found: {END_MARKER}")

    old_body = install_lines[start_idx + 1 : end_idx]
    new_body = service_content.splitlines(keepends=True)
    # Ensure final newline
    if new_body and not new_body[-1].endswith("\n"):
        new_body[-1] += "\n"

    if old_body == new_body:
        print("Already in sync — no changes needed.")
        return

    old_count = len(old_body)
    new_count = len(new_body)

    result = install_lines[: start_idx + 1] + new_body + install_lines[end_idx:]
    INSTALL_SH.write_text("".join(result))

    print(f"Replaced {old_count} lines with {new_count} lines in install.sh")


if __name__ == "__main__":
    main()

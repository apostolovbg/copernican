# Last Updated: 2025-11-24
"""Tests for the program diagnostics logging helpers."""

from pathlib import Path

from copernican_lib import logger as log_mod


def test_program_log_rotation(tmp_path):
    log_path = log_mod.setup_program_logging(
        log_dir=tmp_path,
        base_dir=str(tmp_path),
        rollover_mb=0.0001,
        backup_count=1,
    )
    prog_logger = log_mod.get_program_logger()
    message = "x" * 200
    for _ in range(200):
        prog_logger.info(message)
    primary = Path(log_path)
    rotated = primary.with_name(primary.name + ".1")
    assert primary.exists()
    assert rotated.exists()
    assert primary.read_text(encoding="utf-8")

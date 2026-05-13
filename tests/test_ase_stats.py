import json
from pathlib import Path

from gaussiangpt_ae.data.ase_scene import read_ase_stats


def test_read_ase_stats_reads_json_and_records_sources(tmp_path: Path) -> None:
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    (stats_dir / "stats.json").write_text(
        json.dumps({"psnr": 28.5, "ssim": 0.91, "num_GS": 123}),
        encoding="utf-8",
    )

    stats = read_ase_stats(stats_dir)

    assert stats["psnr"] == 28.5
    assert stats["ssim"] == 0.91
    assert stats["num_GS"] == 123
    assert stats["_source_files"] == ["stats.json"]
    assert stats["_parse_errors"] == {}


def test_read_ase_stats_records_parse_errors(tmp_path: Path) -> None:
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    (stats_dir / "bad.txt").write_text("not parseable", encoding="utf-8")

    stats = read_ase_stats(stats_dir)

    assert "bad.txt" in stats["_parse_errors"]

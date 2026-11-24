import pytest

from gemclip.chunks import plan_windows


def test_one_hour_video_is_split_with_overlap():
    windows = plan_windows(0, 3600, chunk_sec=900, overlap_sec=60)
    assert windows[0] == (0.0, 900.0)
    assert windows[-1][1] == 3600.0
    assert all(b - a <= 900 for a, b in windows)
    # setiap jendela berikutnya dimulai 60 detik sebelum jendela sebelumnya berakhir
    for (a1, b1), (a2, _b2) in zip(windows, windows[1:]):
        assert a2 == pytest.approx(b1 - 60)


def test_short_range_is_single_window():
    assert plan_windows(600, 900, chunk_sec=900, overlap_sec=60) == [(600.0, 900.0)]


def test_empty_range_returns_no_windows():
    assert plan_windows(100, 100) == []
    assert plan_windows(200, 100) == []


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        plan_windows(0, 1000, chunk_sec=60, overlap_sec=60)


import shutil
import subprocess

from gemclip.chunks import cut_chunk

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg tidak tersedia")


@needs_ffmpeg
def test_cut_chunk_produces_exact_duration_from_offset(tmp_path):
    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=20",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )
    dest = tmp_path / "part.mp4"
    cut_chunk(source, start=5.0, duration=4.0, dest=dest)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(dest)],
        capture_output=True, text=True, check=True,
    )
    assert float(probe.stdout.strip()) == pytest.approx(4.0, abs=0.2)

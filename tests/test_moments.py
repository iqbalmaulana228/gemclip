import pytest

from gemclip.models import Word
from gemclip.moments import MomentDraft, clean_moments, parse_drafts


def _draft(start, end, score, title="Judul", hashtags=None):
    return MomentDraft(
        start_sec=start,
        end_sec=end,
        viral_score=score,
        title=title,
        hook="Hook",
        caption="Caption",
        hashtags=hashtags if hashtags is not None else ["viral", "#shorts"],
    )


def _words(span_end=120.0, step=0.5):
    out = []
    t = 0.0
    while t + step <= span_end:
        out.append(Word(start=round(t, 2), end=round(t + step - 0.05, 2), text="kata"))
        t += step
    return out


def test_clean_moments_sorts_by_score_and_limits_count():
    drafts = [_draft(0, 40, 60), _draft(50, 90, 90), _draft(100, 140, 75)]
    result = clean_moments(drafts, _words(), duration=200, count=2, min_sec=30, max_sec=60)
    assert [m.score for m in result] == [90, 75]


def test_clean_moments_drops_overlapping_lower_score():
    drafts = [_draft(10, 50, 80), _draft(30, 70, 95)]
    result = clean_moments(drafts, _words(), duration=200, count=5, min_sec=30, max_sec=60)
    assert len(result) == 1
    assert result[0].score == 95


def test_clean_moments_drops_too_short_clips():
    result = clean_moments([_draft(10, 20, 80)], _words(), duration=200, count=5, min_sec=30, max_sec=60)
    assert result == []


def test_clean_moments_clamps_to_video_duration():
    result = clean_moments([_draft(150, 400, 80)], _words(), duration=180, count=5, min_sec=20, max_sec=60)
    assert len(result) == 1
    assert result[0].end <= 180


def test_clean_moments_trims_overlong_clip_to_max_length():
    result = clean_moments([_draft(0, 200, 80)], _words(span_end=300), duration=300, count=5, min_sec=30, max_sec=60)
    assert len(result) == 1
    assert result[0].end - result[0].start <= 60 + 0.01


def test_clean_moments_snaps_boundaries_to_word_edges():
    words = [Word(10.0, 10.4, "a"), Word(40.0, 40.4, "b"), Word(70.0, 70.4, "c")]
    result = clean_moments([_draft(10.3, 40.1, 80)], words, duration=100, count=1, min_sec=20, max_sec=60)
    m = result[0]
    # awal di-snap ke 10.0 lalu dikurangi padding 0.15, akhir di-snap ke 40.4 lalu ditambah 0.25
    assert m.start == pytest.approx(9.85, abs=0.01)
    assert m.end == pytest.approx(40.65, abs=0.01)


def test_clean_moments_normalises_hashtags():
    result = clean_moments(
        [_draft(0, 40, 80, hashtags=["viral", "#shorts", "  tips rahasia  ", "", "viral"])],
        _words(),
        duration=100,
        count=1,
        min_sec=30,
        max_sec=60,
    )
    assert result[0].hashtags == ("#viral", "#shorts", "#tipsrahasia")


def test_parse_drafts_accepts_valid_json():
    raw = (
        '[{"start_sec": 1.5, "end_sec": 40.0, "viral_score": 88, "title": "T", '
        '"hook": "H", "caption": "C", "hashtags": ["a"]}]'
    )
    drafts = parse_drafts(raw)
    assert drafts[0].viral_score == 88


def test_parse_drafts_rejects_invalid_json():
    with pytest.raises(RuntimeError):
        parse_drafts("bukan json")


def test_clean_moments_limits_results_to_range():
    drafts = [_draft(0, 40, 90), _draft(120, 160, 80), _draft(300, 340, 70)]
    result = clean_moments(
        drafts, _words(span_end=400), duration=400, count=5, min_sec=30, max_sec=60,
        range_start=100, range_end=200,
    )
    assert len(result) == 1
    assert result[0].start >= 100 and result[0].end <= 200


def test_build_prompt_includes_instruction_only_when_given():
    from gemclip.moments import build_prompt
    from gemclip.models import Segment

    seg = [Segment(0.0, 5.0, "halo")]
    plain = build_prompt(seg, 3, 30, 60, window_len=900)
    assert "Instruksi tambahan" not in plain
    assert "900 detik" in plain

    with_instruction = build_prompt(seg, 3, 30, 60, window_len=900, instruction="fokus pada harga")
    assert "Instruksi tambahan" in with_instruction
    assert "fokus pada harga" in with_instruction


def test_build_prompt_truncates_long_instruction():
    from gemclip.moments import MAX_INSTRUCTION_CHARS, build_prompt

    prompt = build_prompt([], 3, 30, 60, window_len=900, instruction="a" * 5000)
    assert "a" * (MAX_INSTRUCTION_CHARS + 1) not in prompt

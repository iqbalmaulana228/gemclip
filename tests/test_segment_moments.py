import pytest

from gemclip.models import Segment
from gemclip.moments import SegmentDraft, build_segment_prompt, parse_segment_drafts, segments_to_moments

SEGS = [
    Segment(0.0, 10.0, "satu"),
    Segment(10.5, 22.0, "dua"),
    Segment(22.5, 40.0, "tiga"),
    Segment(41.0, 70.0, "empat"),
    Segment(80.0, 100.0, "lima"),
]


def _draft(first, last, score=80, hashtags=None):
    return SegmentDraft(
        first=first,
        last=last,
        viral_score=score,
        title="Judul",
        hook="Hook",
        caption="Caption",
        hashtags=hashtags if hashtags is not None else ["a"],
    )


def _moments(drafts, count=5, lo=0.0, hi=200.0, segments=SEGS):
    return segments_to_moments(drafts, segments, count, 30, 60, lo, hi)


def test_moment_starts_and_ends_on_sentence_edges_with_padding():
    [m] = _moments([_draft(0, 2)])
    assert m.start == 0.0
    assert m.end == pytest.approx(40.25)  # akhir kalimat ke-3 (40.0) ditambah napas 0.25


def test_short_moment_is_extended_to_minimum_length():
    # kalimat 1-2 hanya 29.5 detik, jadi kalimat ke-4 ikut masuk
    [m] = _moments([_draft(1, 2)])
    assert m.start == pytest.approx(10.35)
    assert m.end == pytest.approx(70.25)


def test_overlong_moment_backs_off_to_sentence_that_fits():
    # kalimat 0-4 mencakup 100 detik, jadi berhenti di kalimat ke-3 yang masih dalam 60 detik
    [m] = _moments([_draft(0, 4)])
    assert m.end == pytest.approx(40.25)
    assert m.end - m.start <= 60 + 0.4


def test_out_of_range_or_too_short_drafts_are_dropped():
    assert _moments([_draft(-1, 2), _draft(3, 9), _draft(4, 2)]) == []
    assert _moments([_draft(4, 4)]) == [], "satu kalimat 20 detik tidak bisa mencapai 30 detik"


def test_range_limits_the_moment():
    [m] = _moments([_draft(0, 2)], lo=5.0, hi=45.0)
    assert m.start == 5.0
    assert m.end <= 45.0


def test_overlapping_moments_keep_higher_score():
    result = _moments([_draft(0, 2, score=60), _draft(0, 2, score=90)])
    assert len(result) == 1
    assert result[0].score == 90


def test_count_limits_results_to_best_scores():
    drafts = [_draft(0, 2, score=70), _draft(3, 4, score=95)]
    result = _moments(drafts, count=1)
    assert [m.score for m in result] == [95]


def test_hashtags_are_normalised():
    [m] = _moments([_draft(0, 2, hashtags=["viral", "#shorts", "viral"])])
    assert m.hashtags == ("#viral", "#shorts")


def test_parse_segment_drafts_accepts_indexes_and_rejects_garbage():
    raw = (
        '[{"first": 1, "last": 2, "viral_score": 88, "title": "T", '
        '"hook": "H", "caption": "C", "hashtags": ["a"]}]'
    )
    [draft] = parse_segment_drafts(raw)
    assert (draft.first, draft.last, draft.viral_score) == (1, 2, 88)
    with pytest.raises(RuntimeError):
        parse_segment_drafts("bukan json")


def test_prompt_numbers_each_sentence_and_asks_for_indexes():
    prompt = build_segment_prompt(SEGS[:2], count=3, min_sec=30, max_sec=60, range_start=0, range_end=40)
    assert "0 [0.0-10.0] satu" in prompt
    assert "1 [10.5-22.0] dua" in prompt
    assert "Jangan menulis waktu sendiri" in prompt

import pytest

from gemclip.captions import STYLES, build_ass, build_srt, clip_text, group_words
from gemclip.models import Word


def test_group_words_breaks_on_word_limit():
    words = [Word(i * 0.3, i * 0.3 + 0.25, f"w{i}") for i in range(9)]
    groups = group_words(words)
    assert all(len(g) <= 4 for g in groups)
    assert sum(len(g) for g in groups) == 9


def test_group_words_breaks_on_long_pause():
    words = [Word(0.0, 0.3, "satu"), Word(0.4, 0.7, "dua"), Word(3.0, 3.3, "tiga")]
    groups = group_words(words)
    assert [len(g) for g in groups] == [2, 1]


def test_build_ass_uses_relative_times_and_karaoke_tags():
    words = [
        Word(10.0, 10.5, "Halo"),
        Word(10.6, 11.0, "dunia"),
        Word(50.0, 50.5, "di luar klip"),
    ]
    ass = build_ass(words, clip_start=10.0, clip_duration=5.0)
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue: 0")]
    assert len(dialogues) == 1
    assert "0:00:00.00" in dialogues[0]
    assert "\\k50" in dialogues[0]
    assert "di luar klip" not in ass


def test_build_ass_adds_hook_overlay():
    ass = build_ass([Word(0.0, 0.5, "ya")], clip_start=0.0, clip_duration=10.0, hook="Ini rahasianya")
    assert "Dialogue: 1,0:00:00.00,0:00:03.00,Hook" in ass
    assert "Ini rahasianya" in ass


def test_build_ass_strips_ass_control_characters():
    ass = build_ass([Word(0.0, 0.5, "{\\an8}curang")], clip_start=0.0, clip_duration=5.0)
    assert "{\\an8}curang" not in ass
    assert "an8curang" in ass


def test_build_ass_default_size_is_vertical_1080x1920():
    ass = build_ass([Word(0.0, 0.5, "ya")], clip_start=0.0, clip_duration=2.0)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass


@pytest.mark.parametrize("size", [(1080, 1080), (1920, 1080)])
def test_build_ass_follows_output_size(size):
    ass = build_ass([Word(0.0, 0.5, "ya")], clip_start=0.0, clip_duration=2.0, size=size)
    assert f"PlayResX: {size[0]}" in ass
    assert f"PlayResY: {size[1]}" in ass


@pytest.mark.parametrize("style", STYLES)
def test_every_style_builds_a_valid_header(style):
    ass = build_ass([Word(0.0, 0.5, "ya")], clip_start=0.0, clip_duration=2.0, hook="H", style=style)
    assert "Style: Caption," in ass and "Style: Hook," in ass
    assert "\\k50" in ass


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError):
        build_ass([], clip_start=0.0, clip_duration=1.0, style="pelangi")


def test_build_srt_is_numbered_and_relative():
    words = [
        Word(10.0, 10.5, "Halo"),
        Word(10.6, 11.0, "dunia"),
        Word(12.0, 12.4, "lagi"),
        Word(40.0, 40.5, "di luar"),
    ]
    srt = build_srt(words, clip_start=10.0, clip_duration=5.0)
    blocks = srt.strip().split("\n\n")
    assert len(blocks) == 2
    assert blocks[0].startswith("1\n00:00:00,000 --> ")
    assert "Halo dunia" in blocks[0]
    assert blocks[1].startswith("2\n00:00:02,000 --> ")
    assert "di luar" not in srt


def test_clip_text_keeps_only_words_inside_range():
    words = [Word(1.0, 1.4, "satu"), Word(2.0, 2.4, "dua"), Word(9.0, 9.4, "tiga")]
    assert clip_text(words, 0.5, 3.0) == "satu dua"
    assert clip_text(words, 20.0, 30.0) == ""

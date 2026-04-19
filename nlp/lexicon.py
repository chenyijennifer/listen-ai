"""Lexicon-based sentiment (baseline)."""

from __future__ import annotations

import re

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "love",
    "awesome",
    "happy",
    "amazing",
    "nice",
    "best",
    "positive",
    "fast",
    "smooth",
    "reliable",
}

POSITIVE_WORDS_ZH_TW = {
    "好",
    "很好",
    "優秀",
    "喜歡",
    "讚",
    "開心",
    "高興",
    "棒",
    "最佳",
    "正面",
    "快速",
    "順暢",
    "可靠",
    "滿意",
    "推薦",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "hate",
    "worst",
    "slow",
    "bug",
    "bugs",
    "issue",
    "issues",
    "angry",
    "broken",
    "negative",
    "expensive",
}

NEGATIVE_WORDS_ZH_TW = {
    "差",
    "糟糕",
    "很糟",
    "討厭",
    "最差",
    "慢",
    "錯誤",
    "問題",
    "生氣",
    "壞掉",
    "負面",
    "昂貴",
    "失望",
    "卡頓",
}

NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "hardly",
    "不",
    "沒",
    "無",
    "未",
    "別",
    "不是",
}

POSITIVE_WORDS_ALL = POSITIVE_WORDS | POSITIVE_WORDS_ZH_TW
NEGATIVE_WORDS_ALL = NEGATIVE_WORDS | NEGATIVE_WORDS_ZH_TW

CJK_LEXICON_TERMS = sorted(
    POSITIVE_WORDS_ZH_TW | NEGATIVE_WORDS_ZH_TW | {w for w in NEGATION_WORDS if re.search(r"[\u4e00-\u9fff]", w)},
    key=len,
    reverse=True,
)


def _tokenize_cjk_segment(segment: str) -> list[str]:
    tokens: list[str] = []
    idx = 0

    while idx < len(segment):
        match = ""
        for term in CJK_LEXICON_TERMS:
            if segment.startswith(term, idx):
                match = term
                break

        if match:
            tokens.append(match)
            idx += len(match)
        else:
            tokens.append(segment[idx])
            idx += 1

    return tokens


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-zA-Z']+|[\u4e00-\u9fff]+", text.lower())
    tokens: list[str] = []

    for raw in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            tokens.extend(_tokenize_cjk_segment(raw))
        else:
            tokens.append(raw)

    return tokens


def classify_text_lexicon(text: str) -> tuple[str, int]:
    tokens = tokenize(text)
    score = 0
    previous_tokens = ["", ""]

    for token in tokens:
        is_negated = any(prev in NEGATION_WORDS for prev in previous_tokens)

        if token in POSITIVE_WORDS_ALL:
            score += -1 if is_negated else 1
        elif token in NEGATIVE_WORDS_ALL:
            score += 1 if is_negated else -1

        previous_tokens = [previous_tokens[-1], token]

    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "neutral", score

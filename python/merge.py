"""
Module for transcription segment merging utilities
"""

import re
from typing import List, Literal, Tuple
import difflib


def overlap_interval(
    a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[float, float]:
    """Return the overlapping time range between two intervals, or (0, 0) if none."""
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return (start, end) if end > start else (0.0, 0.0)


def merge_strings(
    a: str, b: str, mode: Literal["union", "intersect"] = "intersect"
) -> str:
    """
    Merge two texts at the word level.

    Args:
        a: First text segment
        b: Second text segment
        mode:
            - "intersect": keep only words/phrases appearing in both
            - "union": merge all words, preserving both unique and shared parts

    Returns:
        Merged string
    """
    a_words = a.split()
    b_words = b.split()
    matcher = difflib.SequenceMatcher(None, a_words, b_words)
    merged_parts: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if mode == "intersect":
            if tag == "equal":
                merged_parts.extend(a_words[i1:i2])
        elif mode == "union":
            if tag == "equal":
                merged_parts.extend(a_words[i1:i2])
            elif tag == "replace":
                merged_parts.extend(a_words[i1:i2] + b_words[j1:j2])
            elif tag == "insert":
                merged_parts.extend(b_words[j1:j2])
            elif tag == "delete":
                merged_parts.extend(a_words[i1:i2])
        else:
            raise ValueError("mode must be 'union' or 'intersect'")

    merged_text = " ".join(merged_parts)
    return re.sub(r"\s+", " ", merged_text).strip()


def merge_segments(
    segments: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    """
    Find overlapping pairs of segments and merge their overlapping parts.
    Returns a list of validated merged (start, end, text) segments.
    """
    if not segments:
        return []

    # Ensure sorted by start time
    segments = sorted(segments, key=lambda x: x[0])

    validated_segs: List[Tuple[float, float, str]] = []

    # get intersections between segments
    for i, _ in enumerate(segments):
        for j in range(i + 1, len(segments)):
            s1, e1, t1 = segments[i]
            s2, e2, t2 = segments[j]
            overlap = overlap_interval((s1, e1), (s2, e2))
            if overlap != (0.0, 0.0):
                start, end = overlap
                merged_text = merge_strings(t1, t2, "intersect")
                validated_segs.append((start, end, merged_text))
            elif s2 > e1:
                # No need to continue further since segments are sorted
                break

    # Merge validated segments that overlap each other
    merged_overlaps: List[Tuple[float, float, str]] = []
    for seg in sorted(validated_segs, key=lambda x: x[0]):
        if not merged_overlaps:
            merged_overlaps.append(seg)
        else:
            prev_s, prev_e, prev_t = merged_overlaps[-1]
            s, e, t = seg
            if s <= prev_e:  # overlap or touch
                new_e = max(prev_e, e)
                new_t = merge_strings(prev_t, t, "union")
                merged_overlaps[-1] = (prev_s, new_e, new_t)
            else:
                merged_overlaps.append(seg)
    return " ".join([segment[2] for segment in merged_overlaps])

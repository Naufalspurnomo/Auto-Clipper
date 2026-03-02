from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from mlbb_pipeline.common import clamp, seconds_to_timestamp

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional during bootstrap before deps install
    OpenAI = None


def build_candidates(
    duration_sec: float,
    audio_scores: list[dict[str, Any]],
    ocr_scores: list[dict[str, Any]],
    motion_scores: list[dict[str, Any]],
    clip_min_sec: int,
    clip_max_sec: int,
    max_candidates: int = 50,
    transcript_segments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fuse audio/OCR/motion signals into peak-centered clip candidates."""
    bucket: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"audio": 0.0, "ocr": 0.0, "motion": 0.0, "hits": set()}
    )

    for item in audio_scores:
        sec = int(round(item.get("timestamp_sec", 0.0)))
        bucket[sec]["audio"] = max(bucket[sec]["audio"], float(item.get("score", 0.0)))

    for item in ocr_scores:
        sec = int(round(item.get("timestamp_sec", 0.0)))
        bucket[sec]["ocr"] = max(bucket[sec]["ocr"], float(item.get("score", 0.0)))
        for hit in item.get("hits", []):
            bucket[sec]["hits"].add(hit)

    for item in motion_scores:
        sec = int(round(item.get("timestamp_sec", 0.0)))
        bucket[sec]["motion"] = max(bucket[sec]["motion"], float(item.get("score", 0.0)))

    points = []
    end_sec = int(max(1, round(duration_sec)))
    for sec in range(end_sec + 1):
        frame = bucket[sec]
        combo = 0.45 * frame["audio"] + 0.35 * frame["ocr"] + 0.20 * frame["motion"]
        points.append(
            {
                "sec": sec,
                "score": float(combo),
                "audio_score": float(frame["audio"]),
                "ocr_score": float(frame["ocr"]),
                "motion_score": float(frame["motion"]),
                "hits": sorted(frame["hits"]),
            }
        )

    peaks = _pick_peaks(points=points, distance=max(5, clip_min_sec // 2), limit=max_candidates)
    target_duration = int((clip_min_sec + clip_max_sec) / 2)

    candidates: list[dict[str, Any]] = []
    for idx, peak in enumerate(peaks, 1):
        start = clamp(peak["sec"] - target_duration / 2.0, 0.0, duration_sec)
        end = clamp(start + target_duration, 0.0, duration_sec)
        if end - start < clip_min_sec:
            end = clamp(start + clip_min_sec, 0.0, duration_sec)
            start = clamp(end - clip_min_sec, 0.0, duration_sec)

        snippet = _find_transcript_snippet(
            transcript_segments=transcript_segments or [],
            start_sec=start,
            end_sec=end,
        )
        candidates.append(
            {
                "candidate_id": f"cand_{idx:03d}",
                "peak_sec": peak["sec"],
                "peak_ts": seconds_to_timestamp(peak["sec"]),
                "start_sec": round(float(start), 3),
                "end_sec": round(float(end), 3),
                "duration_sec": round(float(end - start), 3),
                "combined_score": round(float(peak["score"]), 4),
                "audio_peak_score": round(float(peak["audio_score"]), 4),
                "ocr_score": round(float(peak["ocr_score"]), 4),
                "motion_score": round(float(peak["motion_score"]), 4),
                "ocr_hits": peak["hits"],
                "transcript_snippet": snippet,
            }
        )

    candidates.sort(key=lambda item: item["combined_score"], reverse=True)
    return candidates[:max_candidates]


def llm_rank_and_refine(
    candidates: list[dict[str, Any]],
    clip_count: int,
    clip_min_sec: int,
    clip_max_sec: int,
    llm_model: str,
    api_key: str | None,
    base_url: str | None,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Ask LLM to select best candidate windows and refine boundaries."""
    if not candidates:
        return []
    if not api_key or OpenAI is None:
        logger.warning("LLM ranking disabled (missing api key or openai package).")
        return _fallback_selection(candidates, clip_count)

    payload = [
        {
            "candidate_id": c["candidate_id"],
            "peak_ts": c["peak_ts"],
            "start_sec": c["start_sec"],
            "end_sec": c["end_sec"],
            "duration_sec": c["duration_sec"],
            "combined_score": c["combined_score"],
            "audio_peak_score": c["audio_peak_score"],
            "ocr_score": c["ocr_score"],
            "motion_score": c["motion_score"],
            "ocr_hits": c["ocr_hits"],
            "transcript_snippet": c.get("transcript_snippet", ""),
        }
        for c in candidates[:50]
    ]

    system_prompt = (
        "You are ranking MLBB gaming clip candidates for TikTok virality. "
        "Select punchy moments with high action and clear payoff. "
        "Return JSON array only."
    )
    user_prompt = (
        "Pick top clips from candidates.\n"
        f"Need exactly {clip_count} clips.\n"
        f"Each clip duration must be between {clip_min_sec} and {clip_max_sec} seconds.\n"
        "For each selected item return: "
        "candidate_id, start_sec, end_sec, hook_text, reason.\n"
        "Keep start/end close to candidate suggestion unless clear improvement.\n\n"
        f"Candidates:\n{json.dumps(payload, ensure_ascii=True)}"
    )

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    try:
        response = client.chat.completions.create(
            model=llm_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _extract_json_array(content)
        if not parsed:
            raise ValueError("Empty/invalid JSON from LLM")
        return _coerce_llm_selection(
            llm_items=parsed,
            candidates=candidates,
            clip_count=clip_count,
            clip_min_sec=clip_min_sec,
            clip_max_sec=clip_max_sec,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM ranking failed, fallback to heuristic. reason=%s", exc)
        return _fallback_selection(candidates, clip_count)


def _pick_peaks(points: list[dict[str, Any]], distance: int, limit: int) -> list[dict[str, Any]]:
    if not points:
        return []
    threshold = _percentile([p["score"] for p in points], 70)
    locals_only = []
    for i in range(1, len(points) - 1):
        prev_score = points[i - 1]["score"]
        current = points[i]["score"]
        next_score = points[i + 1]["score"]
        if current >= prev_score and current >= next_score and current >= threshold:
            locals_only.append(points[i])
    locals_only.sort(key=lambda item: item["score"], reverse=True)

    selected: list[dict[str, Any]] = []
    for point in locals_only:
        if len(selected) >= limit:
            break
        if any(abs(point["sec"] - existing["sec"]) < distance for existing in selected):
            continue
        selected.append(point)

    if len(selected) < min(limit, 10):
        # Backfill with global highs in case local peaks are sparse.
        fallback = sorted(points, key=lambda item: item["score"], reverse=True)
        for point in fallback:
            if len(selected) >= limit:
                break
            if any(abs(point["sec"] - existing["sec"]) < distance for existing in selected):
                continue
            selected.append(point)
    return selected


def _find_transcript_snippet(
    transcript_segments: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> str:
    if not transcript_segments:
        return ""
    snippets: list[str] = []
    for segment in transcript_segments:
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", 0.0))
        if seg_end < start_sec or seg_start > end_sec:
            continue
        text = str(segment.get("text", "")).strip()
        if text:
            snippets.append(text)
        if len(" ".join(snippets)) > 180:
            break
    return " ".join(snippets)[:180]


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _coerce_llm_selection(
    llm_items: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    clip_count: int,
    clip_min_sec: int,
    clip_max_sec: int,
) -> list[dict[str, Any]]:
    candidates_by_id = {item["candidate_id"]: item for item in candidates}
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for item in llm_items:
        candidate_id = str(item.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id in used_ids:
            continue
        base = candidates_by_id.get(candidate_id)
        if not base:
            continue

        start = float(item.get("start_sec", base["start_sec"]))
        end = float(item.get("end_sec", base["end_sec"]))
        if end <= start:
            start = float(base["start_sec"])
            end = float(base["end_sec"])
        duration = end - start
        if duration < clip_min_sec:
            end = start + clip_min_sec
        if duration > clip_max_sec:
            end = start + clip_max_sec
        if end <= start:
            end = start + clip_min_sec

        selected.append(
            {
                **base,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "hook_text": str(item.get("hook_text", "")).strip(),
                "reason": str(item.get("reason", "")).strip(),
                "selection_source": "llm",
            }
        )
        used_ids.add(candidate_id)
        if len(selected) >= clip_count:
            break

    if len(selected) < clip_count:
        fallback = _fallback_selection(candidates, clip_count)
        for item in fallback:
            if item["candidate_id"] in used_ids:
                continue
            selected.append(item)
            if len(selected) >= clip_count:
                break
    return selected[:clip_count]


def _fallback_selection(candidates: list[dict[str, Any]], clip_count: int) -> list[dict[str, Any]]:
    selected = []
    for item in candidates[:clip_count]:
        selected.append(
            {
                **item,
                "hook_text": "",
                "reason": "Heuristic top score fallback",
                "selection_source": "heuristic",
            }
        )
    return selected


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    index = int((q / 100.0) * (len(values_sorted) - 1))
    return float(values_sorted[index])


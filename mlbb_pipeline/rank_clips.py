from __future__ import annotations

import json
import logging
from collections import Counter
from collections import defaultdict
from typing import Any

from mlbb_pipeline.common import clamp, seconds_to_timestamp

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional during bootstrap before deps install
    OpenAI = None


OCR_HIT_WEIGHTS = {
    "SAVAGE": 1.0,
    "MANIAC": 0.94,
    "TRIPLE KILL": 0.84,
    "DOUBLE KILL": 0.68,
    "MEGA KILL": 0.56,
    "MONSTER KILL": 0.64,
    "KILLING SPREE": 0.48,
    "UNSTOPPABLE": 0.6,
    "GODLIKE": 0.74,
    "LEGENDARY": 0.9,
    "WIPED OUT": 0.96,
    "WIPE OUT": 0.96,
    "SHUT DOWN": 0.72,
    "HAS SLAIN": 0.34,
    "DESTROYED": 0.44,
    "TURRET": 0.38,
    "LORD": 0.8,
    "TURTLE": 0.62,
}

EVENT_PROFILE_HOOKS = {
    "teamfight_explosion": "Teamfight ini pecah total!",
    "objective_swing": "Objective ini langsung balik momentum!",
    "snowball_spike": "Momentum snowball di sini gila!",
    "pickoff_punish": "Pickoff ini langsung ngubah game!",
    "mechanical_outplay": "Outplay ini wajib lihat ending-nya!",
    "gameplay_spike": "Momen ini langsung naik tensi!",
}


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
    """Fuse gameplay signals into ranked action windows instead of raw peaks."""
    bucket: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "audio": 0.0,
            "ocr": 0.0,
            "motion": 0.0,
            "hits": set(),
            "event_types": set(),
            "event_strength": 0.0,
            "burst": 0.0,
        }
    )

    for item in audio_scores:
        sec = int(round(item.get("timestamp_sec", 0.0)))
        bucket[sec]["audio"] = max(bucket[sec]["audio"], float(item.get("score", 0.0)))

    for item in ocr_scores:
        sec = int(round(item.get("timestamp_sec", 0.0)))
        bucket[sec]["ocr"] = max(bucket[sec]["ocr"], float(item.get("score", 0.0)))
        bucket[sec]["event_strength"] = max(
            bucket[sec]["event_strength"],
            float(item.get("event_strength", _score_ocr_hits(item.get("hits", [])))),
        )
        bucket[sec]["burst"] = max(bucket[sec]["burst"], float(item.get("burst", 0.0)))
        for hit in item.get("hits", []):
            bucket[sec]["hits"].add(hit)
        for event_type in item.get("event_types", []):
            bucket[sec]["event_types"].add(event_type)

    for item in motion_scores:
        sec = int(round(item.get("timestamp_sec", 0.0)))
        bucket[sec]["motion"] = max(bucket[sec]["motion"], float(item.get("score", 0.0)))

    points = []
    end_sec = int(max(1, round(duration_sec)))
    for sec in range(end_sec + 1):
        frame = bucket[sec]
        combo = (
            0.32 * frame["audio"]
            + 0.22 * frame["ocr"]
            + 0.18 * frame["motion"]
            + 0.18 * frame["event_strength"]
            + 0.10 * min(1.0, frame["burst"] * 2.4)
        )
        points.append(
            {
                "sec": sec,
                "score": float(combo),
                "audio_score": float(frame["audio"]),
                "ocr_score": float(frame["ocr"]),
                "motion_score": float(frame["motion"]),
                "hits": sorted(frame["hits"]),
                "event_types": sorted(frame["event_types"]),
                "event_strength": float(frame["event_strength"]),
                "burst": float(frame["burst"]),
            }
        )

    _attach_smoothed_scores(points)
    peaks = _pick_peaks(points=points, distance=max(5, clip_min_sec // 2), limit=max_candidates * 3)

    candidates: list[dict[str, Any]] = []
    for idx, peak in enumerate(peaks, 1):
        candidate = _build_candidate_window(
            peak=peak,
            points=points,
            duration_sec=duration_sec,
            clip_min_sec=clip_min_sec,
            clip_max_sec=clip_max_sec,
            transcript_segments=transcript_segments or [],
        )
        if not candidate:
            continue
        candidate["candidate_id"] = f"cand_{idx:03d}"
        candidates.append(candidate)

    candidates = _dedupe_candidates(candidates, max_candidates=max_candidates)
    for idx, candidate in enumerate(candidates, 1):
        candidate["candidate_id"] = f"cand_{idx:03d}"
    return candidates[:max_candidates]


def _attach_smoothed_scores(points: list[dict[str, Any]]) -> None:
    if not points:
        return

    raw_scores = [float(item.get("score", 0.0)) for item in points]
    for index, point in enumerate(points):
        left = max(0, index - 2)
        right = min(len(points), index + 3)
        window = raw_scores[left:right]
        point["smoothed_score"] = float(sum(window) / max(1, len(window)))
        point["peak_seed"] = float(
            0.72 * point["smoothed_score"]
            + 0.14 * point.get("event_strength", 0.0)
            + 0.08 * point.get("burst", 0.0)
            + 0.06 * point.get("motion_score", 0.0)
        )


def _build_candidate_window(
    peak: dict[str, Any],
    points: list[dict[str, Any]],
    duration_sec: float,
    clip_min_sec: int,
    clip_max_sec: int,
    transcript_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    peak_sec = float(peak.get("sec", 0.0))
    target_duration = int((clip_min_sec + clip_max_sec) / 2)
    duration_options = sorted({clip_min_sec, target_duration, clip_max_sec})
    center_biases = (0.58, 0.52, 0.65, 0.45, 0.72)

    best_window: dict[str, Any] | None = None
    best_window_score = -1.0

    for duration in duration_options:
        for center_bias in center_biases:
            start = clamp(peak_sec - duration * center_bias, 0.0, duration_sec)
            end = clamp(start + duration, 0.0, duration_sec)
            if end - start < clip_min_sec:
                end = clamp(start + clip_min_sec, 0.0, duration_sec)
                start = clamp(end - clip_min_sec, 0.0, duration_sec)
            if end - start > clip_max_sec:
                end = clamp(start + clip_max_sec, 0.0, duration_sec)

            if peak_sec < start or peak_sec > end or end <= start:
                continue

            analysis = _analyze_window(points=points, start_sec=start, end_sec=end)
            if not analysis:
                continue

            peak_position = (peak_sec - start) / max(0.001, end - start)
            structure_bonus = max(0.0, 1.0 - abs(peak_position - 0.62))
            window_score = analysis["selection_score"] + 0.08 * structure_bonus

            if window_score > best_window_score:
                best_window_score = window_score
                best_window = {
                    "peak_sec": int(round(peak_sec)),
                    "peak_ts": seconds_to_timestamp(peak_sec),
                    "start_sec": round(float(start), 3),
                    "end_sec": round(float(end), 3),
                    "duration_sec": round(float(end - start), 3),
                    "combined_score": round(float(analysis["combined_score"]), 4),
                    "selection_score": round(float(analysis["selection_score"]), 4),
                    "audio_peak_score": round(float(analysis["audio_peak_score"]), 4),
                    "ocr_score": round(float(analysis["ocr_peak_score"]), 4),
                    "motion_score": round(float(analysis["motion_peak_score"]), 4),
                    "ocr_hits": analysis["priority_hits"],
                    "event_types": analysis["event_types"],
                    "event_profile": analysis["event_profile"],
                    "action_grade": analysis["action_grade"],
                    "signal_density": round(float(analysis["signal_density"]), 4),
                    "payoff_score": round(float(analysis["payoff_score"]), 4),
                    "burst_peak": round(float(analysis["burst_peak"]), 4),
                    "signal_reason": analysis["signal_reason"],
                    "analysis_summary": analysis["analysis_summary"],
                    "gameplay_analysis": analysis["gameplay_analysis"],
                    "transcript_snippet": _find_transcript_snippet(
                        transcript_segments=transcript_segments,
                        start_sec=start,
                        end_sec=end,
                    ),
                }

    return best_window or {}


def _analyze_window(
    points: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    window = [
        item for item in points
        if float(item.get("sec", 0.0)) >= start_sec and float(item.get("sec", 0.0)) <= end_sec
    ]
    if not window:
        return {}

    avg_signal = _avg(item.get("smoothed_score", item.get("score", 0.0)) for item in window)
    max_signal = max(float(item.get("smoothed_score", item.get("score", 0.0))) for item in window)
    avg_audio = _avg(item.get("audio_score", 0.0) for item in window)
    peak_audio = max(float(item.get("audio_score", 0.0)) for item in window)
    avg_motion = _avg(item.get("motion_score", 0.0) for item in window)
    peak_motion = max(float(item.get("motion_score", 0.0)) for item in window)
    peak_ocr = max(float(item.get("ocr_score", 0.0)) for item in window)
    peak_event = max(float(item.get("event_strength", 0.0)) for item in window)
    burst_peak = max(float(item.get("burst", 0.0)) for item in window)
    signal_density = sum(
        1 for item in window if float(item.get("smoothed_score", item.get("score", 0.0))) >= 0.62
    ) / max(1, len(window))

    midpoint = max(1, len(window) // 2)
    front = window[:midpoint]
    back = window[midpoint:]
    front_avg = _avg(item.get("smoothed_score", item.get("score", 0.0)) for item in front)
    back_avg = _avg(item.get("smoothed_score", item.get("score", 0.0)) for item in back)
    back_peak = max(float(item.get("smoothed_score", item.get("score", 0.0))) for item in back or window)
    payoff_score = max(
        0.0,
        min(
            1.0,
            0.48 * back_peak
            + 0.28 * back_avg
            + 0.24 * max(0.0, back_avg - front_avg + 0.25),
        ),
    )

    hit_counter: Counter[str] = Counter()
    event_type_counter: Counter[str] = Counter()
    for item in window:
        for hit in item.get("hits", []):
            normalized = str(hit).strip().upper()
            if normalized:
                hit_counter[normalized] += 1
        for event_type in item.get("event_types", []):
            normalized_type = str(event_type).strip().lower()
            if normalized_type:
                event_type_counter[normalized_type] += 1

    priority_hits = _rank_hits(hit_counter)
    event_profile = _classify_event_profile(
        hit_counter=hit_counter,
        event_type_counter=event_type_counter,
        peak_motion=peak_motion,
        peak_audio=peak_audio,
        payoff_score=payoff_score,
    )
    event_types = [name for name, _ in event_type_counter.most_common(3)]
    synergy = 0.0
    synergy += 0.34 if peak_audio >= 0.68 else 0.0
    synergy += 0.33 if peak_motion >= 0.62 else 0.0
    synergy += 0.33 if peak_event >= 0.55 or peak_ocr >= 0.5 else 0.0
    hit_bonus = min(0.18, len(priority_hits) * 0.04)

    selection_score = max(
        0.0,
        min(
            1.0,
            0.20 * avg_signal
            + 0.18 * max_signal
            + 0.14 * peak_event
            + 0.12 * peak_motion
            + 0.10 * peak_audio
            + 0.10 * signal_density
            + 0.10 * payoff_score
            + 0.06 * burst_peak
            + 0.06 * synergy
            + 0.04 * hit_bonus,
        ),
    )

    action_grade = _score_to_grade(selection_score)
    reason_bits = []
    if priority_hits:
        reason_bits.append(f"OCR kuat: {', '.join(priority_hits[:2])}")
    if peak_motion >= 0.68:
        reason_bits.append("motion tinggi")
    if peak_audio >= 0.68:
        reason_bits.append("audio ramai")
    if signal_density >= 0.35:
        reason_bits.append("aksi rapat")
    if payoff_score >= 0.62:
        reason_bits.append("payoff kuat")
    if not reason_bits:
        reason_bits.append("kombinasi sinyal gameplay stabil")

    analysis_summary = (
        f"{event_profile.replace('_', ' ')} | grade {action_grade} | "
        f"peak={max_signal:.2f} density={signal_density:.2f}"
    )

    return {
        "combined_score": max_signal,
        "selection_score": selection_score,
        "audio_peak_score": peak_audio,
        "ocr_peak_score": peak_ocr,
        "motion_peak_score": peak_motion,
        "burst_peak": burst_peak,
        "signal_density": signal_density,
        "payoff_score": payoff_score,
        "priority_hits": priority_hits,
        "event_types": event_types,
        "event_profile": event_profile,
        "action_grade": action_grade,
        "signal_reason": "; ".join(reason_bits[:4]),
        "analysis_summary": analysis_summary,
        "gameplay_analysis": {
            "avg_signal": round(float(avg_signal), 4),
            "peak_signal": round(float(max_signal), 4),
            "avg_audio": round(float(avg_audio), 4),
            "peak_audio": round(float(peak_audio), 4),
            "avg_motion": round(float(avg_motion), 4),
            "peak_motion": round(float(peak_motion), 4),
            "peak_event_strength": round(float(peak_event), 4),
            "burst_peak": round(float(burst_peak), 4),
            "signal_density": round(float(signal_density), 4),
            "payoff_score": round(float(payoff_score), 4),
            "priority_hits": priority_hits,
            "event_profile": event_profile,
            "event_type_counts": dict(event_type_counter),
        },
    }


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
            "selection_score": c.get("selection_score", c["combined_score"]),
            "audio_peak_score": c["audio_peak_score"],
            "ocr_score": c["ocr_score"],
            "motion_score": c["motion_score"],
            "ocr_hits": c["ocr_hits"],
            "event_types": c.get("event_types", []),
            "event_profile": c.get("event_profile", ""),
            "action_grade": c.get("action_grade", ""),
            "signal_density": c.get("signal_density", 0.0),
            "payoff_score": c.get("payoff_score", 0.0),
            "signal_reason": c.get("signal_reason", ""),
            "transcript_snippet": c.get("transcript_snippet", ""),
        }
        for c in candidates[:50]
    ]

    system_prompt = (
        "You are ranking MLBB gaming clip candidates for TikTok virality. "
        "Prioritize windows with clear payoff, multi-signal action, strong OCR events, "
        "and non-overlapping clip variety. Favor moments that feel complete: setup -> chaos -> payoff. "
        "Return JSON array only."
    )
    user_prompt = (
        "Pick top clips from candidates.\n"
        f"Need exactly {clip_count} clips.\n"
        f"Each clip duration must be between {clip_min_sec} and {clip_max_sec} seconds.\n"
        "Prefer candidates with high selection_score, payoff_score, and strong event_profile.\n"
        "Avoid selecting overlapping candidates unless there is a very strong reason.\n"
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
    threshold = _percentile([p.get("peak_seed", p["score"]) for p in points], 72)
    locals_only = []
    for i in range(1, len(points) - 1):
        prev_score = points[i - 1].get("peak_seed", points[i - 1]["score"])
        current = points[i].get("peak_seed", points[i]["score"])
        next_score = points[i + 1].get("peak_seed", points[i + 1]["score"])
        strong_event = points[i].get("event_strength", 0.0) >= 0.6
        if current >= prev_score and current >= next_score and (current >= threshold or strong_event):
            locals_only.append(points[i])
    locals_only.sort(key=lambda item: item.get("peak_seed", item["score"]), reverse=True)

    selected: list[dict[str, Any]] = []
    for point in locals_only:
        if len(selected) >= limit:
            break
        if any(abs(point["sec"] - existing["sec"]) < distance for existing in selected):
            continue
        selected.append(point)

    if len(selected) < min(limit, 10):
        # Backfill with global highs in case local peaks are sparse.
        fallback = sorted(points, key=lambda item: item.get("peak_seed", item["score"]), reverse=True)
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

        candidate = {
            **base,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(end - start, 3),
            "hook_text": str(item.get("hook_text", "")).strip() or _candidate_default_hook(base),
            "reason": str(item.get("reason", "")).strip() or base.get("signal_reason", ""),
            "selection_source": "llm",
        }
        if any(_overlap_ratio(candidate, existing) >= 0.55 for existing in selected):
            continue
        selected.append(candidate)
        used_ids.add(candidate_id)
        if len(selected) >= clip_count:
            break

    if len(selected) < clip_count:
        fallback = _fallback_selection(candidates, clip_count)
        for item in fallback:
            if item["candidate_id"] in used_ids:
                continue
            if any(_overlap_ratio(item, existing) >= 0.55 for existing in selected):
                continue
            selected.append(item)
            if len(selected) >= clip_count:
                break
    return selected[:clip_count]


def _fallback_selection(candidates: list[dict[str, Any]], clip_count: int) -> list[dict[str, Any]]:
    selected = []
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.get("selection_score", item.get("combined_score", 0.0)),
            item.get("payoff_score", 0.0),
            item.get("signal_density", 0.0),
        ),
        reverse=True,
    )
    for item in ranked:
        if any(_overlap_ratio(item, existing) >= 0.55 for existing in selected):
            continue
        selected.append(
            {
                **item,
                "hook_text": _candidate_default_hook(item),
                "reason": item.get("signal_reason", "Heuristic top score fallback"),
                "selection_source": "heuristic",
            }
        )
        if len(selected) >= clip_count:
            break
    return selected


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    index = int((q / 100.0) * (len(values_sorted) - 1))
    return float(values_sorted[index])


def _score_ocr_hits(hits: list[str]) -> float:
    if not hits:
        return 0.0
    weights = [OCR_HIT_WEIGHTS.get(str(hit).strip().upper(), 0.2) for hit in hits]
    strongest = max(weights)
    count_bonus = min(0.2, max(0, len(set(hits)) - 1) * 0.05)
    return min(1.0, strongest + count_bonus)


def _rank_hits(hit_counter: Counter[str]) -> list[str]:
    ranked = sorted(
        hit_counter.items(),
        key=lambda item: (OCR_HIT_WEIGHTS.get(item[0], 0.2), item[1]),
        reverse=True,
    )
    return [name for name, _ in ranked[:4]]


def _classify_event_profile(
    hit_counter: Counter[str],
    event_type_counter: Counter[str],
    peak_motion: float,
    peak_audio: float,
    payoff_score: float,
) -> str:
    priority_hits = _rank_hits(hit_counter)
    if any(hit in {"SAVAGE", "MANIAC", "WIPED OUT", "WIPE OUT", "TRIPLE KILL"} for hit in priority_hits):
        return "teamfight_explosion"
    if any(hit in {"LORD", "TURTLE", "DESTROYED", "TURRET"} for hit in priority_hits):
        return "objective_swing"
    if any(hit in {"LEGENDARY", "GODLIKE", "UNSTOPPABLE", "MONSTER KILL", "MEGA KILL"} for hit in priority_hits):
        return "snowball_spike"
    if any(hit in {"SHUT DOWN", "HAS SLAIN"} for hit in priority_hits):
        return "pickoff_punish"
    if peak_motion >= 0.72 and peak_audio >= 0.62:
        return "mechanical_outplay"
    if event_type_counter.get("teamfight", 0) >= 2 or payoff_score >= 0.7:
        return "teamfight_explosion"
    return "gameplay_spike"


def _score_to_grade(score: float) -> str:
    if score >= 0.82:
        return "S"
    if score >= 0.68:
        return "A"
    if score >= 0.54:
        return "B"
    return "C"


def _avg(values) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _dedupe_candidates(candidates: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.get("selection_score", item.get("combined_score", 0.0)),
            item.get("payoff_score", 0.0),
            item.get("signal_density", 0.0),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    for item in ranked:
        if any(_overlap_ratio(item, existing) >= 0.62 for existing in selected):
            continue
        selected.append(item)
        if len(selected) >= max_candidates:
            break
    return selected


def _overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    start_a = float(left.get("start_sec", 0.0))
    end_a = float(left.get("end_sec", 0.0))
    start_b = float(right.get("start_sec", 0.0))
    end_b = float(right.get("end_sec", 0.0))
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    if intersection <= 0.0:
        return 0.0
    duration_a = max(0.001, end_a - start_a)
    duration_b = max(0.001, end_b - start_b)
    return intersection / min(duration_a, duration_b)


def _candidate_default_hook(candidate: dict[str, Any]) -> str:
    priority_hits = [str(hit).strip().upper() for hit in candidate.get("ocr_hits", [])]
    profile = str(candidate.get("event_profile", "")).strip().lower()
    if priority_hits:
        primary = priority_hits[0]
        if primary in {"SAVAGE", "MANIAC", "TRIPLE KILL", "DOUBLE KILL"}:
            return f"{primary.title()} ini bikin chaos!"
        if primary in {"LORD", "TURTLE"}:
            return f"{primary.title()} fight ini nentuin game!"
        if primary in {"WIPED OUT", "WIPE OUT"}:
            return "Wipe out ini langsung nutup game!"
        if primary == "SHUT DOWN":
            return "Shutdown ini langsung balik momentum!"
        if primary in {"LEGENDARY", "GODLIKE"}:
            return f"{primary.title()} moment yang bikin mental drop!"

    return EVENT_PROFILE_HOOKS.get(profile, "Momen MLBB ini wajib ditonton!")

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


TEMPLATE_GAMEPLAY_EPIC = "GAMEPLAY_EPIC"
TEMPLATE_TALK_HOTTAKE = "TALK_HOTTAKE"
TEMPLATE_COMEDY_REACTION = "COMEDY_REACTION"
TEMPLATE_MULTICAM = "MULTICAM"
TEMPLATE_DONATION_QNA = "DONATION_QNA"
TEMPLATE_PRO_ENCOUNTER = "PRO_ENCOUNTER"

ALL_TEMPLATES = {
    TEMPLATE_GAMEPLAY_EPIC,
    TEMPLATE_TALK_HOTTAKE,
    TEMPLATE_COMEDY_REACTION,
    TEMPLATE_MULTICAM,
    TEMPLATE_DONATION_QNA,
    TEMPLATE_PRO_ENCOUNTER,
}

GAMEPLAY_HIT_KEYWORDS = {
    "SAVAGE",
    "MANIAC",
    "TRIPLE KILL",
    "DOUBLE KILL",
    "WIPED OUT",
    "WIPE OUT",
    "LEGENDARY",
    "GODLIKE",
    "SHUT DOWN",
    "HAS SLAIN",
    "LORD",
    "TURTLE",
}

DONATION_KEYWORDS = (
    "donasi",
    "saweria",
    "trakteer",
    "superchat",
    "gift",
    "thanks bang",
    "terima kasih bang",
    "baca chat",
    "chat nanya",
    "tip",
)
TALK_KEYWORDS = (
    "drama",
    "tanggapan",
    "opini",
    "klarifikasi",
    "bahas",
    "jawab",
    "kenapa",
    "gimana",
    "katanya",
    "rumor",
    "transfer",
    "isu",
)
COMEDY_KEYWORDS = (
    "wkwk",
    "wkwwk",
    "haha",
    "hahaha",
    "ngakak",
    "lucu",
    "ketawa",
    "lol",
)
PRO_KEYWORDS = (
    "pro player",
    "top global",
    "nabrak",
    "ketemu",
    "versus",
    "lawan",
    "rrq",
    "evos",
    "onic",
    "geek",
    "alter ego",
    "aura",
    "bigetron",
    "liquid",
    "falcons",
)

DEFAULT_HOOKS = {
    TEMPLATE_GAMEPLAY_EPIC: "Momen ini pecah banget!",
    TEMPLATE_TALK_HOTTAKE: "Omongan ini langsung bikin panas.",
    TEMPLATE_COMEDY_REACTION: "Reaksi ini bikin ngakak.",
    TEMPLATE_MULTICAM: "Mabar rame, reaksinya chaos.",
    TEMPLATE_DONATION_QNA: "Pertanyaan chat ini langsung dijawab.",
    TEMPLATE_PRO_ENCOUNTER: "Nabrak nama besar di rank ini.",
}

HOOK_DURATION_BY_TEMPLATE = {
    TEMPLATE_GAMEPLAY_EPIC: 1.8,
    TEMPLATE_TALK_HOTTAKE: 2.2,
    TEMPLATE_COMEDY_REACTION: 2.0,
    TEMPLATE_MULTICAM: 2.0,
    TEMPLATE_DONATION_QNA: 2.2,
    TEMPLATE_PRO_ENCOUNTER: 2.0,
}


@dataclass
class TemplateDecision:
    template: str
    confidence: float
    reasons: list[str]
    hook_text: str
    hook_duration_sec: float


def classify_edit_template(
    clip: dict[str, Any],
    facecam_mode: str,
    transcript_text: str,
) -> TemplateDecision:
    mode = str(facecam_mode or "").strip().upper()
    text = _normalize_text(
        " ".join(
            [
                str(clip.get("hook_text", "")),
                str(clip.get("reason", "")),
                transcript_text,
            ]
        )
    )
    gameplay_analysis = clip.get("gameplay_analysis") or {}
    ocr_hits = {str(item).strip().upper() for item in (clip.get("ocr_hits") or [])}
    ocr_hits.update(
        str(item).strip().upper()
        for item in (gameplay_analysis.get("priority_hits") or [])
        if str(item).strip()
    )
    motion = _as_float(clip.get("motion_score"))
    audio = _as_float(clip.get("audio_peak_score"))
    combo = _as_float(clip.get("combined_score"))
    selection_score = _as_float(clip.get("selection_score"))
    signal_density = _as_float(clip.get("signal_density"))
    payoff_score = _as_float(clip.get("payoff_score"))
    event_profile = str(clip.get("event_profile", "")).strip().lower()
    action_grade = str(clip.get("action_grade", "")).strip().upper()
    signal_reason = str(clip.get("signal_reason", "")).strip()

    scores = {name: 0.0 for name in ALL_TEMPLATES}
    reasons: list[str] = []

    if mode == "MULTI_CAM_STRIP":
        scores[TEMPLATE_MULTICAM] += 5.0
        reasons.append("Facecam terdeteksi strip multi-cam.")

    if ocr_hits.intersection(GAMEPLAY_HIT_KEYWORDS):
        scores[TEMPLATE_GAMEPLAY_EPIC] += 3.5
        scores[TEMPLATE_PRO_ENCOUNTER] += 0.8
        reasons.append(f"OCR hit gameplay: {sorted(ocr_hits.intersection(GAMEPLAY_HIT_KEYWORDS))}.")

    if event_profile == "teamfight_explosion":
        scores[TEMPLATE_GAMEPLAY_EPIC] += 3.0
        scores[TEMPLATE_PRO_ENCOUNTER] += 0.7
        reasons.append("Analisis gameplay membaca teamfight dengan payoff kuat.")
    elif event_profile == "objective_swing":
        scores[TEMPLATE_GAMEPLAY_EPIC] += 2.4
        reasons.append("Analisis gameplay membaca objective swing.")
    elif event_profile == "snowball_spike":
        scores[TEMPLATE_GAMEPLAY_EPIC] += 2.1
        scores[TEMPLATE_PRO_ENCOUNTER] += 0.9
        reasons.append("Analisis gameplay membaca momentum snowball.")
    elif event_profile == "pickoff_punish":
        scores[TEMPLATE_GAMEPLAY_EPIC] += 1.8
        reasons.append("Analisis gameplay membaca pickoff/shutdown.")
    elif event_profile == "mechanical_outplay":
        scores[TEMPLATE_GAMEPLAY_EPIC] += 2.6
        reasons.append("Analisis gameplay membaca outplay mekanik.")

    if motion >= 0.65:
        scores[TEMPLATE_GAMEPLAY_EPIC] += 2.0
        reasons.append("Motion tinggi, cocok gameplay dominan.")
    elif motion >= 0.45:
        scores[TEMPLATE_GAMEPLAY_EPIC] += 1.0

    if audio >= 0.65:
        scores[TEMPLATE_COMEDY_REACTION] += 0.7
        scores[TEMPLATE_TALK_HOTTAKE] += 0.4

    if action_grade == "S":
        scores[TEMPLATE_GAMEPLAY_EPIC] += 1.4
    elif action_grade == "A":
        scores[TEMPLATE_GAMEPLAY_EPIC] += 0.9

    if selection_score >= 0.72:
        scores[TEMPLATE_GAMEPLAY_EPIC] += 1.0
    if signal_density >= 0.34:
        scores[TEMPLATE_GAMEPLAY_EPIC] += 0.9
    if payoff_score >= 0.62:
        scores[TEMPLATE_GAMEPLAY_EPIC] += 0.8

    donation_hits = _keyword_hits(text, DONATION_KEYWORDS)
    talk_hits = _keyword_hits(text, TALK_KEYWORDS)
    comedy_hits = _keyword_hits(text, COMEDY_KEYWORDS)
    pro_hits = _keyword_hits(text, PRO_KEYWORDS)

    if donation_hits:
        scores[TEMPLATE_DONATION_QNA] += 3.2 + 0.2 * len(donation_hits)
        reasons.append(f"Terdeteksi konteks donasi/chat: {donation_hits[:3]}.")

    if talk_hits:
        scores[TEMPLATE_TALK_HOTTAKE] += 2.5 + 0.15 * len(talk_hits)
        reasons.append(f"Terdeteksi konteks tanggapan/drama: {talk_hits[:3]}.")

    if comedy_hits:
        scores[TEMPLATE_COMEDY_REACTION] += 2.1 + 0.15 * len(comedy_hits)
        reasons.append(f"Terdeteksi konteks komedi: {comedy_hits[:3]}.")

    if pro_hits:
        scores[TEMPLATE_PRO_ENCOUNTER] += 2.8 + 0.15 * len(pro_hits)
        reasons.append(f"Terdeteksi konteks pro encounter: {pro_hits[:3]}.")

    if combo >= 0.72:
        scores[TEMPLATE_GAMEPLAY_EPIC] += 0.9
    elif combo <= 0.35:
        scores[TEMPLATE_TALK_HOTTAKE] += 0.5

    if mode == "NO_FACE":
        scores[TEMPLATE_GAMEPLAY_EPIC] += 0.5

    # If not strongly multicam, prevent false-positive over-trigger.
    if mode != "MULTI_CAM_STRIP":
        scores[TEMPLATE_MULTICAM] -= 1.0

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    template = ranked[0][0]
    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = _clamp(0.5 + (top_score - second_score) * 0.12, 0.5, 0.98)

    if not reasons:
        reasons.append("Fallback ke heuristik score tertinggi.")
    elif signal_reason:
        reasons.append(signal_reason)

    hook_text = build_hook_text(template=template, clip=clip, transcript_text=transcript_text)
    hook_duration = HOOK_DURATION_BY_TEMPLATE.get(template, 2.0)

    return TemplateDecision(
        template=template,
        confidence=round(confidence, 4),
        reasons=reasons[:4],
        hook_text=hook_text,
        hook_duration_sec=float(hook_duration),
    )


def extract_transcript_text(transcript_payload: dict[str, Any]) -> str:
    if not transcript_payload:
        return ""
    text = str(transcript_payload.get("text", "")).strip()
    if text:
        return _normalize_ws(text)[:500]

    segments = transcript_payload.get("segments") or []
    if isinstance(segments, list):
        joined = " ".join(str(item.get("text", "")).strip() for item in segments[:20] if isinstance(item, dict))
        if joined.strip():
            return _normalize_ws(joined)[:500]
    return ""


def build_hook_text(
    template: str,
    clip: dict[str, Any],
    transcript_text: str,
) -> str:
    explicit = _normalize_ws(str(clip.get("hook_text", "")).strip())
    if explicit:
        return _truncate(explicit, 82)

    ocr_hits = [
        str(item).strip().upper()
        for item in (clip.get("ocr_hits") or [])
        if str(item).strip()
    ]
    if not ocr_hits:
        gameplay_analysis = clip.get("gameplay_analysis") or {}
        ocr_hits = [
            str(item).strip().upper()
            for item in (gameplay_analysis.get("priority_hits") or [])
            if str(item).strip()
        ]

    event_profile = str(clip.get("event_profile", "")).strip().lower()
    if ocr_hits:
        primary = ocr_hits[0]
        if primary in {"SAVAGE", "MANIAC", "TRIPLE KILL", "DOUBLE KILL"}:
            return _truncate(f"{primary.title()} ini bikin lobby panik!", 82)
        if primary in {"LORD", "TURTLE"}:
            return _truncate(f"{primary.title()} fight ini penentu game!", 82)
        if primary in {"WIPED OUT", "WIPE OUT"}:
            return _truncate("Wipe out ini langsung jadi penutup!", 82)
        if primary == "SHUT DOWN":
            return _truncate("Shutdown ini langsung ubah tempo game!", 82)
        if primary in {"LEGENDARY", "GODLIKE", "UNSTOPPABLE"}:
            return _truncate(f"{primary.title()} moment yang bikin mental drop!", 82)

    if event_profile:
        profile_hooks = {
            "teamfight_explosion": "Teamfight ini pecah banget di ending!",
            "objective_swing": "Objective ini langsung balik momentum!",
            "snowball_spike": "Momentum snowball di sini gila banget!",
            "pickoff_punish": "Pickoff ini langsung ngubah game!",
            "mechanical_outplay": "Outplay ini bersih banget!",
            "gameplay_spike": "Momen ini bikin tensi langsung naik!",
        }
        if event_profile in profile_hooks:
            return _truncate(profile_hooks[event_profile], 82)

    transcript = _normalize_ws(transcript_text)
    if transcript:
        lead = _lead_phrase(transcript)
        if lead:
            if template == TEMPLATE_GAMEPLAY_EPIC:
                return _truncate(f"{lead}... lihat ending-nya!", 82)
            if template == TEMPLATE_TALK_HOTTAKE:
                return _truncate(f"{lead}... tanggapannya tegas banget.", 82)
            if template == TEMPLATE_DONATION_QNA:
                return _truncate(f"{lead}... jawaban chat ini daging.", 82)
            if template == TEMPLATE_COMEDY_REACTION:
                return _truncate(f"{lead}... ending-nya bikin ngakak.", 82)
            if template == TEMPLATE_PRO_ENCOUNTER:
                return _truncate(f"{lead}... momen lawan nama besar.", 82)
            if template == TEMPLATE_MULTICAM:
                return _truncate(f"{lead}... reaksi satu lobby rame.", 82)
    return DEFAULT_HOOKS.get(template, "Momen ini wajib ditonton.")


def hook_duration_for_template(template: str) -> float:
    return float(HOOK_DURATION_BY_TEMPLATE.get(template, 2.0))


def _lead_phrase(text: str) -> str:
    parts = re.split(r"[.!?]", text)
    for part in parts:
        cleaned = _normalize_ws(part)
        if len(cleaned.split()) >= 4:
            return _truncate(cleaned, 56)
    return _truncate(text, 56)


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for key in keywords:
        if key in text:
            hits.append(key)
    return hits


def _normalize_text(text: str) -> str:
    lowered = _normalize_ws(text).lower()
    return lowered


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

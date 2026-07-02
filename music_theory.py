from __future__ import annotations

import math
from typing import Any, Mapping


def _normalized_entropy(rows: list[Mapping[str, Any]]) -> float | None:
    counts = [float(row.get("count", 0) or 0) for row in rows if float(row.get("count", 0) or 0) > 0]
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return None
    probabilities = [count / total for count in counts]
    entropy = -sum(prob * math.log2(prob) for prob in probabilities if prob > 0)
    return entropy / math.log2(len(probabilities))


def _piece_density(piece: Mapping[str, Any]) -> int:
    return int(((piece.get("note_profile") or {}).get("note_events")) or 0)


def build_theory_surface(package: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(package.get("summary") or {})
    pieces = list(package.get("pieces") or [])
    parsed_note_events = int(summary.get("parsed_note_events") or 0)
    parsed_chord_events = int(summary.get("parsed_chord_events") or 0)
    notation_link_rate = float(summary.get("notation_link_rate") or 0.0)
    note_rows = list(summary.get("top_notes") or [])
    chord_rows = list(summary.get("top_chords") or [])
    transition_rows = list(summary.get("top_note_combinations") or [])
    chord_transition_rows = list(summary.get("top_chord_transitions") or [])
    note_entropy = _normalized_entropy(note_rows)
    chord_entropy = _normalized_entropy(chord_rows)
    note_dense_pieces = sorted(
        [
            {
                "title": piece.get("title"),
                "artist": piece.get("artist"),
                "genre": piece.get("genre"),
                "note_events": _piece_density(piece),
                "matched_file_count": int(
                    ((piece.get("notation_summary") or {}).get("matched_file_count")) or 0
                ),
            }
            for piece in pieces
            if _piece_density(piece) > 0
        ],
        key=lambda item: (-item["note_events"], str(item["title"] or "")),
    )[:8]

    if parsed_note_events > 0:
        headline = "The score-linked lane is active."
        posture = "measured"
        claim_boundaries = [
            "These note and chord signals come only from linked notation assets inside the repository.",
            "They do not stand in for the whole public music corpus or every uploaded recording.",
            "Market claims should be joined with the public reach lane before they are treated as strategic evidence.",
        ]
    else:
        headline = "The score-aware lane is ready, but still sparse."
        posture = "waiting_for_scores"
        claim_boundaries = [
            "The repository can parse notation, MIDI, ABC, LilyPond, tablature, and chord charts.",
            "No note-bearing assets are linked strongly enough yet to support harmonic claims across the public catalog.",
            "This lane is currently best used to show where deeper evidence is still missing.",
        ]
    method_choices = [
        {
            "measure": "Normalized note and chord entropy",
            "why": "Useful for understanding whether the linked score lane is harmonically varied or collapsing into a narrow set of repeated shapes.",
        },
        {
            "measure": "Transition surfaces",
            "why": "Motion between notes and chords often explains more than isolated counts. It shows how pieces travel, not only where they stop.",
        },
        {
            "measure": "Coverage before conclusion",
            "why": "A sparse score lane should invite contribution before it is turned into a market claim.",
        },
    ]

    return {
        "generated_at": summary.get("generated_at"),
        "headline": headline,
        "posture": posture,
        "coverage": {
            "catalog_song_count": int(summary.get("catalog_song_count") or 0),
            "matched_catalog_songs": int(summary.get("matched_catalog_songs") or 0),
            "notation_link_rate": round(notation_link_rate, 4),
            "pieces_with_notes": int(summary.get("pieces_with_notes") or 0),
            "pieces_with_chords": int(summary.get("pieces_with_chords") or 0),
            "parsed_note_events": parsed_note_events,
            "parsed_chord_events": parsed_chord_events,
        },
        "pitch_surface": {
            "normalized_note_entropy": None if note_entropy is None else round(note_entropy, 4),
            "normalized_chord_entropy": None if chord_entropy is None else round(chord_entropy, 4),
            "top_notes": note_rows[:8],
            "top_chords": chord_rows[:8],
        },
        "motion_surface": {
            "top_note_transitions": transition_rows[:8],
            "top_chord_transitions": chord_transition_rows[:8],
            "tempo_summary": summary.get("tempo_summary") or {},
            "key_distribution": list(summary.get("key_distribution") or [])[:8],
            "time_signature_distribution": list(summary.get("time_signature_distribution") or [])[:8],
        },
        "piece_examples": note_dense_pieces,
        "next_inputs": list(summary.get("priority_queue") or [])[:8],
        "method_choices": method_choices,
        "claim_boundaries": claim_boundaries,
    }

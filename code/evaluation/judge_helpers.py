"""Frozen judge context formatting and response-key aliases."""
import json

def format_video_context(metadata):
    if not isinstance(metadata, dict) or not metadata:
        return ""
    parts = []
    if metadata.get("candidate_id"):
        parts.append(f"Candidate id: {metadata.get('candidate_id')}")
    source_start = metadata.get("source_start_sec")
    source_end = metadata.get("source_end_sec")
    clip_start = metadata.get("clip_start_sec")
    clip_end = metadata.get("clip_end_sec")
    if source_start is not None and source_end is not None and clip_start is not None:
        relative_start = float(source_start) - float(clip_start)
        relative_end = float(source_end) - float(clip_start)
        parts.append(f"Candidate time in the provided clip: {relative_start:.2f}-{relative_end:.2f} seconds")
        if clip_end is not None:
            parts.append(f"The provided clip maps to source-video time {float(clip_start):.2f}-{float(clip_end):.2f} seconds")
        parts.append(f"Source-video candidate time: {float(source_start):.2f}-{float(source_end):.2f} seconds")
    elif metadata.get("Segment"):
        parts.append(f"Candidate time: {metadata.get('Segment')}")
    elif source_start is not None and source_end is not None:
        parts.append(f"Candidate time: {float(source_start):.2f}-{float(source_end):.2f} seconds")
    if metadata.get("evidence_text"):
        parts.append(f"Text-only candidate evidence: {metadata.get('evidence_text')}")
    if metadata.get("speaker_role"):
        parts.append(f"Target role: {metadata.get('speaker_role')}")
    uses_clip_transcript = "target_transcript_clip_context" in metadata or "partner_transcript_clip_context" in metadata
    if uses_clip_transcript:
        target_context = metadata.get("target_transcript_clip_context")
        partner_context = metadata.get("partner_transcript_clip_context")
    else:
        target_context = metadata.get("target_transcript_context")
        partner_context = metadata.get("partner_transcript_context")
    transcript_time_basis = "provided-clip-relative word times" if uses_clip_transcript else "source-video word times"
    if target_context:
        parts.append(f"Target transcript around candidate with {transcript_time_basis}: {target_context}")
    if partner_context:
        parts.append(f"Partner transcript around candidate with {transcript_time_basis}: {partner_context}")
    if metadata.get("reference_analysis"):
        parts.append(
            "Expected semantic motion from the independent reference annotation: "
            f"{metadata.get('reference_analysis')}"
        )
    if metadata.get("reference_motion_evidence"):
        parts.append(
            "Expected backchannel motion from the independent reference annotation: "
            f"{metadata.get('reference_motion_evidence')}"
        )
    if metadata.get("reference_identity_verified"):
        parts.append(
            "Local SHA256 integrity check: VIDEO 2 is byte-for-byte identical to VIDEO 1. "
            "Their visible motion and timing are therefore identical; do not report a visual difference."
        )
    if metadata.get("expected_backchannel_motion"):
        parts.append(f"Expected backchannel motion type for this test item: {metadata.get('expected_backchannel_motion')}")
    if not parts:
        return ""
    return (
        "\n\nPer-video evaluation context. Reference motion descriptions define "
        "the matching target but do not imply that the current video contains "
        "the described motion:\n"
        + "\n".join(parts)
    )

def normalize_seed_response(response):
    try:
        data = json.loads(response)
    except Exception:
        return response
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        return response
    aliases = {
        'Timing': ['timing'],
        'Appropriateness': ['appropriateness'],
        'Gesture backchannel visibility': ['Gesture_backchannel_visibility', 'gesture_backchannel_visibility', 'Visibility', 'visibility'],
        'Visual clarity': ['Visual_clarity', 'visual_clarity'],
        'Text-motion semantic match': ['Text_motion_semantic_match', 'text_motion_semantic_match'],
        'Concreteness': ['concreteness'],
        'Beat/fidget penalty': ['Beat_fidget_penalty', 'beat_fidget_penalty'],
    }
    for canonical, names in aliases.items():
        if canonical in data:
            continue
        for name in names:
            if name in data:
                data[canonical] = data.pop(name)
                break
    return json.dumps(data, ensure_ascii=False)

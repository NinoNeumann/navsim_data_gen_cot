import re
import torch
import numpy as np

def _pos_to_str(rel_pos):
    """
    Convert your provided coordinates to a string *as-is*, with no rounding.
    - If values are numbers, use Python's default stringification; if you need
      to preserve formatting exactly, pass coordinate strings in object_position_info.
    """
    x, y = rel_pos
    return f"({x}, {y})"

def _build_positions_and_canon_map(object_position_info):
    """
    Returns:
    - positions_text: the authoritative list text inserted into the system prompt
    - canon_map: {canonical_name: "(x, y)"} used by the post-processor to enforce coordinates
    """
    if not object_position_info:
        return "", {}

    lines = ["\n\nObject positions relative to ego vehicle (BEV perspective):"]
    canon_map = {}
    for obj in object_position_info:
        name = obj["name"]
        rel_pos = obj["relative_position"]
        distance = obj.get("distance", None)
        heading = obj.get("heading", None)

        pos_str = _pos_to_str(rel_pos)  # Do not round; keep raw string
        canon_map[name] = pos_str

        # Direction description (optional, to aid model understanding)
        x, y = rel_pos
        if abs(x) < 2 and abs(y) < 2:
            direction = "very close"
        elif x > 0 and y > 0:
            direction = "front-right"
        elif x > 0 and y < 0:
            direction = "front-left"
        elif x < 0 and y > 0:
            direction = "rear-right"
        elif x < 0 and y < 0:
            direction = "rear-left"
        elif x > 0:
            direction = "front"
        elif x < 0:
            direction = "rear"
        elif y > 0:
            direction = "right"
        else:
            direction = "left"

        heading_deg = None
        if heading is not None:
            heading_deg = float(np.degrees(heading))
        dist_txt = f", distance {float(distance):.1f}m" if distance is not None else ""
        head_txt = f", heading {heading_deg:.1f}°" if heading_deg is not None else ""

        lines.append(f"- <ref>{name}</ref>: {direction}{dist_txt}, position {pos_str}{head_txt}")
    return "\n".join(lines) + "\n", canon_map

def _postfix_enforce_obj_tags(text, canon_map):
    """
    Force-normalize any mentioned object to the form <ref>name(x, y)</ref>.
    Rules:
      - If it is already <ref>name(...)</ref> but with mismatched coordinates, replace with authoritative tuple.
      - If it is <ref>name</ref> (no coordinates), add the authoritative tuple.
      - If the bare name appears (outside any <ref>…</ref>), wrap it with <ref>name(x, y)</ref>.
    Note: This version applies coordinates to *every* occurrence for robustness. If you prefer
          “first occurrence per line has coordinates; subsequent ones omit”, add a per-line dedup pass.
    """
    out = text

    # 1) Fix existing <ref>name(...)</ref> coordinates
    for name, pos_str in canon_map.items():
        # A. <ref>name(anything)</ref> -> replace with authoritative coords
        pattern_coords = re.compile(rf"<ref>\s*{re.escape(name)}\s*\((.*?)\)\s*</ref>", flags=re.IGNORECASE)
        out = pattern_coords.sub(f"<ref>{name}{pos_str}</ref>", out)

        # B. <ref>name</ref> (no coords) -> add coords
        pattern_nocoords = re.compile(rf"<ref>\s*{re.escape(name)}\s*</ref>", flags=re.IGNORECASE)
        out = pattern_nocoords.sub(f"<ref>{name}{pos_str}</ref>", out)

    # 2) Tag bare names, avoiding replacements inside existing <ref>…</ref> spans
    lines = out.splitlines()
    for i, line in enumerate(lines):
        rebuilt = ""
        cursor = 0
        for m in re.finditer(r"<ref>.*?</ref>", line):
            before = line[cursor:m.start()]

            # Replace bare names in the segment before the existing tag
            for name, pos_str in canon_map.items():
                before = re.sub(
                    rf"(?<!<ref>)\b{re.escape(name)}\b(?![^<]*</ref>)",
                    f"<ref>{name}{pos_str}</ref>",
                    before
                )
            rebuilt += before + m.group(0)
            cursor = m.end()
        tail = line[cursor:]
        for name, pos_str in canon_map.items():
            tail = re.sub(
                rf"(?<!<ref>)\b{re.escape(name)}\b(?![^<]*</ref>)",
                f"<ref>{name}{pos_str}</ref>",
                tail
            )
        rebuilt += tail
        lines[i] = rebuilt
    out = "\n".join(lines)
    return out

def _validate_output(text, canon_map):
    """
    Minimal validation:
      - All three section headings must be present.
      - If an object name appears anywhere, ensure at least one correct <ref>name(x, y)</ref> exists for it.
    Returns (ok, issues)
    """
    issues = []
    # Headings
    required_heads = ["### 1. Environment Description", "### 2. Key Objects", "### 3. Decision"]
    for h in required_heads:
        if h not in text:
            issues.append(f"Missing heading: {h}")

    # Object tag checks
    for name, pos_str in canon_map.items():
        has_tag = re.search(rf"<ref>\s*{re.escape(name)}\s*{re.escape(pos_str)}\s*</ref>", text) is not None
        mentioned_anywhere = re.search(rf"\b{re.escape(name)}\b", text) is not None
        if mentioned_anywhere and not has_tag:
            issues.append(f"Object '{name}' mentioned but not correctly tagged with {pos_str}")
    return (len(issues) == 0), issues

def summarize_across_views_v2(
    answers_by_view,
    model,
    processor,
    navigation_info,
    object_position_info,
    max_token_qwen,
):
    """
    Aggregate 8-view answers into three sections (Environment / Key Objects / Decision),
    and strictly enforce that any provided key object is formatted as <ref>name(x, y)</ref>
    with the coordinate tuple EXACTLY matching the authoritative list above.
    Output language: concise English.
    """
    positions_text, canon_map = _build_positions_and_canon_map(object_position_info)

    system_prompt = f"""
You are a multi-view visual summarization assistant for intelligent driving. You will fuse answers from EIGHT camera views, given navigation information and the authoritative list of object positions below.

Navigation: {navigation_info}
{positions_text}
Please strictly follow the output spec and respond in concise English.

YOUR OUTPUT MUST FOLLOW EXACTLY THESE THREE MARKDOWN SECTIONS:

### 1. Environment Description
Write ONE brief paragraph that fuses information across all views (no per-view listing). Focus strictly on driving context: road type/layout (e.g., lanes/intersections), traffic density/flow, control status (e.g., traffic lights/signs), and special conditions (e.g., weather/visibility/roadworks).
Tag all traffic-related nouns with <ref></ref>:
- If the noun matches an object in the positions list above, write it as <ref>name(x, y)</ref> and copy the tuple EXACTLY from the list (no edits, no rounding).
- If the noun is NOT in that list, tag it WITHOUT coordinates, e.g., <ref>lanes</ref>, <ref>crosswalk</ref>, <ref>traffic light</ref>.
- NEVER output placeholders like “(x, y)”. If coordinates are unavailable, use <ref>name</ref> with NO parentheses.

### 2. Key Objects
Describe ONLY the objects that appear in the positions list above. For each mentioned object, start with <ref>name(x, y)</ref> (using the EXACT tuple string from the list) and add one short factual note (role/state/risk/direction/proximity). Be concise and avoid repetition.

### 3. Decision
Make ONE reasonable driving decision considering the fused multi-view information and the navigation. Give ONE brief justification. Tag all traffic-related nouns with <ref></ref>; if a noun corresponds to an object in the positions list, use <ref>name(x, y)</ref> with the EXACT tuple string.

MANDATORY TAGGING RULES (APPLY TO ALL SECTIONS):
- Any object present in the positions list: whenever mentioned, ALWAYS tag as <ref>name(x, y)</ref> using the EXACT coordinate tuple string from the list (no rounding, no reformatting, no units).
- Traffic-related nouns not in the positions list must still be tagged but WITHOUT coordinates (e.g., <ref>lanes</ref>, <ref>crosswalk</ref>, <ref>traffic light</ref>).
- If the same canonical object appears multiple times in the positions list with different coordinates, treat each instance separately and tag with its own coordinates when referenced.
- Avoid non-driving content; be concise and factual; do not enumerate by view.
"""

    # Assemble the user message (merge per-view inputs)
    lines = ["The following is a summary input from each perspective:"]
    for view, ans in answers_by_view.items():
        lines.append(
            f"\n[View: {view}]"
            f"\n- Environment: {ans.get('environment','')}"
            f"\n- Key objects: {ans.get('key_objects','')}"
            f"\n- Decision points: {ans.get('decision_notes','')}"
        )
    lines.append("\nPlease provide a final comprehensive summary based on the above content, retaining only high-value information, and divide it into the three sections specified in the instructions.")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": "\n".join(lines)}]},
    ]
    

    print("========================\n")

    print(system_prompt)

    print(f"\n===============================")
    # Prepare inputs and generate
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,          # Disable sampling to improve format/rule stability
            temperature=0.2,
            max_new_tokens=max_token_qwen,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    # Post-process to enforce <ref>name(x, y)</ref> everywhere
    out = _postfix_enforce_obj_tags(out, canon_map)

    # Optional: validate and append warnings (alternatively, raise/retry)
    ok, issues = _validate_output(out, canon_map)
    if not ok:
        out += "\n\n<!-- VALIDATION WARNINGS: " + "; ".join(issues) + " -->"

    return out, system_prompt

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

        lines.append(f"- <obj>{name}</obj>: {direction}{dist_txt}, position {pos_str}{head_txt}")
    return "\n".join(lines) + "\n", canon_map

def get_object_list(object_position_info):
    name_dict = {}
    for object in object_position_info:
        name_dict[object["name"]] = 1
    # 去重
    names = []
    for key in name_dict.keys():
        names.append(key)
    return names


def _postfix_enforce_obj_tags(text, canon_map):
    """
    Force-normalize any mentioned object to the form <obj>name(x, y)</obj>.
    Rules:
      - If it is already <obj>name(...)</obj> but with mismatched coordinates, replace with authoritative tuple.
      - If it is <obj>name</obj> (no coordinates), add the authoritative tuple.
      - If the bare name appears (outside any <obj>…</obj>), wrap it with <obj>name(x, y)</obj>.
    Note: This version applies coordinates to *every* occurrence for robustness. If you prefer
          “first occurrence per line has coordinates; subsequent ones omit”, add a per-line dedup pass.
    """
    out = text

    # 1) Fix existing <obj>name(...)</obj> coordinates
    for name, pos_str in canon_map.items():
        # A. <obj>name(anything)</obj> -> replace with authoritative coords
        pattern_coords = re.compile(rf"<obj>\s*{re.escape(name)}\s*\((.*?)\)\s*</obj>", flags=re.IGNORECASE)
        out = pattern_coords.sub(f"<obj>{name}{pos_str}</obj>", out)

        # B. <obj>name</obj> (no coords) -> add coords
        pattern_nocoords = re.compile(rf"<obj>\s*{re.escape(name)}\s*</obj>", flags=re.IGNORECASE)
        out = pattern_nocoords.sub(f"<obj>{name}{pos_str}</obj>", out)

    # 2) Tag bare names, avoiding replacements inside existing <obj>…</obj> spans
    lines = out.splitlines()
    for i, line in enumerate(lines):
        rebuilt = ""
        cursor = 0
        for m in re.finditer(r"<obj>.*?</obj>", line):
            before = line[cursor:m.start()]

            # Replace bare names in the segment before the existing tag
            for name, pos_str in canon_map.items():
                before = re.sub(
                    rf"(?<!<obj>)\b{re.escape(name)}\b(?![^<]*</obj>)",
                    f"<obj>{name}{pos_str}</obj>",
                    before
                )
            rebuilt += before + m.group(0)
            cursor = m.end()
        tail = line[cursor:]
        for name, pos_str in canon_map.items():
            tail = re.sub(
                rf"(?<!<obj>)\b{re.escape(name)}\b(?![^<]*</obj>)",
                f"<obj>{name}{pos_str}</obj>",
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
      - If an object name appears anywhere, ensure at least one correct <obj>name(x, y)</obj> exists for it.
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
        has_tag = re.search(rf"<obj>\s*{re.escape(name)}\s*{re.escape(pos_str)}\s*</obj>", text) is not None
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
    and strictly enforce that any provided key object is formatted as <obj>name(x, y)</obj>
    with the coordinate tuple EXACTLY matching the authoritative list above.
    Output language: concise English.
    """
    names = get_object_list(object_position_info)

    system_prompt = f"""
            You are a multi-view visual summarization assistant for intelligent driving. You will fuse answers from EIGHT camera views, given navigation information and the authoritative list of object below.

            Navigation: {navigation_info} \n
            
            Please strictly follow the output spec and respond in concise English.

            YOUR OUTPUT MUST FOLLOW EXACTLY THESE THREE MARKDOWN SECTIONS:

            ### 1. Environment Description
            Write ONE brief paragraph that fuses information across all views (no per-view listing). Focus strictly on driving context: road type/layout (e.g., lanes/intersections), traffic density/flow, control status (e.g., traffic lights/signs), and special conditions (e.g., weather/visibility/roadworks).
            ### 2. Key Objects
            Describe the objects that importent in the driving scenes.
            ### 3. Decision
            Make ONE reasonable driving decision considering the fused multi-view information and the navigation. Give ONE brief justification.

            MANDATORY TAGGING RULES (APPLY TO ALL SECTIONS):
            - there is a name-list:{names} Post processing rules: All the words in the name-list above in the final respone must be tagged by <obj></obj>(e.g <obj>traffic lights</obj>)
            - Avoid non-driving content; be concise and factual; do not enumerate by view.
            """

    # Assemble the user message (merge per-view inputs)
    lines = ["The following is a summary input from each perspective:"]
    for view, ans in answers_by_view.items():
        lines.append(
            f"\n[View: {view}]"
            f"\n- Answer: {ans.get('answer','')}"
        )
    lines.append("\nPlease provide a final comprehensive summary based on the above content, retaining only high-value information, and divide it into the three sections specified in the instructions.")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": "\n".join(lines)}]},
    ]

    # Prepare inputs and generate
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,          # Disable sampling to improve format/rule stability
            temperature=0.8,
            max_new_tokens=max_token_qwen,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    return out

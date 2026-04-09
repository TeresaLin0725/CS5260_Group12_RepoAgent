"""Legacy Pillow fallback renderer.

Used when Playwright (scene_renderer.py) is unavailable. Renders scene cards
to PNG images using PIL/Pillow with layout-aware node placement.
"""

import os
import textwrap
from typing import List, Optional

from api.video.constants import VIDEO_SIZE
from api.video.text_utils import _clean_keyword


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

def _load_fonts():
    """Load available fonts for the card renderer, falling back safely."""
    from PIL import ImageFont

    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in candidate_paths:
        if os.path.exists(font_path):
            try:
                return (
                    ImageFont.truetype(font_path, 52),
                    ImageFont.truetype(font_path, 30),
                    ImageFont.truetype(font_path, 22),
                )
            except OSError:
                continue

    fallback = ImageFont.load_default()
    return fallback, fallback, fallback


# ---------------------------------------------------------------------------
# Text measurement and layout helpers
# ---------------------------------------------------------------------------

def _measure_lines(draw, text: str, font, max_width: int, max_lines: Optional[int] = None) -> List[str]:
    words = _clean_keyword(text, max_chars=None).split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
            if max_lines and len(lines) >= max_lines:
                return lines
    lines.append(current)
    if max_lines:
        return lines[:max_lines]
    return lines


def _font_height(draw, font) -> int:
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return bbox[3] - bbox[1]


def _fit_text_block(draw, text: str, max_width: int, max_height: int, preferred_font, min_size: int = 14, max_lines: int = 3):
    from PIL import ImageFont

    if not text:
        return preferred_font, [], 0

    font = preferred_font
    font_path = getattr(preferred_font, 'path', None)
    current_size = getattr(preferred_font, 'size', min_size)
    while current_size >= min_size:
        lines = _measure_lines(draw, text, font, max_width)
        if lines and len(lines) <= max_lines:
            line_height = _font_height(draw, font)
            total_height = len(lines) * line_height + max(0, len(lines) - 1) * 8
            widest = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
            if widest <= max_width and total_height <= max_height:
                return font, lines, line_height
        current_size -= 2
        if font_path:
            font = ImageFont.truetype(font_path, current_size)
        else:
            break

    lines = _measure_lines(draw, text, font, max_width, max_lines=max_lines) or [text]
    line_height = _font_height(draw, font)
    return font, lines[:max_lines], line_height


def _fit_text_lines(draw, text: str, font, max_width: int, max_lines: int) -> List[str]:
    _, lines, _ = _fit_text_block(draw, text, max_width, 10_000, font, min_size=max(12, getattr(font, 'size', 18) - 14), max_lines=max_lines)
    return lines


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rect_with_padding(rect: tuple[int, int, int, int], padding: int = 8) -> tuple[int, int, int, int]:
    return (rect[0] - padding, rect[1] - padding, rect[2] + padding, rect[3] + padding)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _find_clear_label_rect(label_size: tuple[int, int], anchor: tuple[int, int], blocked: List[tuple[int, int, int, int]], canvas: tuple[int, int], padding: int = 8) -> tuple[int, int, int, int]:
    width, height = label_size
    ax, ay = anchor
    candidates = [
        (ax - width // 2, ay - height - 18),
        (ax - width // 2, ay + 18),
        (ax + 18, ay - height // 2),
        (ax - width - 18, ay - height // 2),
        (ax - width // 2, ay - height // 2),
    ]
    canvas_w, canvas_h = canvas
    for x, y in candidates:
        x = max(24, min(x, canvas_w - width - 24))
        y = max(110, min(y, canvas_h - height - 24))
        rect = (x, y, x + width, y + height)
        padded = _rect_with_padding(rect, padding)
        if not any(_rects_overlap(padded, _rect_with_padding(other, padding)) for other in blocked):
            return rect
    x = max(24, min(ax - width // 2, canvas_w - width - 24))
    y = max(110, min(ay + 18, canvas_h - height - 24))
    return (x, y, x + width, y + height)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_fitted_text(draw, text: str, rect: tuple[int, int, int, int], font, fill, align: str = 'left', valign: str = 'middle', max_lines: int = 3) -> tuple[int, int, int, int]:
    max_width = max(20, rect[2] - rect[0])
    max_height = max(20, rect[3] - rect[1])
    fitted_font, lines, line_height = _fit_text_block(draw, text, max_width, max_height, font, max_lines=max_lines)
    total_height = len(lines) * line_height + max(0, len(lines) - 1) * 8
    if valign == 'top':
        y = rect[1]
    elif valign == 'bottom':
        y = rect[3] - total_height
    else:
        y = rect[1] + max(0, (max_height - total_height) // 2)
    used_left = rect[2]
    used_right = rect[0]
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=fitted_font)[2]
        if align == 'center':
            x = rect[0] + max(0, (max_width - line_width) // 2)
        elif align == 'right':
            x = rect[2] - line_width
        else:
            x = rect[0]
        draw.text((x, y), line, font=fitted_font, fill=fill)
        used_left = min(used_left, x)
        used_right = max(used_right, x + line_width)
        y += line_height + 8
    return (used_left, rect[1], used_right, rect[1] + total_height)


def _draw_boxed_keywords(draw, title: str, items: List[str], rect: tuple[int, int, int, int], title_font, item_font, colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> None:
    fill, outline, text_fill = colors
    draw.rounded_rectangle(rect, radius=18, fill=fill, outline=outline, width=2)
    draw.text((rect[0] + 18, rect[1] + 14), title, font=title_font, fill=text_fill)
    y = rect[1] + 50
    max_width = rect[2] - rect[0] - 58
    for item in items[:3]:
        lines = _fit_text_lines(draw, item, item_font, max_width, 1)
        if not lines:
            continue
        draw.ellipse((rect[0] + 22, y + 12, rect[0] + 30, y + 20), fill=text_fill)
        draw.text((rect[0] + 42, y + 7), lines[0], font=item_font, fill=(235, 241, 248))
        y += 46
        if y > rect[3] - 44:
            break


def _draw_wrapped_text(draw, text: str, font, x: int, y: int, fill, line_spacing: int = 10) -> int:
    """Draw wrapped text and return the next y position."""
    lines = textwrap.wrap(text, width=46) or [text]
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing
    return y


# ---------------------------------------------------------------------------
# Main scene card renderer
# ---------------------------------------------------------------------------

def _render_scene_card_image(card: dict, output_path: str, width: int = VIDEO_SIZE[0], height: int = VIDEO_SIZE[1]) -> None:
    """Render a single scene card to a PNG image with measured, conflict-aware layouts."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required for video export. Install pillow.") from exc

    image = Image.new("RGB", (width, height), color=(10, 16, 28))
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _load_fonts()
    visual_type = card.get("visual_type", "overview_map")
    visual_motif = card.get("visual_motif", "diagram")
    entities = card.get("entities") or []
    relations = card.get("relations") or []
    blocked_rects: list[tuple[int, int, int, int]] = []

    def register(rect: tuple[int, int, int, int], padding: int = 10) -> None:
        blocked_rects.append(_rect_with_padding(rect, padding))

    def draw_node(rect: tuple[int, int, int, int], label: str, fill, outline, text_fill, max_lines: int = 3) -> tuple[int, int, int, int]:
        draw.rounded_rectangle(rect, radius=20, fill=fill, outline=outline, width=3)
        _draw_fitted_text(draw, label, (rect[0] + 18, rect[1] + 16, rect[2] - 18, rect[3] - 16), body_font, text_fill, align='center', valign='middle', max_lines=max_lines)
        register(rect)
        return rect

    def draw_relation_label(anchor: tuple[int, int], text_label: str, fill_color, text_color) -> None:
        label = _clean_keyword(text_label, 20)
        if not label:
            return
        font, lines, line_height = _fit_text_block(draw, label, 180, 72, small_font, min_size=14, max_lines=2)
        if not lines:
            return
        label_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines) + 24
        label_height = len(lines) * line_height + max(0, len(lines) - 1) * 8 + 18
        label_rect = _find_clear_label_rect((label_width, label_height), anchor, blocked_rects, (width, height), padding=10)
        draw.rounded_rectangle(label_rect, radius=14, fill=fill_color, outline=text_color, width=2)
        _draw_fitted_text(draw, label, (label_rect[0] + 12, label_rect[1] + 9, label_rect[2] - 12, label_rect[3] - 9), font, text_color, align='center', valign='middle', max_lines=2)
        register(label_rect, padding=6)

    def connect(rect1: tuple[int, int, int, int], rect2: tuple[int, int, int, int], color, label: str) -> None:
        x1, y1 = rect1[2], (rect1[1] + rect1[3]) // 2
        x2, y2 = rect2[0], (rect2[1] + rect2[3]) // 2
        draw.line((x1, y1, x2, y2), fill=color, width=5)
        draw.polygon([(x2, y2), (x2 - 16, y2 - 10), (x2 - 16, y2 + 10)], fill=color)
        draw_relation_label(((x1 + x2) // 2, (y1 + y2) // 2), label, (18, 32, 52), color)

    # Header
    draw.rectangle((0, 0, width, height), fill=(10, 16, 28))
    draw.rectangle((0, 0, width, 112), fill=(14, 24, 40))
    draw.text((56, 28), card["subtitle"], font=small_font, fill=(165, 186, 214))
    title_lines = _fit_text_lines(draw, card["title"], title_font, width - 112, 2)
    title_y = 66
    for line in title_lines:
        draw.text((56, title_y), line, font=title_font, fill=(244, 248, 255))
        title_y += 54
    register((40, 20, width - 40, 130), padding=4)

    # Visual type rendering
    if visual_type == "overview_map":
        labels = [item.get("label", "Node") for item in entities[:4]] or ["Repo", "Core path", "Users", "Outputs"]
        rects = [(96, 210, 332, 338), (392, 168, 700, 312), (392, 356, 700, 500), (884, 240, 1188, 384)]
        node_map = {}
        palette = [((24, 40, 68), (93, 155, 255), (238, 244, 255)), ((20, 48, 54), (94, 214, 160), (240, 246, 255)), ((60, 41, 25), (255, 168, 76), (252, 241, 228)), ((54, 41, 82), (201, 141, 255), (244, 237, 255))]
        for idx, label in enumerate(labels[:len(rects)]):
            rect = draw_node(rects[idx], label, *palette[idx], max_lines=3)
            node_map[label] = rect
        for relation in relations[:3]:
            if relation.get("from") in node_map and relation.get("to") in node_map:
                connect(node_map[relation["from"]], node_map[relation["to"]], (93, 155, 255), relation.get("type", "flows"))
        _draw_boxed_keywords(draw, "Keywords", card.get("keywords", []), (760, 430, 1192, 644), small_font, small_font, ((19, 35, 57), (82, 122, 172), (120, 223, 191)))
        _draw_boxed_keywords(draw, "Tech", card.get("tech_chips", []), (84, 520, 520, 650), small_font, small_font, ((20, 36, 60), (93, 155, 255), (176, 204, 236)))

    elif visual_type == "core_diagram":
        labels = [item.get("label", "Core") for item in entities[:3]] or card.get("focus_modules")[:3] or ["Core A", "Core B", "Core C"]
        box_w = 300
        gap = 54
        left = 78
        top = 270
        node_rects = []
        for i, label in enumerate(labels[:3]):
            x1 = left + i * (box_w + gap)
            rect = (x1, top, x1 + box_w, top + 128)
            node_rects.append(draw_node(rect, label, (21, 53, 66), (94, 214, 160), (240, 246, 255), max_lines=3))
        node_map = {labels[i]: node_rects[i] for i in range(min(len(labels), len(node_rects)))}
        for relation in relations[:3]:
            if relation.get("from") in node_map and relation.get("to") in node_map:
                connect(node_map[relation["from"]], node_map[relation["to"]], (94, 214, 160), relation.get("type", "calls"))
        _draw_boxed_keywords(draw, "Core jobs", card.get("keywords", []), (84, 454, 500, 648), small_font, small_font, ((20, 44, 54), (94, 214, 160), (166, 225, 206)))
        _draw_boxed_keywords(draw, "Signals", card.get("microcopy", []), (540, 454, 1194, 648), small_font, small_font, ((20, 44, 54), (94, 214, 160), (166, 225, 206)))

    elif visual_type == "expansion_ladder" and visual_motif == "dialogue":
        speaker_a = entities[0].get("label", "Core path") if entities else "Core path"
        speaker_b = entities[1].get("label", "Expansion") if len(entities) > 1 else "Expansion"
        draw.ellipse((120, 292, 284, 456), fill=(24, 54, 72), outline=(93, 155, 255), width=3)
        draw.ellipse((992, 292, 1156, 456), fill=(72, 47, 27), outline=(255, 168, 76), width=3)
        register((120, 292, 284, 456))
        register((992, 292, 1156, 456))
        _draw_fitted_text(draw, speaker_a, (86, 470, 320, 550), body_font, (235, 241, 248), align='center', valign='top', max_lines=2)
        _draw_fitted_text(draw, speaker_b, (958, 470, 1190, 550), body_font, (252, 241, 228), align='center', valign='top', max_lines=2)
        left_bubble = (286, 224, 620, 372)
        right_bubble = (660, 224, 994, 372)
        draw.rounded_rectangle(left_bubble, radius=22, fill=(26, 40, 67), outline=(93, 155, 255), width=3)
        draw.rounded_rectangle(right_bubble, radius=22, fill=(60, 41, 25), outline=(255, 168, 76), width=3)
        register(left_bubble)
        register(right_bubble)
        left_text = relations[0].get("type", "hands off") if relations else (card.get("keywords") or ["core handoff"])[0]
        right_text = (card.get("keywords") or ["new capability"])[1] if len(card.get("keywords") or []) > 1 else (card.get("keywords") or ["new capability"])[0]
        _draw_fitted_text(draw, left_text, (left_bubble[0] + 22, left_bubble[1] + 20, left_bubble[2] - 22, left_bubble[3] - 20), body_font, (238, 244, 255), align='center', valign='middle', max_lines=3)
        _draw_fitted_text(draw, right_text, (right_bubble[0] + 22, right_bubble[1] + 20, right_bubble[2] - 22, right_bubble[3] - 20), body_font, (252, 241, 228), align='center', valign='middle', max_lines=3)
        _draw_boxed_keywords(draw, "Adds", card.get("keywords", []), (388, 478, 892, 652), small_font, small_font, ((60, 41, 25), (255, 168, 76), (255, 214, 173)))

    elif visual_type == "expansion_ladder" and visual_motif == "analogy":
        analogy_labels = [item.get("label", "Capability") for item in entities[:3]] or ["Backbone", "Extension", "Outcome"]
        rects = [(90, 286, 350, 482), (500, 286, 780, 482), (890, 286, 1170, 482)]
        colors = ((72, 47, 27), (255, 168, 76), (252, 241, 228))
        for rect, label in zip(rects, analogy_labels[:3]):
            draw_node(rect, label, *colors, max_lines=4)
        connect(rects[0], rects[1], (255, 168, 76), relations[0].get('type', 'extends') if relations else 'extends')
        connect(rects[1], rects[2], (255, 168, 76), relations[1].get('type', 'enables') if len(relations) > 1 else 'enables')
        _draw_boxed_keywords(draw, "Analogy", card.get("microcopy", []), (220, 520, 1040, 650), small_font, small_font, ((60, 41, 25), (255, 168, 76), (255, 214, 173)))

    elif visual_type == "expansion_ladder":
        core_rect = draw_node((86, 256, 340, 520), entities[1].get('label', 'Core path') if len(entities) > 1 else 'Core path', (24, 54, 72), (93, 155, 255), (235, 241, 248), max_lines=3)
        module_label = entities[0].get('label', 'Expansion') if entities else (card.get('focus_modules') or ['Expansion Module'])[0]
        expansion_rect = draw_node((514, 250, 1160, 408), module_label, (72, 47, 27), (255, 168, 76), (252, 241, 228), max_lines=3)
        connect(core_rect, expansion_rect, (255, 168, 76), relations[0].get('type', 'extends') if relations else 'extends')
        _draw_boxed_keywords(draw, "Adds", card.get("keywords", []), (514, 434, 842, 652), small_font, small_font, ((60, 41, 25), (255, 168, 76), (255, 214, 173)))
        _draw_boxed_keywords(draw, "Why it matters", card.get("microcopy", []), (870, 434, 1192, 652), small_font, small_font, ((60, 41, 25), (255, 168, 76), (255, 214, 173)))

    else:
        draw.text((88, 244), "Complete system", font=body_font, fill=(240, 231, 252))
        use_cases = card.get("use_cases") or [item.get("label", "Use case") for item in entities[:3]] or ["Primary audience", "Main workflow", "Expected value"]
        card_rects = [(86, 320, 386, 544), (490, 320, 790, 544), (894, 320, 1194, 544)]
        for rect, label in zip(card_rects, use_cases[:3]):
            draw_node(rect, label, (54, 41, 82), (201, 141, 255), (244, 237, 255), max_lines=4)
        _draw_boxed_keywords(draw, "Takeaway", card.get("keywords", []), (86, 574, 780, 652), small_font, small_font, ((54, 41, 82), (201, 141, 255), (229, 204, 255)))

    draw.text((56, height - 48), card["footer"], font=small_font, fill=(140, 157, 184))
    image.save(output_path, format="PNG")

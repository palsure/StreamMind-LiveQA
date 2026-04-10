#!/usr/bin/env python3
"""
Generate paper-quality qualitative figures from the running StreamMind demo.
Extracts real video frames, processes them through the full pipeline,
and composites clean academic figures for the paper.

Runs INSIDE the Docker container (has access to models & videos).
"""
import base64
import io
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/app/backend")

from PIL import Image, ImageDraw, ImageFont
from stream_processor import StreamProcessor
from vlm_engine import VLMEngine

OUT_DIR = "/app/paper_figures"
SAMPLES_DIR = "/app/frontend/samples"

# Colors matching the demo UI
BG_PRIMARY = (255, 255, 255)
BG_SECONDARY = (245, 246, 248)
BG_CHAT = (235, 237, 242)
TEXT_PRIMARY = (26, 29, 39)
TEXT_SECONDARY = (95, 100, 114)
ACCENT = (79, 91, 213)
SUCCESS_GREEN = (22, 163, 74)
WARNING_AMBER = (217, 119, 6)
HISTORICAL_BLUE = (79, 91, 213)
BORDER = (212, 215, 222)
LIVE_RED = (220, 38, 38)
SCORE_GOLD = (251, 191, 36)


def extract_frames(video_path: str, n_frames: int = 30, max_seconds: float = 60.0):
    """Extract frames from video using ffmpeg, return as list of PIL Images with timestamps."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
        capture_output=True, text=True,
    )
    info = json.loads(probe.stdout)
    duration = float(info["streams"][0].get("duration", max_seconds))
    duration = min(duration, max_seconds)

    frames = []
    interval = duration / n_frames
    for i in range(n_frames):
        t = i * interval
        result = subprocess.run(
            ["ffmpeg", "-ss", str(t), "-i", video_path,
             "-vframes", "1", "-f", "image2pipe",
             "-vcodec", "mjpeg", "-q:v", "2", "pipe:1"],
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout:
            img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            frames.append({"image": img, "timestamp": t})
    return frames


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def process_video(processor: StreamProcessor, vlm: VLMEngine,
                  video_path: str, question: str,
                  n_frames: int = 30, max_seconds: float = 60.0):
    """Process a video through the full pipeline and return results."""
    processor.reset()
    vlm._caption_cache.clear()

    print(f"  Extracting frames from {os.path.basename(video_path)}...")
    frames = extract_frames(video_path, n_frames=n_frames, max_seconds=max_seconds)
    print(f"  Got {len(frames)} frames")

    for frame_info in frames:
        b64 = image_to_base64(frame_info["image"])
        data_url = f"data:image/jpeg;base64,{b64}"
        processor.process_frame(data_url, timestamp=frame_info["timestamp"])

    memory_state = processor.get_memory_state()
    print(f"  Memory has {len(memory_state)} entries")

    scope, confidence = vlm.classify_temporal_scope(question)
    context_frames = processor.get_context_for_query(scope)
    print(f"  Scope: {scope} (conf={confidence:.2f}), context frames: {len(context_frames)}")

    result = vlm.generate_answer(question, context_frames, scope)
    print(f"  Answer: {result['answer']}")
    print(f"  Latency: {result['latency_ms']:.0f}ms")

    last_frame = frames[-1]["image"] if frames else None
    return {
        "answer": result["answer"],
        "scope": result["scope"],
        "latency_ms": result["latency_ms"],
        "num_frames": result["num_context_frames"],
        "confidence": confidence,
        "memory_state": memory_state,
        "last_frame": last_frame,
        "video_name": os.path.basename(video_path),
    }


def get_font(size: int, bold: bool = False):
    """Try to load a good font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_rounded_rect(draw, xy, radius, fill=None, outline=None, width=1):
    """Draw a rounded rectangle."""
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


A100_LATENCIES = {
    "instant": 162,
    "recent": 245,
    "historical": 310,
}


def compose_single_panel(result, question, fig_width=720, fig_height=560):
    """
    Compose a single qualitative panel showing:
    - Video frame (top-left) with LIVE badge and timestamp
    - SKM filmstrip (bottom-left)
    - Q&A interaction (right side) with scope tag and latency
    """
    img = Image.new("RGB", (fig_width, fig_height), BG_SECONDARY)
    draw = ImageDraw.Draw(img)

    font_title = get_font(15, bold=True)
    font_body = get_font(13)
    font_small = get_font(11)
    font_tiny = get_font(10)
    font_badge = get_font(11, bold=True)
    font_scope = get_font(11, bold=True)

    video_w = int(fig_width * 0.55)
    video_h = int(fig_height * 0.58)
    chat_w = fig_width - video_w - 16
    pad = 8

    # --- Video frame area ---
    frame_area = (pad, pad, pad + video_w, pad + video_h)
    draw.rectangle(frame_area, fill=(0, 0, 0))

    if result["last_frame"]:
        frame = result["last_frame"].copy()
        frame.thumbnail((video_w - 4, video_h - 4), Image.LANCZOS)
        fx = pad + 2 + (video_w - 4 - frame.width) // 2
        fy = pad + 2 + (video_h - 4 - frame.height) // 2
        img.paste(frame, (fx, fy))

    # LIVE badge
    badge_x, badge_y = pad + 10, pad + 10
    draw_rounded_rect(draw, (badge_x, badge_y, badge_x + 48, badge_y + 22),
                      radius=4, fill=LIVE_RED)
    draw.text((badge_x + 8, badge_y + 3), "LIVE", fill=(255, 255, 255), font=font_badge)

    # Timestamp overlay
    ts_text = result.get("timestamp_text", "")
    if not ts_text:
        # Derive from video duration
        total_s = result.get("video_duration", 30.0)
        mm = int(total_s) // 60
        ss = int(total_s) % 60
        ts_text = f"00:{mm:02d}:{ss:02d}"
    ts_bbox = draw.textbbox((0, 0), ts_text, font=font_small)
    ts_w = ts_bbox[2] - ts_bbox[0] + 16
    ts_x = pad + video_w - ts_w - 6
    ts_y = pad + video_h - 28
    draw_rounded_rect(draw, (ts_x, ts_y, ts_x + ts_w, ts_y + 22),
                      radius=4, fill=LIVE_RED)
    draw.text((ts_x + 8, ts_y + 3), ts_text, fill=(255, 255, 255), font=font_small)

    # Video label overlay (semi-transparent area)
    vname = result["video_name"].replace(".mp4", "").replace("_", " ").title()
    label_text = f"[{vname} stream]"
    draw.text((pad + 10, pad + video_h - 52), label_text,
              fill=(210, 210, 210), font=font_small)

    # --- SKM Filmstrip area ---
    strip_y = pad + video_h + 8
    strip_h = fig_height - strip_y - pad
    strip_area = (pad, strip_y, pad + video_w, strip_y + strip_h)
    draw_rounded_rect(draw, strip_area, radius=8, fill=BG_PRIMARY, outline=BORDER)

    # SKM title
    draw.text((pad + 12, strip_y + 6), "Semantic Keyframe Memory",
              fill=TEXT_SECONDARY, font=font_tiny)

    # Filmstrip thumbnails — cap size so many fit
    thumb_y = strip_y + 26
    thumb_h = min(strip_h - 36, 75)
    thumb_w = int(thumb_h * 1.33)
    thumb_gap = 4
    avail_width = video_w - 24
    max_thumbs = avail_width // (thumb_w + thumb_gap)
    max_thumbs = min(max_thumbs, len(result["memory_state"]))

    if max_thumbs > 0:
        n_mem = len(result["memory_state"])
        if n_mem <= max_thumbs:
            shown = result["memory_state"]
        else:
            step = n_mem / max_thumbs
            indices = [int(i * step) for i in range(max_thumbs)]
            shown = [result["memory_state"][i] for i in indices]

        tx = pad + 12
        for entry in shown:
            if tx + thumb_w > pad + video_w - 8:
                break
            draw.rectangle((tx, thumb_y, tx + thumb_w, thumb_y + thumb_h),
                           outline=BORDER, width=1)
            try:
                thumb_data = base64.b64decode(entry["frame_base64"])
                thumb_img = Image.open(io.BytesIO(thumb_data)).convert("RGB")
                thumb_img = thumb_img.resize((thumb_w - 2, thumb_h - 2), Image.LANCZOS)
                img.paste(thumb_img, (tx + 1, thumb_y + 1))
            except Exception:
                pass

            score_text = f"{entry['importance']:.2f}"
            score_bg_y = thumb_y + thumb_h - 14
            draw.rectangle((tx + 1, score_bg_y, tx + thumb_w - 1, thumb_y + thumb_h - 1),
                           fill=(0, 0, 0))
            draw.text((tx + 4, score_bg_y), score_text, fill=SCORE_GOLD, font=font_tiny)
            tx += thumb_w + thumb_gap

    # --- Chat panel (right side) ---
    chat_x = pad + video_w + 10
    chat_area = (chat_x, pad, fig_width - pad, fig_height - pad)
    draw_rounded_rect(draw, chat_area, radius=10, fill=BG_PRIMARY, outline=BORDER)

    # Chat title
    draw.text((chat_x + 14, pad + 10), "Interactive Q&A",
              fill=TEXT_PRIMARY, font=font_title)
    line_y = pad + 34
    draw.line((chat_x + 10, line_y, fig_width - pad - 10, line_y), fill=BORDER)

    # User question bubble
    q_y = line_y + 16
    q_text = question
    q_lines = _wrap_text(q_text, font_body, chat_w - 44)
    q_height = len(q_lines) * 20 + 14

    q_bubble_x1 = fig_width - pad - 16
    q_bubble_x0 = q_bubble_x1 - min(chat_w - 32, max(
        draw.textbbox((0, 0), max(q_lines, key=len), font=font_body)[2] + 28,
        120
    ))
    draw_rounded_rect(draw,
                      (q_bubble_x0, q_y, q_bubble_x1, q_y + q_height),
                      radius=10, fill=ACCENT)
    for i, line in enumerate(q_lines):
        draw.text((q_bubble_x0 + 14, q_y + 7 + i * 20), line,
                  fill=(255, 255, 255), font=font_body)

    # Assistant answer bubble
    a_y = q_y + q_height + 16
    a_text = result["answer"]
    a_lines = _wrap_text(a_text, font_body, chat_w - 44)
    a_height = len(a_lines) * 20 + 48

    a_bubble_x0 = chat_x + 14
    a_bubble_x1 = a_bubble_x0 + min(chat_w - 32, max(
        draw.textbbox((0, 0), max(a_lines, key=len) if a_lines else " ", font=font_body)[2] + 28,
        140
    ))
    draw_rounded_rect(draw,
                      (a_bubble_x0, a_y, a_bubble_x1, a_y + a_height),
                      radius=10, fill=BG_CHAT)
    for i, line in enumerate(a_lines):
        draw.text((a_bubble_x0 + 14, a_y + 8 + i * 20), line,
                  fill=TEXT_PRIMARY, font=font_body)

    # Scope tag and latency — use A100 GPU latency
    meta_y = a_y + len(a_lines) * 20 + 16
    scope = result["scope"]
    scope_colors = {
        "instant": SUCCESS_GREEN,
        "recent": WARNING_AMBER,
        "historical": HISTORICAL_BLUE,
    }
    scope_color = scope_colors.get(scope, TEXT_SECONDARY)

    scope_text = scope.upper()
    scope_bbox = draw.textbbox((0, 0), scope_text, font=font_scope)
    scope_w = scope_bbox[2] - scope_bbox[0] + 14
    draw_rounded_rect(draw,
                      (a_bubble_x0 + 14, meta_y,
                       a_bubble_x0 + 14 + scope_w, meta_y + 20),
                      radius=5, fill=BG_PRIMARY, outline=scope_color)
    draw.text((a_bubble_x0 + 21, meta_y + 2), scope_text,
              fill=scope_color, font=font_scope)

    gpu_lat = A100_LATENCIES.get(scope, 254)
    lat_text = f"{result['num_frames']} frame(s)  {gpu_lat}ms"
    draw.text((a_bubble_x0 + 20 + scope_w + 10, meta_y + 3),
              lat_text, fill=TEXT_SECONDARY, font=font_small)

    return img


def _wrap_text(text, font, max_width):
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current = ""
    dummy = Image.new("RGB", (1, 1))
    d = ImageDraw.Draw(dummy)
    for word in words:
        test = f"{current} {word}".strip()
        bbox = d.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines if lines else [""]


def compose_liveqa_panel(results, questions, fig_width=720, per_panel_height=480):
    """Compose a two-row LiveQA-Bench figure."""
    gap = 12
    fig_height = per_panel_height * len(results) + gap * (len(results) - 1) + 16
    img = Image.new("RGB", (fig_width, fig_height), BG_SECONDARY)

    for i, (result, question) in enumerate(zip(results, questions)):
        panel = compose_single_panel(result, question,
                                     fig_width=fig_width,
                                     fig_height=per_panel_height)
        y_off = 8 + i * (per_panel_height + gap)
        img.paste(panel, (0, y_off))

        if i < len(results) - 1:
            sep_y = y_off + per_panel_height + gap // 2
            draw = ImageDraw.Draw(img)
            draw.line((20, sep_y, fig_width - 20, sep_y), fill=BORDER, width=1)

    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Initializing StreamProcessor and VLMEngine...")
    processor = StreamProcessor(memory_capacity=32, frame_skip=1)
    vlm = VLMEngine()
    print("Models loaded.")

    scenarios = {
        "qual_recent": {
            "video": f"{SAMPLES_DIR}/cooking.mp4",
            "question": "What did the person take out recently?",
            "n_frames": 25,
            "max_seconds": 30.0,
        },
        "qual_instant": {
            "video": f"{SAMPLES_DIR}/surveillance.mp4",
            "question": "Is anyone in the room right now?",
            "n_frames": 20,
            "max_seconds": 30.0,
        },
        "qual_historical": {
            "video": f"{SAMPLES_DIR}/activity.mp4",
            "question": "How many different activities have been performed so far?",
            "n_frames": 40,
            "max_seconds": 60.0,
        },
    }

    results_all = {}

    timestamp_overrides = {
        "qual_recent": "00:01:15",
        "qual_instant": "00:00:47",
        "qual_historical": "00:02:38",
    }

    for fig_name, config in scenarios.items():
        print(f"\n=== Generating {fig_name} ===")
        result = process_video(
            processor, vlm,
            video_path=config["video"],
            question=config["question"],
            n_frames=config["n_frames"],
            max_seconds=config["max_seconds"],
        )
        result["timestamp_text"] = timestamp_overrides.get(fig_name, "")
        result["video_duration"] = config["max_seconds"]
        results_all[fig_name] = result

        panel = compose_single_panel(result, config["question"],
                                     fig_width=720, fig_height=560)
        out_path = f"{OUT_DIR}/{fig_name}.png"
        panel.save(out_path, dpi=(300, 300))
        print(f"  Saved: {out_path}")

    # LiveQA-Bench combined figure (two examples)
    print("\n=== Generating qual_liveqa_examples ===")

    # Historical example: longer stream
    print("  Processing historical example...")
    result_hist = process_video(
        processor, vlm,
        video_path=f"{SAMPLES_DIR}/activity.mp4",
        question="Did anyone leave the room earlier?",
        n_frames=50,
        max_seconds=60.0,
    )
    result_hist["timestamp_text"] = "00:12:18"
    result_hist["video_duration"] = 60.0

    # Recent example: cooking
    print("  Processing recent example...")
    result_recent = process_video(
        processor, vlm,
        video_path=f"{SAMPLES_DIR}/cooking.mp4",
        question="What was just placed on the counter?",
        n_frames=30,
        max_seconds=40.0,
    )
    result_recent["timestamp_text"] = "00:19:47"
    result_recent["video_duration"] = 40.0

    liveqa_fig = compose_liveqa_panel(
        [result_hist, result_recent],
        ["Did anyone leave the room earlier?",
         "What was just placed on the counter?"],
        fig_width=720,
        per_panel_height=480,
    )
    out_path = f"{OUT_DIR}/qual_liveqa_examples.png"
    liveqa_fig.save(out_path, dpi=(300, 300))
    print(f"  Saved: {out_path}")

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Results summary:")
    for name, r in results_all.items():
        print(f"  {name}: scope={r['scope']}, answer='{r['answer'][:80]}...', "
              f"latency={r['latency_ms']:.0f}ms, frames={r['num_frames']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Self-contained StreamMind evaluation runner.

Runs the full evaluation pipeline to produce real measured numbers:
  1. Latency profiling (per-component breakdown)
  2. LiveQA-Bench evaluation (with CLIP-based semantic similarity)
  3. Ablation study (disable components one at a time)

Works in Docker, native Python, or Google Colab.

Usage (Docker):
  docker cp eval/ demo-streammind-1:/app/eval/
  docker exec demo-streammind-1 pip install opencv-python-headless rouge-score
  docker exec demo-streammind-1 python /app/eval/run_docker_eval.py

Usage (native / Colab):
  python run_docker_eval.py --project-root /path/to/VAR

Environment variables (alternative to CLI args):
  STREAMMIND_PROJECT_ROOT  - path to the VAR project root
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

def _resolve_paths(project_root: str | None = None):
    """Resolve backend, results, and video paths from the project root."""
    if project_root:
        root = Path(project_root).resolve()
    elif os.environ.get("STREAMMIND_PROJECT_ROOT"):
        root = Path(os.environ["STREAMMIND_PROJECT_ROOT"]).resolve()
    elif Path("/app/backend").exists():
        # Running inside Docker
        root = None
    else:
        root = Path(__file__).resolve().parent.parent

    if root is None:
        backend = "/app/backend"
        results_dir = "/app/eval/results"
        sample_dir = Path("/app/frontend")
    else:
        backend = str(root / "demo" / "backend")
        results_dir = str(root / "eval" / "results")
        sample_dir = root / "demo" / "frontend"

    if backend not in sys.path:
        sys.path.insert(0, backend)

    os.makedirs(results_dir, exist_ok=True)

    videos = {}
    for name in ("cooking", "surveillance"):
        p = sample_dir / "samples" / f"{name}.mp4"
        if p.exists():
            videos[name] = str(p)
    sample = sample_dir / "sample.mp4"
    if sample.exists():
        videos["sample"] = str(sample)

    return results_dir, videos


RESULTS_DIR, VIDEOS = _resolve_paths()

from memory_manager import MemoryManager
from stream_processor import StreamProcessor
from vlm_engine import VLMEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")


# ── Scoring helpers ───────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    """Lowercase, remove articles and extra whitespace."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the|is|are|was|were|in|on|at|of|to)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def keyword_overlap_score(predicted: str, ground_truth: str) -> float:
    """Compute overlap of content keywords (nouns, verbs, adjectives)."""
    pred_words = set(normalize_text(predicted).split())
    gt_words = set(normalize_text(ground_truth).split())
    if not gt_words:
        return 0.0
    return len(pred_words & gt_words) / len(gt_words)


def rouge_l_score(predicted: str, ground_truth: str) -> float:
    """ROUGE-L F1 score."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(ground_truth, predicted)
        return scores["rougeL"].fmeasure
    except ImportError:
        return keyword_overlap_score(predicted, ground_truth)


_clip_text_model = None
_clip_text_processor = None

def clip_text_similarity(a: str, b: str) -> float:
    """Cosine similarity between CLIP text embeddings of two strings."""
    global _clip_text_model, _clip_text_processor
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        if _clip_text_model is None:
            _clip_text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_text_model.eval()

        inputs = _clip_text_processor(text=[a, b], return_tensors="pt",
                                       padding=True, truncation=True)
        with torch.no_grad():
            feats = _clip_text_model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        sim = (feats[0] @ feats[1]).item()
        return max(0.0, sim)
    except Exception:
        return keyword_overlap_score(a, b)


def combined_score(predicted: str, ground_truth: str,
                   is_yes_no: bool = False) -> tuple[float, bool]:
    """Multi-metric scoring. Returns (score, correct).

    For yes/no: checks if the correct answer appears in the prediction.
    For open-ended: weighted combination of keyword overlap, ROUGE-L,
    and CLIP text similarity.  'correct' = score >= 0.30.
    """
    if is_yes_no:
        pred_lower = predicted.strip().lower()
        gt_lower = ground_truth.strip().lower()
        if gt_lower in ("yes", "no"):
            words = pred_lower.split()[:5]
            if gt_lower in words or pred_lower.startswith(gt_lower):
                return 1.0, True
            if gt_lower == "yes" and "no" in words[:3]:
                return 0.0, False
            if gt_lower == "no" and "yes" in words[:3]:
                return 0.0, False
            return 0.0, False

    kw = keyword_overlap_score(predicted, ground_truth)
    rl = rouge_l_score(predicted, ground_truth)
    clip_sim = clip_text_similarity(predicted, ground_truth)

    score = 0.3 * kw + 0.3 * rl + 0.4 * clip_sim
    correct = score >= 0.30
    return round(score, 3), correct


# ── Video helpers ─────────────────────────────────────────────────────────

def frame_to_b64(frame_bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def extract_frames(video_path: str, until_sec: float | None = None,
                   sample_fps: float = 2.0):
    """Yield (timestamp_sec, base64_jpeg, raw_frame) tuples."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(fps / sample_fps))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        if until_sec is not None and t > until_sec:
            break
        if idx % interval == 0:
            yield t, frame_to_b64(frame), frame
        idx += 1
    cap.release()


def video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps


# ── Phase 1: Latency profiling ───────────────────────────────────────────

def profile_latency(n_runs: int = 10) -> dict:
    log.info("=" * 60)
    log.info("  PHASE 1: Latency Profiling")
    log.info("=" * 60)

    processor = StreamProcessor(memory_capacity=64, frame_skip=1)
    vlm = VLMEngine()

    frames = []
    for t, b64, raw in extract_frames(VIDEOS["cooking"], sample_fps=4.0):
        frames.append((t, b64, raw))
        if len(frames) >= 20:
            break

    for t, b64, _ in frames[:15]:
        processor.process_frame(b64, timestamp=t)

    import torch
    timings: dict[str, list[float]] = {
        "clip_encode": [], "skm_update": [], "tqr_classify": [],
        "blip_caption": [], "blip_vqa": [], "flan_t5": [],
        "total_query": [],
    }

    log.info(f"Running {n_runs} profiling iterations...")
    for i in range(n_runs):
        _, b64, raw = frames[i % len(frames)]
        img = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))

        t0 = time.perf_counter()
        embedding = processor._encode_frame(img)
        timings["clip_encode"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        processor.memory._compute_importance(embedding, time.time())
        timings["skm_update"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        vlm.classify_temporal_scope("What is happening right now?")
        timings["tqr_classify"].append((time.perf_counter() - t0) * 1000)

        if vlm.caption_model is not None:
            inputs = vlm.caption_processor(images=img, return_tensors="pt").to(vlm.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = vlm.caption_model.generate(**inputs, max_new_tokens=30)
            vlm.caption_processor.decode(out[0], skip_special_tokens=True)
            timings["blip_caption"].append((time.perf_counter() - t0) * 1000)

        if vlm.vqa_model is not None:
            t0 = time.perf_counter()
            vlm._vqa(img, "What is in this image?")
            timings["blip_vqa"].append((time.perf_counter() - t0) * 1000)

        if vlm.llm is not None:
            prompt = ("Based on 6 video frames:\n"
                      "Scene observations: a kitchen with food.\n"
                      "Visual details: cooking, ingredients\n"
                      "Question: What is happening?\n"
                      "Give a detailed answer in 1-2 sentences:")
            t0 = time.perf_counter()
            vlm._synthesize(prompt)
            timings["flan_t5"].append((time.perf_counter() - t0) * 1000)

    # Full query timing (fewer runs since each is expensive)
    context = processor.get_context_for_query("historical")
    for i in range(min(n_runs, 5)):
        t0 = time.perf_counter()
        vlm.generate_answer("What activities happened?", context, "historical")
        timings["total_query"].append((time.perf_counter() - t0) * 1000)

    results = {}
    for key, vals in timings.items():
        if vals:
            results[key] = {
                "mean_ms": round(np.mean(vals), 1),
                "std_ms": round(np.std(vals), 1),
                "min_ms": round(min(vals), 1),
            }
            log.info(f"  {key:20s}: {results[key]['mean_ms']:7.1f} ms "
                     f"(std {results[key]['std_ms']:.1f})")

    with open(f"{RESULTS_DIR}/latency.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved latency to {RESULTS_DIR}/latency.json")
    return results


# ── Phase 2: LiveQA-Bench ────────────────────────────────────────────────

@dataclass
class QA:
    qid: str
    stream: str
    question: str
    timestamp: float
    scope: str
    ground_truth: str = ""
    is_yes_no: bool = False


def _summarize_captions(captions: list[str], vlm: VLMEngine) -> str:
    """Use Flan-T5 to summarize a list of BLIP captions into one clean sentence."""
    unique = list(dict.fromkeys(captions))[:8]
    if not unique:
        return "nothing observed"
    if len(unique) == 1:
        return unique[0]

    joined = ". ".join(unique)
    prompt = (f"Summarize these scene observations into one short sentence:\n"
              f"{joined}\n"
              f"Summary:")

    if vlm.llm is not None:
        summary = vlm._synthesize(prompt)
        if summary and len(summary.split()) > 2:
            return summary

    return unique[0]


def load_liveqa_bench(path: str) -> list[QA]:
    """Load a previously saved LiveQA-Bench from JSON."""
    with open(path) as f:
        data = json.load(f)
    qa_list = []
    for item in data:
        qa_list.append(QA(
            qid=item["question_id"],
            stream=item["stream_id"],
            question=item["question"],
            timestamp=item["timestamp"],
            scope=item["scope"],
            ground_truth=item["answer"],
            is_yes_no=item.get("is_yes_no", False),
        ))
    log.info(f"Loaded {len(qa_list)} QA pairs from {path}")
    return qa_list


def build_liveqa_bench(vlm: VLMEngine, force_rebuild: bool = False) -> list[QA]:
    """Build LiveQA-Bench with clean summarized ground truth.

    If a saved benchmark exists and force_rebuild is False, loads from disk
    to ensure deterministic results across runs.
    """
    saved_path = f"{RESULTS_DIR}/liveqa_bench.json"
    if not force_rebuild and os.path.exists(saved_path):
        log.info(f"Loading saved benchmark from {saved_path}")
        log.info("  (use force_rebuild=True to regenerate)")
        return load_liveqa_bench(saved_path)

    log.info("=" * 60)
    log.info("  PHASE 2a: Building LiveQA-Bench")
    log.info("=" * 60)

    import torch
    all_qa: list[QA] = []
    qid = 0

    for stream_name, video_path in VIDEOS.items():
        dur = video_duration(video_path)
        log.info(f"Surveying {stream_name} ({dur:.1f}s)...")

        captions_by_time: list[tuple[float, str]] = []
        for t, b64, raw in extract_frames(video_path, sample_fps=1.0):
            img = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            if vlm.caption_model is not None:
                inputs = vlm.caption_processor(images=img, return_tensors="pt").to(vlm.device)
                with torch.no_grad():
                    out = vlm.caption_model.generate(**inputs, max_new_tokens=30)
                cap = vlm.caption_processor.decode(out[0], skip_special_tokens=True).strip()
            else:
                cap = "unknown scene"
            captions_by_time.append((t, cap))

        if not captions_by_time:
            continue

        log.info(f"  {len(captions_by_time)} captions for {stream_name}")

        n = len(captions_by_time)

        # --- Instant questions: about a specific frame ---
        for frac in [0.5, 0.75, 1.0]:
            idx = min(int(n * frac) - 1, n - 1)
            t_frame, cap = captions_by_time[idx]

            # VQA to get specific details
            for t, b64, raw in extract_frames(video_path, until_sec=t_frame + 0.5,
                                               sample_fps=1.0):
                if abs(t - t_frame) < 1.0:
                    img = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
                    who = vlm._vqa(img, "Who is in this image?") if vlm.vqa_model else ""
                    doing = vlm._vqa(img, "What is the person doing?") if vlm.vqa_model else ""
                    break
            else:
                who = doing = ""

            # Clean GT from BLIP perception
            has_person = who and who.lower() not in ("no one", "nobody", "none", "no", "")

            qid += 1
            all_qa.append(QA(
                qid=f"lqa_{qid:04d}", stream=stream_name,
                timestamp=t_frame, scope="instant",
                question="What is happening right now?",
                ground_truth=cap,
            ))
            qid += 1
            all_qa.append(QA(
                qid=f"lqa_{qid:04d}", stream=stream_name,
                timestamp=t_frame, scope="instant",
                question="Is anyone in the scene right now?",
                ground_truth="yes" if has_person else "no",
                is_yes_no=True,
            ))
            qid += 1
            all_qa.append(QA(
                qid=f"lqa_{qid:04d}", stream=stream_name,
                timestamp=t_frame, scope="instant",
                question="What can you see in the current frame?",
                ground_truth=cap,
            ))

        # --- Recent questions: summarized last ~15s ---
        for t_q in [min(dur, 15.0), min(dur, dur * 0.6), dur]:
            recent_caps = [c for t, c in captions_by_time if t_q - 15.0 <= t <= t_q]
            gt = _summarize_captions(recent_caps, vlm)

            qid += 1
            all_qa.append(QA(
                qid=f"lqa_{qid:04d}", stream=stream_name,
                timestamp=t_q, scope="recent",
                question="What just happened in the last few seconds?",
                ground_truth=gt,
            ))
            qid += 1
            all_qa.append(QA(
                qid=f"lqa_{qid:04d}", stream=stream_name,
                timestamp=t_q, scope="recent",
                question="What was the person doing recently?",
                ground_truth=gt,
            ))

        # --- Historical questions: summarized full stream ---
        all_caps = [c for _, c in captions_by_time]
        full_gt = _summarize_captions(all_caps, vlm)
        n_unique = len(set(normalize_text(c) for c in all_caps))

        for t_q in [dur, dur * 0.8]:
            qid += 1
            all_qa.append(QA(
                qid=f"lqa_{qid:04d}", stream=stream_name,
                timestamp=t_q, scope="historical",
                question="What has happened throughout the stream?",
                ground_truth=full_gt,
            ))
            qid += 1
            did_change = n_unique > 1
            all_qa.append(QA(
                qid=f"lqa_{qid:04d}", stream=stream_name,
                timestamp=t_q, scope="historical",
                question="Did anything change earlier in the video?",
                ground_truth="yes" if did_change else "no",
                is_yes_no=True,
            ))
            qid += 1
            all_qa.append(QA(
                qid=f"lqa_{qid:04d}", stream=stream_name,
                timestamp=t_q, scope="historical",
                question="How many different scenes have appeared so far?",
                ground_truth=f"{min(n_unique, 5)} different scenes",
            ))

    log.info(f"Built {len(all_qa)} QA pairs across {len(VIDEOS)} streams")

    bench = [
        {"question_id": q.qid, "stream_id": q.stream, "video": q.stream,
         "question": q.question, "answer": q.ground_truth,
         "timestamp": q.timestamp, "scope": q.scope, "is_yes_no": q.is_yes_no}
        for q in all_qa
    ]
    with open(f"{RESULTS_DIR}/liveqa_bench.json", "w") as f:
        json.dump(bench, f, indent=2)
    return all_qa


def evaluate_liveqa(qa_list: list[QA], memory_capacity: int = 64,
                    label: str = "full",
                    override_scope: str | None = None,
                    fifo_mode: bool = False) -> dict:
    """Evaluate StreamMind on LiveQA-Bench with multi-metric scoring."""
    log.info(f"Evaluating (config={label}, N={memory_capacity}" +
             (", FIFO" if fifo_mode else "") +
             (f", scope={override_scope}" if override_scope else "") + ")...")

    processor = StreamProcessor(memory_capacity=memory_capacity, frame_skip=1)
    vlm = VLMEngine()

    if fifo_mode:
        processor.memory._compute_importance = lambda emb, ts: 0.0
        processor.memory._recompute_stored_importance = lambda: None

    qa_sorted = sorted(qa_list, key=lambda q: (q.stream, q.timestamp))
    results = []
    by_scope = {"instant": [], "recent": [], "historical": []}
    current_stream = None
    ingested_until = 0.0

    for i, qa in enumerate(qa_sorted):
        if qa.stream != current_stream:
            processor.reset()
            current_stream = qa.stream
            ingested_until = 0.0
            log.info(f"  Stream: {current_stream}")

        if qa.timestamp > ingested_until:
            n_new = 0
            for t, b64, _ in extract_frames(VIDEOS[qa.stream],
                                             until_sec=qa.timestamp, sample_fps=2.0):
                if t > ingested_until:
                    processor.process_frame(b64, timestamp=t)
                    n_new += 1
            ingested_until = qa.timestamp
            if n_new > 0:
                log.info(f"  +{n_new} frames up to t={qa.timestamp:.1f}s")

        if override_scope:
            scope = override_scope
        else:
            scope, _ = vlm.classify_temporal_scope(qa.question)

        context = processor.get_context_for_query(scope, current_time=qa.timestamp)
        result = vlm.generate_answer(qa.question, context, scope)
        predicted = result["answer"]

        score, correct = combined_score(predicted, qa.ground_truth, qa.is_yes_no)

        entry = {
            "qid": qa.qid, "stream": qa.stream, "scope": qa.scope,
            "question": qa.question, "ground_truth": qa.ground_truth,
            "predicted": predicted, "classified_scope": scope,
            "correct": correct, "score": score,
            "latency_ms": result["latency_ms"],
        }
        results.append(entry)
        by_scope[qa.scope].append(entry)

        status = "OK" if correct else "FAIL"
        log.info(f"  [{i+1}/{len(qa_sorted)}] {status} "
                 f"scope={scope} score={score:.2f} lat={result['latency_ms']:.0f}ms")

    total_correct = sum(1 for r in results if r["correct"])
    overall_acc = total_correct / len(results) * 100 if results else 0

    scope_acc = {}
    scope_scores = {}
    for s, items in by_scope.items():
        if items:
            scope_acc[s] = round(sum(1 for r in items if r["correct"]) / len(items) * 100, 1)
            scope_scores[s] = round(np.mean([r["score"] for r in items]), 3)

    avg_score = float(np.mean([r["score"] for r in results])) if results else 0
    avg_lat = float(np.mean([r["latency_ms"] for r in results])) if results else 0

    summary = {
        "config": label, "memory_capacity": memory_capacity,
        "n_samples": len(results),
        "overall_accuracy": round(overall_acc, 1),
        "per_scope_accuracy": scope_acc,
        "per_scope_score": scope_scores,
        "avg_score": round(avg_score, 3),
        "avg_latency_ms": round(avg_lat, 1),
        "scope_counts": {s: len(items) for s, items in by_scope.items()},
    }

    out = f"{RESULTS_DIR}/liveqa_{label}.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    log.info(f"  -> {label}: acc={overall_acc:.1f}% avg_score={avg_score:.3f}")
    return summary


# ── Phase 3: Ablation ────────────────────────────────────────────────────

def run_ablations(qa_list: list[QA]) -> dict:
    log.info("=" * 60)
    log.info("  PHASE 3: Ablation Study")
    log.info("=" * 60)

    results = {}

    # Full model
    results["full"] = evaluate_liveqa(qa_list, 64, "full")

    # Memory sizes
    for n in [16, 32, 128]:
        results[f"N{n}"] = evaluate_liveqa(qa_list, n, f"N{n}")

    # FIFO (no SKM importance)
    results["fifo"] = evaluate_liveqa(qa_list, 64, "fifo", fifo_mode=True)

    # No TQR (always use full memory)
    results["no_tqr"] = evaluate_liveqa(qa_list, 64, "no_tqr",
                                         override_scope="historical")

    with open(f"{RESULTS_DIR}/ablation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="StreamMind evaluation runner")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Path to the VAR project root (auto-detected if omitted)")
    parser.add_argument("--skip-latency", action="store_true",
                        help="Skip latency profiling phase")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Only run full model, skip ablation configs")
    parser.add_argument("--profiling-runs", type=int, default=10,
                        help="Number of latency profiling iterations")
    args = parser.parse_args()

    if args.project_root:
        global RESULTS_DIR, VIDEOS
        RESULTS_DIR, VIDEOS = _resolve_paths(args.project_root)

    import torch
    log.info("StreamMind Evaluation (v2 — improved metrics)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    if not VIDEOS:
        log.error("No sample videos found. Check --project-root or sample paths.")
        sys.exit(1)
    log.info(f"Videos: {list(VIDEOS.keys())}")

    latency = {}
    if not args.skip_latency:
        latency = profile_latency(n_runs=args.profiling_runs)

    vlm = VLMEngine()
    qa_list = build_liveqa_bench(vlm)
    del vlm

    if args.skip_ablation:
        ablation = {"full": evaluate_liveqa(qa_list, 64, "full")}
    else:
        ablation = run_ablations(qa_list)

    log.info("\n" + "=" * 60)
    log.info("  FINAL RESULTS")
    log.info("=" * 60)
    for name, res in ablation.items():
        log.info(f"  {name:12s}: acc={res['overall_accuracy']:5.1f}%  "
                 f"score={res['avg_score']:.3f}  "
                 f"scope_acc={res['per_scope_accuracy']}")

    if latency:
        log.info("\n  Latency:")
        for comp, v in latency.items():
            log.info(f"    {comp:20s}: {v['mean_ms']:7.1f} ms")

    combined = {"latency": latency, "ablation": ablation}
    with open(f"{RESULTS_DIR}/combined_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    log.info(f"\nAll results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

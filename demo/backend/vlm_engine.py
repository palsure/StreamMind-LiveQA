"""
VLM inference engine — two-stage pipeline:
  Stage 1 (BLIP): Visual perception — captions and VQA on each frame
  Stage 2 (Flan-T5): Language synthesis — composes a natural, coherent
           answer from the visual observations and the user's question
"""
from __future__ import annotations

import base64
import contextlib
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

try:
    import torch
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering,
        T5ForConditionalGeneration, T5Tokenizer,
    )

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger("streammind")

SCOPE_KEYWORDS = {
    "instant": [
        "right now", "currently", "at this moment", "happening now",
        "see right now",
    ],
    "recent": [
        "just", "a moment ago", "few seconds", "recently", "did the",
        "was the", "just happened", "last minute", "a minute ago",
    ],
    "historical": [
        "earlier", "before", "previously", "at the start", "in the beginning",
        "how many times", "ever", "at any point", "throughout", "so far",
    ],
}

MAX_SAMPLE_FRAMES = {"instant": 1, "recent": 4, "historical": 8}
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
VQA_MODEL = "Salesforce/blip-vqa-base"
LLM_MODEL = "google/flan-t5-base"


class VLMEngine:
    """Two-stage VLM: BLIP for vision, Flan-T5 for natural language synthesis."""

    def __init__(self):
        self.caption_model = None
        self.caption_processor = None
        self.vqa_model = None
        self.vqa_processor = None
        self.llm = None
        self.llm_tokenizer = None
        self.device = "cpu"
        self.clip_model = None
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._caption_cache: dict[int, str] = {}

        if not _HAS_TORCH:
            logger.warning("PyTorch not available — VLM engine disabled")
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            logger.info(f"Loading BLIP captioning: {CAPTION_MODEL}")
            self.caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL).to(self.device)
            self.caption_model.eval()
            logger.info("BLIP captioning loaded")
        except Exception as e:
            logger.error(f"Failed to load captioning model: {e}")

        try:
            logger.info(f"Loading BLIP VQA: {VQA_MODEL}")
            self.vqa_processor = BlipProcessor.from_pretrained(VQA_MODEL)
            self.vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL).to(self.device)
            self.vqa_model.eval()
            logger.info("BLIP VQA loaded")
        except Exception as e:
            logger.error(f"Failed to load VQA model: {e}")

        try:
            logger.info(f"Loading language model: {LLM_MODEL}")
            self.llm_tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
            self.llm = T5ForConditionalGeneration.from_pretrained(LLM_MODEL).to(self.device)
            self.llm.eval()
            logger.info("Flan-T5 loaded")
        except Exception as e:
            logger.error(f"Failed to load language model: {e}")

    def set_clip(self, model, processor):
        self.clip_model = model

    def _inference_context(self):
        """Mixed-precision context for GPU inference."""
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def is_ready(self) -> bool:
        return self.vqa_model is not None or self.caption_model is not None

    def classify_temporal_scope(self, query: str) -> tuple[str, float]:
        query_lower = query.lower().strip()
        scores = {"instant": 0.0, "recent": 0.0, "historical": 0.0}
        for scope, keywords in SCOPE_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    scores[scope] += 1.0
        total = sum(scores.values())
        if total == 0:
            return "historical", 0.4
        best_scope = max(scores, key=scores.get)
        confidence = scores[best_scope] / total
        if confidence < 0.6:
            return "historical", confidence
        return best_scope, confidence

    # --- Low-level helpers ---

    def _decode_frame(self, frame_b64: str) -> "Image.Image":
        raw = base64.b64decode(frame_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _clean_question_for_vqa(self, question: str) -> str:
        """Rephrase user's video/stream question into image-level VQA form."""
        q = question.strip()
        if not q.endswith("?"):
            q += "?"

        replacements = [
            ("in the stream", "in this image"), ("in the video", "in this image"),
            ("in the feed", "in this image"), ("in the scene", "in this image"),
            ("in the frame", "in this image"), ("on screen", "in this image"),
            ("right now", ""), ("currently", ""), ("at this moment", ""),
        ]
        ql = q.lower()
        for old, new in replacements:
            if old in ql:
                idx = ql.index(old)
                q = q[:idx] + new + q[idx + len(old):]
                ql = q.lower()

        if ql.startswith("any "):
            q = "Are there " + q[4:]

        q = " ".join(q.split())
        return q

    def _vqa(self, image: "Image.Image", question: str) -> str:
        if self.vqa_model is None:
            return ""
        inputs = self.vqa_processor(images=image, text=question, return_tensors="pt").to(self.device)
        with torch.no_grad(), self._inference_context():
            output = self.vqa_model.generate(**inputs, max_new_tokens=30)
        return self.vqa_processor.decode(output[0], skip_special_tokens=True).strip()

    _SCENE_QUESTIONS = [
        ("where", "Where is this scene?"),
        ("who", "Who is in this image?"),
        ("doing", "What is the person doing?"),
        ("count", "How many people are in this image?"),
    ]

    _NEGATIVE_PERSON = {"no one", "nobody", "no person", "no people",
                         "no man", "no woman", "none", "empty", "0"}

    def _is_negative_person(self, text: str) -> bool:
        t = text.strip().lower()
        return t in self._NEGATIVE_PERSON or t.startswith("no ")

    _BAD_LOCATIONS = {
        "dark", "light", "outside", "inside", "room", "screen",
        "image", "picture", "photo", "video", "frame", "it",
        "background", "unknown", "none", "no", "yes",
    }

    def _caption_frame(self, image: "Image.Image") -> str:
        """Generate a free-form BLIP caption for the frame."""
        if self.caption_model is None:
            return ""
        inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad(), self._inference_context():
            output = self.caption_model.generate(**inputs, max_new_tokens=50)
        return self.caption_processor.decode(output[0], skip_special_tokens=True).strip()

    _ENRICHMENT_QUESTIONS = [
        ("where", "Where is this scene taking place?"),
        ("action", "What is happening in this image?"),
    ]

    _BAD_ENRICHMENTS = _BAD_LOCATIONS | {
        "good", "bad", "nice", "fine", "great", "normal", "neutral",
        "a lot", "many", "some", "nothing", "something",
    }

    def _describe_frame(self, image: "Image.Image") -> str:
        """Generate a natural scene description by combining BLIP caption with VQA details.

        Runs caption and enrichment VQA concurrently (3 model calls overlap
        on the GPU timeline instead of running back-to-back).
        """
        cap_future = self._pool.submit(self._caption_frame, image)
        enrich_futures = [
            (tag, self._pool.submit(self._vqa, image, question))
            for tag, question in self._ENRICHMENT_QUESTIONS
        ]

        caption = cap_future.result()
        where = ""
        action = ""

        for tag, future in enrich_futures:
            ans = future.result()
            if not ans:
                continue
            ans_clean = ans.strip().rstrip(".")
            ans_lower = ans_clean.lower()
            if ans_lower in self._BAD_ENRICHMENTS or len(ans_clean.split()) < 2:
                continue
            if caption and ans_lower in caption.lower():
                continue
            if tag == "where":
                where = ans_clean
            elif tag == "action":
                action = ans_clean

        if caption:
            if where and where.lower() not in caption.lower():
                loc = where.removeprefix("in ").removeprefix("at ").removeprefix("the ").strip()
                if loc and len(loc.split()) >= 2:
                    caption = f"{caption} in {loc}"
            if action and action.lower() not in caption.lower():
                caption = f"{caption}. {action[0].upper()}{action[1:]}"
        elif where and action:
            caption = f"{action[0].upper()}{action[1:]} in {where}"
        elif action:
            caption = f"{action[0].upper()}{action[1:]}"
        elif where:
            caption = f"A scene in {where}"
        else:
            caption = "an unclear scene"

        if caption and not caption[0].isupper():
            caption = caption[0].upper() + caption[1:]

        return caption

    # --- Batch inference for multi-frame queries ---

    def _batch_caption_frames(self, images: list["Image.Image"]) -> list[str]:
        """Caption multiple frames in a single batched forward pass."""
        if not self.caption_model or not images:
            return [""] * len(images)
        inputs = self.caption_processor(
            images=images, return_tensors="pt", padding=True,
        ).to(self.device)
        with torch.no_grad(), self._inference_context():
            outputs = self.caption_model.generate(**inputs, max_new_tokens=50)
        return [
            self.caption_processor.decode(o, skip_special_tokens=True).strip()
            for o in outputs
        ]

    def _batch_vqa_frames(self, images: list["Image.Image"], question: str) -> list[str]:
        """Run the same VQA question across multiple frames in one batch."""
        if not self.vqa_model or not images:
            return [""] * len(images)
        questions = [question] * len(images)
        inputs = self.vqa_processor(
            images=images, text=questions, return_tensors="pt", padding=True,
        ).to(self.device)
        with torch.no_grad(), self._inference_context():
            outputs = self.vqa_model.generate(**inputs, max_new_tokens=30)
        return [
            self.vqa_processor.decode(o, skip_special_tokens=True).strip()
            for o in outputs
        ]

    def _synthesize(self, prompt: str) -> str:
        """Use Flan-T5 to generate a natural language response from a prompt."""
        if self.llm is None:
            return ""
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        with torch.no_grad(), self._inference_context():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=2,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
            )
        return self.llm_tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def _sample_frames(self, context_frames: list[dict],
                        scope: str = "historical") -> list[dict]:
        max_frames = MAX_SAMPLE_FRAMES.get(scope, 6)
        n = len(context_frames)
        if n <= max_frames:
            return context_frames
        step = n / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        return [context_frames[i] for i in indices]

    # --- Two-stage pipeline ---

    _PERSON_WORDS = {"man", "woman", "person", "people", "boy", "girl",
                      "child", "men", "women", "guy", "lady", "worker",
                      "someone", "standing", "sitting", "walking"}

    def _caption_mentions_person(self, caption: str) -> bool:
        c = caption.lower()
        if any(neg in c for neg in ("no one", "no person", "no people",
                                     "nobody", "empty")):
            return False
        words = set(c.split())
        return bool(words & self._PERSON_WORDS)

    def _asks_about_presence(self, query: str) -> bool:
        """Only match yes/no questions asking whether people are present.
        Does NOT match 'what is the person doing?' or similar descriptive queries."""
        q = query.lower().strip()
        if not self._is_yes_no_question(q):
            return False
        presence_patterns = (
            "anyone", "anybody", "someone", "somebody",
            "is there a person", "is there a man", "is there a woman",
            "are there people", "are there any people",
            "is there somebody", "is there anyone",
            "is somebody", "is anyone",
        )
        return any(p in q for p in presence_patterns)

    def _gather_observations(self, query: str, context_frames: list[dict],
                              scope: str = "historical") -> dict:
        """Stage 1: Run BLIP across sampled frames to gather visual observations.

        Single-frame (instant): full enrichment per frame (caption + where/action VQA)
          with all three model calls running concurrently via _describe_frame.
        Multi-frame (recent/historical): parallel frame decoding, concurrent
          batch captioning + batch VQA, with caption caching.
        """
        sampled = self._sample_frames(context_frames, scope=scope)
        if not sampled:
            return {"captions": [], "vqa_answers": [],
                    "n_sampled": 0, "n_total": len(context_frames)}

        # Parallel frame decoding (CPU-bound, benefits from threading)
        decode_futures = [
            self._pool.submit(self._decode_frame, fd["frame_base64"])
            for fd in sampled
        ]
        images = [f.result() for f in decode_futures]

        if len(images) == 1:
            desc = self._describe_frame(images[0])
            logger.info(f"Frame description: {desc}")
            descriptions = [desc] if desc else []

            if self._asks_about_presence(query):
                person_in_desc = self._caption_mentions_person(desc) if desc else False
                count = self._vqa(images[0], "How many people are in this image?")
                has_people = count and count.strip() not in ("0", "none", "no")
                vqa_answers = ["yes" if (person_in_desc or has_people) else "no"]
            else:
                clean_q = self._clean_question_for_vqa(query)
                ans = self._vqa(images[0], clean_q)
                logger.info(f"BLIP VQA [{clean_q}]: {ans}")
                vqa_answers = [ans] if ans else []
        else:
            # Check caption cache: only batch-caption uncached frames
            cached_caps: dict[int, str] = {}
            uncached_idx: list[int] = []
            for i, fd in enumerate(sampled):
                fid = fd.get("frame_id")
                if fid is not None and fid in self._caption_cache:
                    cached_caps[i] = self._caption_cache[fid]
                else:
                    uncached_idx.append(i)

            if self._asks_about_presence(query):
                if uncached_idx:
                    new_caps = self._batch_caption_frames(
                        [images[i] for i in uncached_idx])
                    for i, cap in zip(uncached_idx, new_caps):
                        cached_caps[i] = cap
                        fid = sampled[i].get("frame_id")
                        if fid is not None:
                            self._caption_cache[fid] = cap
                descriptions = [cached_caps.get(i, "") for i in range(len(sampled))]
                descriptions = [d for d in descriptions if d]
                vqa_answers = [
                    "yes" if self._caption_mentions_person(d) else "no"
                    for d in descriptions
                ]
            else:
                clean_q = self._clean_question_for_vqa(query)

                # Launch captioning (uncached only) and VQA concurrently
                cap_future = None
                if uncached_idx:
                    cap_future = self._pool.submit(
                        self._batch_caption_frames,
                        [images[i] for i in uncached_idx])
                vqa_future = self._pool.submit(
                    self._batch_vqa_frames, images, clean_q)

                if cap_future:
                    new_caps = cap_future.result()
                    for i, cap in zip(uncached_idx, new_caps):
                        cached_caps[i] = cap
                        fid = sampled[i].get("frame_id")
                        if fid is not None:
                            self._caption_cache[fid] = cap

                descriptions = [cached_caps.get(i, "") for i in range(len(sampled))]
                descriptions = [d for d in descriptions if d]
                vqa_answers = [a for a in vqa_future.result() if a]

            for d in descriptions:
                logger.info(f"Frame description: {d}")
            for ans in vqa_answers:
                logger.info(f"BLIP VQA: {ans}")

        return {
            "captions": descriptions,
            "vqa_answers": vqa_answers,
            "n_sampled": len(sampled),
            "n_total": len(context_frames),
        }

    def _is_yes_no_question(self, query: str) -> bool:
        q = query.lower().strip().rstrip("?")
        starters = ("is ", "are ", "was ", "were ", "do ", "does ", "did ",
                     "has ", "have ", "had ", "can ", "could ", "will ", "would ")
        return any(q.startswith(s) for s in starters)

    @staticmethod
    def _rank_by_frequency(items: list[str]) -> list[str]:
        """Return unique items sorted by frequency (most common first)."""
        from collections import Counter
        counts = Counter(items)
        seen = set()
        ranked = []
        for item, _ in counts.most_common():
            if item not in seen:
                seen.add(item)
                ranked.append(item)
        return ranked

    @staticmethod
    def _caption_word_set(text: str) -> set[str]:
        """Extract meaningful words from a caption for overlap comparison."""
        stop = {"a", "an", "the", "in", "on", "of", "and", "is", "are",
                "was", "with", "to", "at", "for", "from", "by", "it", "this"}
        return {w for w in text.lower().split() if w not in stop and len(w) > 1}

    def _deduplicate_captions(self, captions: list[str]) -> list[str]:
        """Remove near-duplicate captions using word-overlap similarity."""
        unique: list[str] = []
        unique_word_sets: list[set[str]] = []
        for c in captions:
            words = self._caption_word_set(c)
            if not words:
                continue
            is_dup = False
            for existing_words in unique_word_sets:
                overlap = len(words & existing_words)
                smaller = min(len(words), len(existing_words))
                if smaller > 0 and overlap / smaller >= 0.6:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)
                unique_word_sets.append(words)
        return unique

    def _build_prompt(self, query: str, obs: dict, scope: str) -> str:
        """Build a scope-aware Flan-T5 prompt from the user's question and visual observations."""
        ranked_captions = self._rank_by_frequency(obs["captions"])
        ranked_answers = self._rank_by_frequency(obs["vqa_answers"])

        if self._is_yes_no_question(query):
            caption_str = ". ".join(ranked_captions[:4]) if ranked_captions else "unknown"
            vqa_str = ", ".join(ranked_answers[:4]) if ranked_answers else "unknown"
            return (
                f"Question: {query}\n"
                f"Visual evidence: {vqa_str}\n"
                f"Scene description: {caption_str}\n"
                f"Answer yes or no, then briefly explain what you see:"
            )

        if scope == "instant":
            caption_str = ranked_captions[0] if ranked_captions else "unknown scene"
            vqa_str = ranked_answers[0] if ranked_answers else ""
            prompt = f"Describe this scene in detail: {caption_str}."
            if vqa_str:
                prompt += f" Additional detail: {vqa_str}."
            prompt += f" Question: {query}"
            return prompt

        unique = self._deduplicate_captions(ranked_captions)

        if scope == "recent":
            scene_str = "; ".join(unique[:4]) if unique else "unknown"
            vqa_str = ", ".join(ranked_answers[:4]) if ranked_answers else ""
            prompt = f"Summarize what happened in the last 30 seconds: {scene_str}."
            if vqa_str:
                prompt += f" Observations: {vqa_str}."
            prompt += f" Question: {query}"
            return prompt

        # historical
        scene_str = "; ".join(unique[:8]) if unique else "unknown"
        vqa_str = ", ".join(ranked_answers[:6]) if ranked_answers else ""
        prompt = (
            f"Summarize what happened across these scenes: {scene_str}. "
            f"There were {len(unique)} distinct scenes in total."
        )
        if vqa_str:
            prompt += f" Observations: {vqa_str}."
        prompt += f" Question: {query}"
        return prompt

    def _direct_answer_from_observations(self, query: str, obs: dict,
                                          scope: str = "instant") -> str:
        """Build a direct answer from BLIP observations without Flan-T5."""
        unique_captions = self._deduplicate_captions(
            self._rank_by_frequency(obs["captions"]))
        unique_answers = self._rank_by_frequency(obs["vqa_answers"])
        is_yn = self._is_yes_no_question(query)

        if is_yn and unique_answers:
            vqa_top = unique_answers[0].strip().lower()
            cap_detail = unique_captions[0] if unique_captions else ""

            if vqa_top == "yes" and cap_detail:
                return f"Yes — {cap_detail}."
            if vqa_top == "no":
                if cap_detail:
                    return f"No. What's visible: {cap_detail}."
                if self._asks_about_presence(query):
                    return "No, the room is currently empty."
                return "No."
            return f"{vqa_top.capitalize()}. {cap_detail.capitalize()}." if cap_detail else vqa_top.capitalize() + "."

        if scope == "instant":
            scene = unique_captions[0] if unique_captions else "an unclear scene"
            if not scene.endswith("."):
                scene += "."
            return scene

        if scope == "recent":
            if len(unique_captions) >= 3:
                earlier = ", ".join(unique_captions[:-1])
                latest = unique_captions[-1]
                return f"In the last few moments: {earlier}. Most recently, {latest}."
            if len(unique_captions) == 2:
                return f"Recently, {unique_captions[0]}, followed by {unique_captions[1]}."
            if unique_captions:
                return f"Just now: {unique_captions[0]}."
            if unique_answers:
                return ". ".join(a.capitalize() for a in unique_answers[:3]) + "."
            return ""

        # historical
        n = len(unique_captions)
        if n == 0:
            if unique_answers:
                return ". ".join(a.capitalize() for a in unique_answers[:4]) + "."
            return ""
        if n == 1:
            return f"Throughout the recording, the main scene has been {unique_captions[0]}."

        def _ensure_sentence(cap: str) -> str:
            cap = cap.strip().rstrip(".")
            if cap and not cap[0].isupper():
                cap = cap[0].upper() + cap[1:]
            return cap

        parts: list[str] = []
        parts.append(f"The video contained {n} distinct scenes.")

        first = _ensure_sentence(unique_captions[0])
        parts.append(f"It started with {first}.")

        if n <= 3:
            for cap in unique_captions[1:]:
                parts.append(f"{_ensure_sentence(cap)}.")
        else:
            mid = unique_captions[1:-1]
            mid_str = ", ".join(_ensure_sentence(c).lower() for c in mid[:5])
            parts.append(f"It then moved through {mid_str}.")
            last = _ensure_sentence(unique_captions[-1])
            parts.append(f"The most recent scene showed {last.lower()}.")

        return " ".join(parts)

    @staticmethod
    def _is_prompt_echo(prompt: str, answer: str) -> bool:
        """Detect when Flan-T5 just echoes back the prompt or its numbered list."""
        if not answer:
            return True
        a_low = answer.lower().strip()
        p_low = prompt.lower().strip()
        if a_low.startswith(p_low[:60]):
            return True
        if p_low.startswith(a_low[:60]):
            return True
        numbered_ratio = sum(1 for line in a_low.split(". ")
                            if line.strip()[:2].rstrip(".").isdigit())
        total_segments = max(len(a_low.split(". ")), 1)
        if numbered_ratio / total_segments > 0.5 and numbered_ratio >= 3:
            return True
        return False

    def _answer_with_pipeline(self, query: str, context_frames: list[dict], scope: str) -> str:
        """Full two-stage pipeline: BLIP perceives, Flan-T5 synthesizes."""
        if not context_frames:
            return "No frames available in memory to analyze."

        if not self.is_ready():
            return "Models are not loaded. Install PyTorch and transformers to enable the full pipeline."

        obs = self._gather_observations(query, context_frames, scope=scope)

        if not obs["captions"] and not obs["vqa_answers"]:
            return "Could not extract any visual information from the available frames."

        prompt = self._build_prompt(query, obs, scope)
        logger.info(f"Flan-T5 prompt: {prompt}")

        # Overlap T5 synthesis (GPU) with direct-answer assembly (CPU)
        t5_future = self._pool.submit(self._synthesize, prompt)
        direct = self._direct_answer_from_observations(query, obs, scope=scope)
        t5_answer = t5_future.result()

        logger.info(f"Flan-T5 answer: {t5_answer}")
        logger.info(f"Direct answer: {direct}")

        if self._is_yes_no_question(query):
            unique_answers = list(dict.fromkeys(obs["vqa_answers"]))
            if unique_answers:
                expected_start = unique_answers[0].strip().lower()
                t5_low = t5_answer.lower().strip() if t5_answer else ""
                t5_is_bare = t5_low in ("yes", "no", "yes.", "no.")
                if t5_answer and t5_low.startswith(expected_start) and not t5_is_bare:
                    return t5_answer
            return direct

        if t5_answer and len(t5_answer.split()) > 5 and not self._is_prompt_echo(prompt, t5_answer):
            return t5_answer

        return direct if direct else (t5_answer or "The model did not produce an answer.")

    def generate_answer(self, query: str, context_frames: list[dict], scope: str) -> dict:
        start_time = time.time()
        n_frames = len(context_frames)

        answer = self._answer_with_pipeline(query, context_frames, scope)

        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "answer": answer,
            "scope": scope,
            "num_context_frames": n_frames,
            "latency_ms": round(elapsed_ms, 1),
        }

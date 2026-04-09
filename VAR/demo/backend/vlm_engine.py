"""
VLM inference engine — two-stage pipeline:
  Stage 1 (BLIP): Visual perception — captions and VQA on each frame
  Stage 2 (Flan-T5): Language synthesis — composes a natural, coherent
           answer from the visual observations and the user's question
"""
from __future__ import annotations

import base64
import io
import logging
import time

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

MAX_SAMPLE_FRAMES = {"instant": 1, "recent": 6, "historical": 12}
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
        with torch.no_grad():
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
        with torch.no_grad():
            output = self.caption_model.generate(**inputs, max_new_tokens=50)
        return self.caption_processor.decode(output[0], skip_special_tokens=True).strip()

    def _describe_frame(self, image: "Image.Image") -> str:
        """Generate a clean scene description via BLIP captioning."""
        caption = self._caption_frame(image)

        where = self._vqa(image, "Where is this scene taking place?")
        if where:
            where_clean = where.removeprefix("in ").removeprefix("at ").removeprefix("the ").strip()
            is_useful = (
                where_clean
                and where_clean.lower() not in self._BAD_LOCATIONS
                and len(where_clean.split()) >= 2
                and where_clean.lower() not in (caption or "").lower()
            )
            if is_useful and caption:
                caption = f"{caption} in {where_clean}"
            elif is_useful:
                caption = f"a scene in {where_clean}"

        if not caption:
            caption = self._caption_frame(image) or "an unclear scene"

        if not caption[0].isupper():
            caption = caption[0].upper() + caption[1:]

        return caption

    def _synthesize(self, prompt: str) -> str:
        """Use Flan-T5 to generate a natural language response from a prompt."""
        if self.llm is None:
            return ""
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        with torch.no_grad():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=120,
                num_beams=4,
                length_penalty=1.2,
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
        """Stage 1: Run BLIP across sampled frames to gather visual observations."""
        sampled = self._sample_frames(context_frames, scope=scope)
        descriptions = []
        vqa_answers = []

        for fd in sampled:
            img = self._decode_frame(fd["frame_base64"])
            desc = self._describe_frame(img)
            logger.info(f"Frame description: {desc}")
            if desc:
                descriptions.append(desc)

            if self._asks_about_presence(query):
                person_in_desc = self._caption_mentions_person(desc) if desc else False
                count = self._vqa(img, "How many people are in this image?")
                has_people = count and count.strip() not in ("0", "none", "no")
                if person_in_desc or has_people:
                    vqa_answers.append("yes")
                else:
                    vqa_answers.append("no")
            else:
                clean_q = self._clean_question_for_vqa(query)
                ans = self._vqa(img, clean_q)
                logger.info(f"BLIP VQA [{clean_q}]: {ans}")
                if ans:
                    vqa_answers.append(ans)

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

    def _build_prompt(self, query: str, obs: dict, scope: str) -> str:
        """Build a scope-aware Flan-T5 prompt from the user's question and visual observations."""
        ranked_captions = self._rank_by_frequency(obs["captions"])
        ranked_answers = self._rank_by_frequency(obs["vqa_answers"])

        n_frames = obs.get("n_sampled", 1)

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
            return (
                f"You are describing what a camera sees right now.\n"
                f"The camera shows: {caption_str}\n"
                f"Question: {query}\n"
                f"Write a clear, plain English description of the scene in 1-2 sentences. "
                f"Do not list objects. Describe the scene as a whole:"
            )

        if scope == "recent":
            numbered = [f"{i+1}. {c}" for i, c in enumerate(ranked_captions[:6])]
            scene_list = "\n".join(numbered) if numbered else "unknown"
            return (
                f"A camera has been recording. In the last 30 seconds, it captured these scenes:\n"
                f"{scene_list}\n"
                f"Question: {query}\n"
                f"Write a short plain English paragraph describing what happened recently. "
                f"Connect the scenes into a narrative rather than listing them:"
            )

        # historical
        unique_scenes = []
        seen = set()
        for c in ranked_captions:
            key = c.lower().split(",")[0].strip()[:30]
            if key not in seen:
                seen.add(key)
                unique_scenes.append(c)
        numbered = [f"{i+1}. {c}" for i, c in enumerate(unique_scenes[:10])]
        scene_list = "\n".join(numbered) if numbered else "unknown"
        return (
            f"A video has been playing for several minutes. "
            f"These {len(unique_scenes)} different scenes appeared:\n"
            f"{scene_list}\n"
            f"Question: {query}\n"
            f"Write a plain English summary paragraph describing the variety of scenes. "
            f"Mention what changed over time. Do not repeat the list:"
        )

    def _direct_answer_from_observations(self, query: str, obs: dict) -> str:
        """Build a direct answer from BLIP observations without Flan-T5."""
        unique_captions = self._rank_by_frequency(obs["captions"])
        unique_answers = self._rank_by_frequency(obs["vqa_answers"])
        is_yn = self._is_yes_no_question(query)

        if is_yn and unique_answers:
            vqa_top = unique_answers[0].strip().lower()
            cap_detail = unique_captions[0] if unique_captions else ""

            if vqa_top == "yes" and cap_detail:
                return f"Yes. The scene shows {cap_detail}."
            if vqa_top == "no":
                if cap_detail:
                    return f"No. The scene shows {cap_detail}."
                if self._asks_about_presence(query):
                    return "No, the room is currently empty."
                return "No."
            return f"{vqa_top.capitalize()}. {cap_detail.capitalize()}." if cap_detail else vqa_top.capitalize() + "."

        if len(unique_captions) >= 3:
            first = unique_captions[0]
            rest = unique_captions[1:4]
            return f"The video shows {first}. Other scenes include {', and '.join(rest)}."
        if len(unique_captions) == 2:
            return f"The video shows {unique_captions[0]}, followed by {unique_captions[1]}."
        if unique_captions:
            scene = unique_captions[0]
            if unique_answers:
                return f"The scene shows {scene}. {unique_answers[0].capitalize()}."
            return f"The scene shows {scene}."
        if unique_answers:
            return ". ".join(a.capitalize() for a in unique_answers[:3]) + "."
        return ""

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
        t5_answer = self._synthesize(prompt)
        logger.info(f"Flan-T5 answer: {t5_answer}")

        direct = self._direct_answer_from_observations(query, obs)
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

        if t5_answer and len(t5_answer.split()) > 3:
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

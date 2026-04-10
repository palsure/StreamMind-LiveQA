"""
Stream processor: handles frame ingestion from WebSocket,
encodes frames with CLIP, and feeds them to the memory manager.
"""
from __future__ import annotations

import base64
import contextlib
import io
import numpy as np
from PIL import Image

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from memory_manager import MemoryManager


class StreamProcessor:
    """Processes incoming video frames and maintains the keyframe memory."""

    def __init__(
        self,
        memory_capacity: int = 32,
        model_name: str = "openai/clip-vit-base-patch32",
        frame_skip: int = 5,
    ):
        self.memory = MemoryManager(capacity=memory_capacity)
        self.frame_skip = frame_skip
        self._frame_count = 0

        if _HAS_TORCH:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
        else:
            self.model = None
            self.processor = None
            self.device = "cpu"

    def _decode_frame(self, frame_data: str) -> Image.Image:
        """Decode a base64-encoded frame to a PIL Image."""
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]
        raw = base64.b64decode(frame_data)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _encode_frame(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL Image to a CLIP embedding vector."""
        if not _HAS_TORCH or self.model is None:
            return np.random.randn(512).astype(np.float32)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        amp = (torch.autocast(device_type="cuda", dtype=torch.float16)
               if self.device == "cuda" else contextlib.nullcontext())
        with torch.no_grad(), amp:
            vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_out.pooler_output
            features = self.model.visual_projection(pooled)
        return features.cpu().numpy().flatten()

    def _make_thumbnail(self, image: Image.Image, size: int = 160) -> str:
        """Create a small base64 thumbnail for memory visualization."""
        thumb = image.copy()
        thumb.thumbnail((size, size))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def process_frame(self, frame_data: str, timestamp: float | None = None) -> dict:
        """
        Process a single frame from the stream.
        Returns a status dict indicating whether the frame was stored.
        Pass explicit `timestamp` (video-relative seconds) for offline evaluation.
        """
        self._frame_count += 1

        if self._frame_count % self.frame_skip != 0:
            return {"stored": False, "frame_number": self._frame_count}

        image = self._decode_frame(frame_data)
        embedding = self._encode_frame(image)
        thumbnail = self._make_thumbnail(image)

        stored = self.memory.add_frame(embedding, thumbnail, timestamp=timestamp)

        return {
            "stored": stored,
            "frame_number": self._frame_count,
            "memory_size": len(self.memory.entries),
        }

    def get_memory_state(self) -> list[dict]:
        return self.memory.get_memory_state()

    def get_context_for_query(self, scope: str, current_time: float | None = None) -> list[dict]:
        """Get memory entries for a given temporal scope."""
        entries = self.memory.get_entries_by_scope(scope, current_time=current_time)
        return [
            {
                "frame_id": e.frame_id,
                "timestamp": e.timestamp,
                "importance": round(e.importance, 3),
                "frame_base64": e.frame_base64,
            }
            for e in entries
        ]

    def reset(self):
        self.memory.clear()
        self._frame_count = 0

"""
Semantic Keyframe Memory (SKM) implementation.
Maintains a fixed-capacity bank of visually salient keyframes
from a live video stream, evicting low-importance entries as
new frames arrive.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    frame_id: int
    timestamp: float
    embedding: np.ndarray
    frame_base64: str
    importance: float = 0.0


class MemoryManager:
    """Fixed-capacity keyframe memory with importance-based eviction."""

    def __init__(self, capacity: int = 32, alpha: float = 0.7, t_max: float = 300.0):
        self.capacity = capacity
        self.alpha = alpha
        self.t_max = t_max
        self.entries: list[MemoryEntry] = []
        self._frame_counter = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_norm, b_norm))

    def _compute_importance(self, embedding: np.ndarray, timestamp: float) -> float:
        if not self.entries:
            return 1.0

        max_sim = max(
            self._cosine_similarity(embedding, e.embedding) for e in self.entries
        )
        novelty = 1.0 - max_sim

        min_time_gap = min(
            abs(timestamp - e.timestamp) for e in self.entries
        )
        temporal_coverage = min(min_time_gap / self.t_max, 1.0)

        return self.alpha * novelty + (1.0 - self.alpha) * temporal_coverage

    def _recompute_stored_importance(self):
        if not self.entries:
            return
        now = max(e.timestamp for e in self.entries)
        for i, entry in enumerate(self.entries):
            others = [e for j, e in enumerate(self.entries) if j != i]
            if not others:
                entry.importance = 1.0
                continue
            max_sim = max(
                self._cosine_similarity(entry.embedding, o.embedding) for o in others
            )
            min_gap = min(abs(entry.timestamp - o.timestamp) for o in others)
            raw = (
                self.alpha * (1.0 - max_sim)
                + (1.0 - self.alpha) * min(min_gap / self.t_max, 1.0)
            )
            age = now - entry.timestamp
            decay = 1.0 / (1.0 + age / self.t_max)
            entry.importance = raw * decay

    def add_frame(self, embedding: np.ndarray, frame_base64: str, timestamp: float | None = None) -> bool:
        """Add a frame to memory. Returns True if it was stored."""
        self._frame_counter += 1
        if timestamp is None:
            timestamp = time.time()
        importance = self._compute_importance(embedding, timestamp)

        if len(self.entries) < self.capacity:
            self.entries.append(MemoryEntry(
                frame_id=self._frame_counter,
                timestamp=timestamp,
                embedding=embedding,
                frame_base64=frame_base64,
                importance=importance,
            ))
            self.entries.sort(key=lambda e: e.timestamp)
            return True

        self._recompute_stored_importance()
        min_entry = min(self.entries, key=lambda e: e.importance)

        if importance > min_entry.importance:
            self.entries.remove(min_entry)
            self.entries.append(MemoryEntry(
                frame_id=self._frame_counter,
                timestamp=timestamp,
                embedding=embedding,
                frame_base64=frame_base64,
                importance=importance,
            ))
            self.entries.sort(key=lambda e: e.timestamp)
            return True

        return False

    def get_entries_by_scope(
        self, scope: str, current_time: float | None = None,
        window_seconds: float = 30.0, min_recent_frames: int = 3,
    ) -> list[MemoryEntry]:
        """Return memory entries filtered by temporal scope.

        For 'recent': returns frames within the last window_seconds. If fewer
        than min_recent_frames are found, returns the most recent frames from
        memory (up to min_recent_frames) so the model always has context.
        """
        if current_time is None:
            current_time = time.time()

        if scope == "instant":
            if not self.entries:
                return []
            return [max(self.entries, key=lambda e: e.timestamp)]

        if scope == "recent":
            cutoff = current_time - window_seconds
            recent = [e for e in self.entries if e.timestamp >= cutoff]
            if len(recent) >= min_recent_frames:
                return recent
            # Fallback: return the N most recent frames in chronological order
            sorted_by_time = sorted(self.entries, key=lambda e: e.timestamp)
            return sorted_by_time[-max(min_recent_frames, len(recent)):]

        return list(self.entries)

    def get_memory_state(self) -> list[dict]:
        """Return serializable representation of current memory."""
        return [
            {
                "frame_id": e.frame_id,
                "timestamp": e.timestamp,
                "importance": round(e.importance, 3),
                "frame_base64": e.frame_base64,
            }
            for e in self.entries
        ]

    def clear(self):
        self.entries.clear()
        self._frame_counter = 0

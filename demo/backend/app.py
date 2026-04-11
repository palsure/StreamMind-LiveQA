"""
StreamMind Demo Server
FastAPI application with WebSocket endpoints for live video streaming
and interactive question answering.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from stream_processor import StreamProcessor
from vlm_engine import VLMEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streammind")

processor: StreamProcessor | None = None
vlm: VLMEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, vlm
    memory_capacity = int(os.environ.get("MEMORY_CAPACITY", "32"))

    logger.info(f"Initializing StreamProcessor (capacity={memory_capacity})")
    processor = StreamProcessor(memory_capacity=memory_capacity)

    logger.info("Initializing VLMEngine (BLIP VQA + captioning)")
    vlm = VLMEngine()

    logger.info("StreamMind demo server ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title="StreamMind Demo", lifespan=lifespan)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

SAMPLES_DIR = os.path.join(FRONTEND_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)


@app.get("/")
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "StreamMind Demo API. Serve frontend from /static/index.html"}


@app.get("/api/status")
async def status():
    return {
        "status": "running",
        "memory_size": len(processor.memory.entries) if processor else 0,
        "memory_capacity": processor.memory.capacity if processor else 0,
        "vlm_loaded": vlm.is_ready() if vlm else False,
    }


@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for video frame ingestion.
    CLIP encoding runs in a thread so the event loop stays responsive.
    A periodic keepalive sends memory state even when no new frame is stored.
    """
    await websocket.accept()
    logger.info("Stream WebSocket connected")

    last_mem_push = 0.0
    KEEPALIVE_INTERVAL = 3.0

    async def _send_memory():
        nonlocal last_mem_push
        memory_state = processor.get_memory_state()
        await websocket.send_json({
            "type": "memory_update",
            "memory": memory_state,
            "memory_size": len(memory_state),
        })
        last_mem_push = asyncio.get_event_loop().time()

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                await _send_memory()
                continue

            msg = json.loads(data)

            if msg.get("type") == "frame":
                result = await asyncio.to_thread(
                    processor.process_frame, msg["data"])

                now = asyncio.get_event_loop().time()
                if result.get("stored") or (now - last_mem_push > KEEPALIVE_INTERVAL):
                    await _send_memory()

            elif msg.get("type") == "reset":
                processor.reset()
                if vlm:
                    vlm._caption_cache.clear()
                await websocket.send_json({
                    "type": "memory_update",
                    "memory": [],
                    "memory_size": 0,
                })
                last_mem_push = asyncio.get_event_loop().time()

    except WebSocketDisconnect:
        logger.info("Stream WebSocket disconnected")
    except Exception as e:
        logger.error(f"Stream error: {e}")


@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for interactive QA.
    Client sends questions; server responds with answers + metadata.
    """
    await websocket.accept()
    logger.info("Chat WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "question":
                query = msg["text"]

                scope, confidence = vlm.classify_temporal_scope(query)
                context_frames = processor.get_context_for_query(scope)

                # Run inference in a thread so frames keep flowing
                result = await asyncio.to_thread(
                    vlm.generate_answer, query, context_frames, scope)
                result["scope_confidence"] = round(confidence, 2)
                result["type"] = "answer"

                await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected")
    except Exception as e:
        logger.error(f"Chat error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

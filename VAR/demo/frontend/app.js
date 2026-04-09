/**
 * StreamMind Demo — Frontend Client
 * Handles webcam capture, WebSocket communication, and chat UI.
 */

const WS_BASE = `ws://${location.host}`;
const FRAME_INTERVAL_MS = 200; // send ~5 fps to server

// DOM elements
const videoFeed = document.getElementById("videoFeed");
const captureCanvas = document.getElementById("captureCanvas");
const videoOverlay = document.getElementById("videoOverlay");
const btnWebcam = document.getElementById("btnWebcam");
const btnSampleVideo = document.getElementById("btnSampleVideo");
const btnLoadVideo = document.getElementById("btnLoadVideo");
const videoFileInput = document.getElementById("videoFileInput");
const btnPause = document.getElementById("btnPause");
const btnStop = document.getElementById("btnStop");
const btnReset = document.getElementById("btnReset");
const streamStatus = document.getElementById("streamStatus");
const statusText = document.getElementById("statusText");
const memoryCount = document.getElementById("memoryCount");
const filmstrip = document.getElementById("filmstrip");
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const btnSend = document.getElementById("btnSend");

let streamWs = null;
let chatWs = null;
let mediaStream = null;
let frameTimer = null;
let isPaused = false;

// ---- WebSocket connections ----

function connectStreamWs() {
  streamWs = new WebSocket(`${WS_BASE}/ws/stream`);

  streamWs.onopen = () => {
    streamStatus.className = "status-dot online";
    statusText.textContent = "Connected";
  };

  streamWs.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "memory_update") {
      updateMemoryDisplay(msg.memory);
      memoryCount.textContent = msg.memory_size;
    }
  };

  streamWs.onclose = () => {
    streamStatus.className = "status-dot offline";
    statusText.textContent = "Disconnected";
  };
}

function connectChatWs() {
  chatWs = new WebSocket(`${WS_BASE}/ws/chat`);

  chatWs.onopen = () => {
    chatInput.disabled = false;
    btnSend.disabled = false;
  };

  chatWs.onmessage = (event) => {
    removeTypingIndicator();
    const msg = JSON.parse(event.data);
    if (msg.type === "answer") {
      addMessage("assistant", msg.answer, {
        scope: msg.scope,
        latency: msg.latency_ms,
        frames: msg.num_context_frames,
      });
      chatInput.disabled = false;
      btnSend.disabled = false;
    }
  };

  chatWs.onclose = () => {
    chatInput.disabled = true;
    btnSend.disabled = true;
    removeTypingIndicator();
  };
}

// ---- Webcam capture ----

async function startWebcam() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });

    videoFeed.srcObject = mediaStream;
    videoOverlay.classList.add("hidden");
    btnWebcam.disabled = true;
    btnPause.disabled = false;
    btnPause.classList.add("btn-pause-active");
    btnStop.disabled = false;

    connectStreamWs();
    connectChatWs();

    captureCanvas.width = 640;
    captureCanvas.height = 480;

    frameTimer = setInterval(sendFrame, FRAME_INTERVAL_MS);
  } catch (err) {
    addMessage("system", `Camera error: ${err.message}`);
  }
}

function sendFrame() {
  if (!streamWs || streamWs.readyState !== WebSocket.OPEN) return;

  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(videoFeed, 0, 0, captureCanvas.width, captureCanvas.height);
  const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.6);

  streamWs.send(JSON.stringify({ type: "frame", data: dataUrl }));
}

function startVideoPlayback(url) {
  videoFeed.srcObject = null;
  videoFeed.src = url;
  videoFeed.loop = true;
  videoFeed.muted = true;

  videoFeed.onloadeddata = () => {
    videoFeed.play();
    videoOverlay.classList.add("hidden");
    btnWebcam.disabled = true;
    btnSampleVideo.disabled = true;
    btnLoadVideo.disabled = true;
    btnPause.disabled = false;
    btnPause.classList.add("btn-pause-active");
    btnStop.disabled = false;

    connectStreamWs();
    connectChatWs();

    captureCanvas.width = 640;
    captureCanvas.height = 480;

    frameTimer = setInterval(sendFrame, FRAME_INTERVAL_MS);
  };
}

function loadVideo(file) {
  startVideoPlayback(URL.createObjectURL(file));
}

function loadSampleVideo(src) {
  startVideoPlayback(src || "/static/samples/trailer_interstellar.mp4");
}

function togglePause() {
  isPaused = !isPaused;
  btnPause.textContent = isPaused ? "Resume" : "Pause";

  if (isPaused) {
    if (frameTimer) clearInterval(frameTimer);
    frameTimer = null;
    if (videoFeed.src && !videoFeed.srcObject) videoFeed.pause();
  } else {
    if (videoFeed.src && !videoFeed.srcObject) videoFeed.play();
    frameTimer = setInterval(sendFrame, FRAME_INTERVAL_MS);
  }
}

function stopStream() {
  if (frameTimer) clearInterval(frameTimer);
  frameTimer = null;

  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }

  if (videoFeed.src && !videoFeed.srcObject) {
    videoFeed.pause();
    URL.revokeObjectURL(videoFeed.src);
    videoFeed.removeAttribute("src");
  }

  videoFeed.srcObject = null;
  videoOverlay.classList.remove("hidden");

  if (streamWs) streamWs.close();
  if (chatWs) chatWs.close();

  isPaused = false;
  btnPause.textContent = "Pause";
  btnPause.disabled = true;
  btnPause.classList.remove("btn-pause-active");
  btnWebcam.disabled = false;
  btnSampleVideo.disabled = false;
  btnLoadVideo.disabled = false;
  btnStop.disabled = true;
  videoFileInput.value = "";
}

function resetMemory() {
  if (streamWs && streamWs.readyState === WebSocket.OPEN) {
    streamWs.send(JSON.stringify({ type: "reset" }));
  }
  memoryCount.textContent = "0";
  filmstrip.innerHTML = '<div class="filmstrip-empty">No keyframes stored yet</div>';
  addMessage("system", "Memory cleared.");
}

// ---- Memory display ----

function updateMemoryDisplay(entries) {
  filmstrip.innerHTML = "";

  if (!entries || entries.length === 0) {
    filmstrip.innerHTML = '<div class="filmstrip-empty">No keyframes stored yet</div>';
    return;
  }

  entries.forEach((entry) => {
    const kf = document.createElement("div");
    kf.className = "keyframe";

    const img = document.createElement("img");
    img.src = `data:image/jpeg;base64,${entry.frame_base64}`;
    img.alt = `Keyframe #${entry.frame_id}`;

    const score = document.createElement("div");
    score.className = "kf-score";
    score.textContent = entry.importance.toFixed(2);

    kf.appendChild(img);
    kf.appendChild(score);
    filmstrip.appendChild(kf);
  });

  filmstrip.scrollLeft = filmstrip.scrollWidth;
}

// ---- Chat UI ----

function addMessage(role, text, meta) {
  const div = document.createElement("div");
  div.className = `message ${role}`;

  const p = document.createElement("p");
  p.textContent = text;
  div.appendChild(p);

  if (meta) {
    const metaDiv = document.createElement("div");
    metaDiv.className = "meta";

    const scopeTag = document.createElement("span");
    scopeTag.className = `scope-tag ${meta.scope}`;
    scopeTag.textContent = meta.scope;
    metaDiv.appendChild(scopeTag);

    if (meta.frames !== undefined) {
      const framesSpan = document.createElement("span");
      framesSpan.textContent = `${meta.frames} frame(s)`;
      metaDiv.appendChild(framesSpan);
    }

    if (meta.latency !== undefined) {
      const latSpan = document.createElement("span");
      latSpan.textContent = `${meta.latency}ms`;
      metaDiv.appendChild(latSpan);
    }

    div.appendChild(metaDiv);
  }

  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
  removeTypingIndicator();
  const div = document.createElement("div");
  div.className = "message assistant typing-indicator";
  div.id = "typingIndicator";
  div.innerHTML = `
    <div class="typing-dots">
      <span class="typing-label">Analyzing frames</span>
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    </div>`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

function handleSubmit(e) {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  addMessage("user", text);
  chatInput.value = "";

  if (chatWs && chatWs.readyState === WebSocket.OPEN) {
    chatInput.disabled = true;
    btnSend.disabled = true;
    showTypingIndicator();
    chatWs.send(JSON.stringify({ type: "question", text }));
  } else {
    addMessage("system", "Not connected. Start the webcam first.");
  }
}

// ---- Presentation mode ----

const btnPresentation = document.getElementById("btnPresentation");
const suggestedQuestions = document.getElementById("suggestedQuestions");

function togglePresentationMode() {
  document.body.classList.toggle("presentation-mode");
  const active = document.body.classList.contains("presentation-mode");
  btnPresentation.textContent = active ? "Exit Presentation" : "Presentation Mode";
}

// ---- Event listeners ----

btnWebcam.addEventListener("click", startWebcam);
btnPresentation.addEventListener("click", togglePresentationMode);

const sampleMenu = document.getElementById("sampleMenu");
btnSampleVideo.addEventListener("click", () => {
  sampleMenu.classList.toggle("open");
});
document.querySelectorAll(".sample-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    sampleMenu.classList.remove("open");
    loadSampleVideo(btn.dataset.src);
  });
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".sample-dropdown")) {
    sampleMenu.classList.remove("open");
  }
});

document.querySelectorAll(".suggested-q").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (chatWs && chatWs.readyState === WebSocket.OPEN) {
      chatInput.value = btn.dataset.q;
      chatForm.dispatchEvent(new Event("submit"));
    }
  });
});

btnLoadVideo.addEventListener("click", () => videoFileInput.click());
videoFileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) loadVideo(e.target.files[0]);
});
btnPause.addEventListener("click", togglePause);
btnStop.addEventListener("click", stopStream);
btnReset.addEventListener("click", resetMemory);
document.getElementById("btnClearChat").addEventListener("click", () => {
  chatMessages.innerHTML = '';
  addMessage("system", "Chat cleared. Ask a new question about the stream.");
});
chatForm.addEventListener("submit", handleSubmit);

import gc
import io
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

from faster_whisper import WhisperModel
from flask import Flask, jsonify, make_response, render_template, request

app = Flask(__name__)

ALLOWED_EXTENSIONS = {
    ".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac",
    ".webm", ".mkv", ".avi", ".mov", ".wma", ".aac",
}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".ts", ".mts"}
ALLOWED_MODELS = {"base", "medium", "large-v3-turbo"}
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")
# Runs transcription on a single CPU thread by default so a job can't hog every
# core on the (shared) host. Slower per job, much lighter on the machine.
CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "1"))
# Unload cached models after this many seconds of inactivity to free the RAM
# they hold while idle (a "medium" model alone is ~1.5GB).
IDLE_UNLOAD_SECONDS = int(os.environ.get("WHISPER_IDLE_UNLOAD_SECONDS", "600"))
MODELS_DIR = "/models"

# pyannote's pretrained pipeline is gated on Hugging Face: an account must
# accept the terms at huggingface.co/pyannote/speaker-diarization-3.1 (and
# .../segmentation-3.0) and generate a read token, passed in here as HF_TOKEN.
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
HF_TOKEN = os.environ.get("HF_TOKEN")
DIARIZATION_KEY = "__diarization__"
SPEAKER_COLORS = ["#7c6fff", "#4ade80", "#facc15", "#f87171", "#38bdf8", "#f472b6", "#fb923c", "#a3e635"]

_models: dict[str, object] = {}
_last_used: dict[str, float] = {}
_models_lock = threading.Lock()

# upload_id -> { path, received, total, ext }
pending_uploads: dict[str, dict] = {}
jobs: dict[str, dict] = {}

# jobs are processed one at a time by a single background worker
job_queue: "queue.Queue[tuple]" = queue.Queue()


def get_model(model_name: str) -> WhisperModel:
    with _models_lock:
        if model_name not in _models:
            _models[model_name] = WhisperModel(
                model_name,
                device="cpu",
                compute_type=COMPUTE_TYPE,
                download_root=MODELS_DIR,
                cpu_threads=CPU_THREADS,
                num_workers=1,
            )
        _last_used[model_name] = time.monotonic()
    return _models[model_name]


def _touch_model(model_name: str) -> None:
    with _models_lock:
        _last_used[model_name] = time.monotonic()


def _model_evictor() -> None:
    while True:
        time.sleep(60)
        evicted = False
        with _models_lock:
            now = time.monotonic()
            for name in list(_models.keys()):
                if now - _last_used.get(name, now) > IDLE_UNLOAD_SECONDS:
                    del _models[name]
                    del _last_used[name]
                    evicted = True
        if evicted:
            gc.collect()


threading.Thread(target=_model_evictor, daemon=True).start()


def get_diarization_pipeline():
    with _models_lock:
        if DIARIZATION_KEY not in _models:
            # Imported lazily: pyannote pulls in torch, which is a much
            # heavier/larger runtime than faster-whisper's ctranslate2 backend,
            # so we only pay for it once diarization is actually requested.
            import torch
            from pyannote.audio import Pipeline

            torch.set_num_threads(CPU_THREADS)
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=HF_TOKEN)
            pipeline.to(torch.device("cpu"))
            _models[DIARIZATION_KEY] = pipeline
        _last_used[DIARIZATION_KEY] = time.monotonic()
    return _models[DIARIZATION_KEY]


def diarize(audio_path: str) -> list[tuple[float, float, str]]:
    pipeline = get_diarization_pipeline()
    diarization = pipeline(audio_path)
    turns = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
    _touch_model(DIARIZATION_KEY)
    return turns


def _assign_speakers(lines: list[dict], turns: list[tuple[float, float, str]]) -> dict[str, str]:
    """Labels each whisper segment with the diarization speaker it overlaps
    most with, renaming raw pyannote labels to "Talare 1", "Talare 2"... in
    order of first appearance. Returns a {label: color} map for display."""
    speaker_order: list[str] = []
    for seg in lines:
        best_speaker, best_overlap = None, 0.0
        for start, end, speaker in turns:
            overlap = min(seg["end_s"], end) - max(seg["start_s"], start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, speaker
        if best_speaker is None:
            seg["speaker"] = None
            continue
        if best_speaker not in speaker_order:
            speaker_order.append(best_speaker)
        seg["speaker"] = f"Talare {speaker_order.index(best_speaker) + 1}"

    return {
        f"Talare {i + 1}": SPEAKER_COLORS[i % len(SPEAKER_COLORS)]
        for i in range(len(speaker_order))
    }


def extract_audio(video_path: str) -> str:
    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    out.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k", out.name],
        check=True, capture_output=True,
    )
    return out.name


def run_transcription(job_id: str, file_path: str, language: str | None, model_name: str, is_video: bool, diarize_flag: bool = False):
    audio_path = None
    jobs[job_id]["status"] = "processing"
    try:
        if is_video:
            jobs[job_id]["status_text"] = "Extraherar ljud från video…"
            audio_path = extract_audio(file_path)
            transcribe_path = audio_path
        else:
            transcribe_path = file_path

        jobs[job_id]["status_text"] = "Transkriberar…"
        model = get_model(model_name)
        segments, info = model.transcribe(
            transcribe_path,
            language=language or None,
            beam_size=5,
            vad_filter=True,
        )

        lines = [{"start": _fmt(s.start), "end": _fmt(s.end), "start_s": s.start, "end_s": s.end, "text": s.text.strip()} for s in segments]
        _touch_model(model_name)

        speaker_colors: dict[str, str] = {}
        if diarize_flag and lines:
            jobs[job_id]["status_text"] = "Identifierar talare…"
            turns = diarize(transcribe_path)
            speaker_colors = _assign_speakers(lines, turns)

        text = "\n".join(
            (f"{s['speaker']}: {s['text']}" if s.get("speaker") else s["text"]) for s in lines
        )
        jobs[job_id] = {
            "status": "done",
            "text": text,
            "segments": lines,
            "language": info.language,
            "duration": round(info.duration, 1),
            "model": model_name,
            "speaker_colors": speaker_colors,
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}
    finally:
        for p in [file_path, audio_path]:
            if p:
                try: os.unlink(p)
                except OSError: pass


def _fmt_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    frac = seconds - int(seconds)
    return (f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}") + f".{int(frac*100):02d}"


def _build_srt(job: dict) -> str:
    colors = job.get("speaker_colors") or {}
    rows = []
    for i, seg in enumerate(job.get("segments", []), 1):
        speaker = seg.get("speaker")
        text = seg["text"]
        if speaker:
            text = f"{speaker}: {text}"
            color = colors.get(speaker)
            if color:
                text = f'<font color="{color}">{text}</font>'
        rows.append(str(i))
        rows.append(f"{_fmt_srt(seg['start_s'])} --> {_fmt_srt(seg['end_s'])}")
        rows.append(text)
        rows.append("")
    return "\n".join(rows)


def _sanitize_filename(name: str, ext: str) -> str:
    name = os.path.basename(name or "transkribering")
    name = re.sub(r'[\\/:"*?<>|\r\n]+', "_", name)
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return f"{name}.{ext}"


def _queue_worker():
    while True:
        job_id, file_path, language, model_name, is_video, diarize_flag = job_queue.get()
        try:
            run_transcription(job_id, file_path, language, model_name, is_video, diarize_flag)
        finally:
            job_queue.task_done()


threading.Thread(target=_queue_worker, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload-chunk", methods=["POST"])
def upload_chunk():
    upload_id  = request.form.get("upload_id")
    chunk_idx  = int(request.form.get("chunk_index", 0))
    total      = int(request.form.get("total_chunks", 1))
    ext        = request.form.get("ext", ".mp4").lower()
    chunk_data = request.files.get("chunk")

    if not upload_id or not chunk_data:
        return jsonify({"error": "Saknar data"}), 400

    if upload_id not in pending_uploads:
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.close()
        pending_uploads[upload_id] = {
            "path": tmp.name, "received": set(),
            "total": total, "ext": ext,
        }

    info = pending_uploads[upload_id]
    # Append chunk at correct byte offset — we write sequentially so just append
    with open(info["path"], "ab") as f:
        f.write(chunk_data.read())
    info["received"].add(chunk_idx)

    return jsonify({"ok": True, "received": chunk_idx})


@app.route("/transcribe-assembled", methods=["POST"])
def transcribe_assembled():
    upload_id   = request.json.get("upload_id")
    language    = request.json.get("language") or None
    model_name  = request.json.get("model", "medium")
    diarize_flag = bool(request.json.get("diarize"))

    if language == "auto":
        language = None
    if model_name not in ALLOWED_MODELS:
        return jsonify({"error": "Okänd modell"}), 400
    if diarize_flag and not HF_TOKEN:
        return jsonify({"error": "Talaridentifiering är inte konfigurerad på servern (HF_TOKEN saknas)"}), 400

    info = pending_uploads.pop(upload_id, None)
    if not info:
        return jsonify({"error": "Okänd uppladdning"}), 404

    ext = info["ext"]
    if len(info["received"]) != info["total"]:
        return jsonify({"error": f"Ofullständig uppladdning: {len(info['received'])}/{info['total']} delar"}), 400

    is_video = ext in VIDEO_EXTENSIONS
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "model": model_name,
        "status_text": f"Väntar i kö ({job_queue.qsize() + 1})…",
    }

    job_queue.put((job_id, info["path"], language, model_name, is_video, diarize_flag))

    return jsonify({"job_id": job_id})


@app.route("/srt/<job_id>")
def download_srt(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Ej klar"}), 404

    filename = request.args.get("filename", "transkribering.srt")
    filename = os.path.basename(filename).replace('"', "").replace("\r", "").replace("\n", "")
    if not filename.lower().endswith(".srt"):
        filename += ".srt"

    resp = make_response(_build_srt(job))
    resp.headers["Content-Type"] = "text/plain; charset=utf-8"
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp


@app.route("/download-zip", methods=["POST"])
def download_zip():
    data = request.json or {}
    fmt = data.get("format", "txt")
    items = data.get("items", [])

    if fmt not in ("txt", "srt"):
        return jsonify({"error": "Ogiltigt format"}), 400
    if not items:
        return jsonify({"error": "Inga filer valda"}), 400

    buf = io.BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            job = jobs.get(item.get("job_id"))
            if not job or job.get("status") != "done":
                continue

            name = _sanitize_filename(item.get("filename"), fmt)
            base, dot_ext = name.rsplit(".", 1)
            n = 2
            while name in used_names:
                name = f"{base}_{n}.{dot_ext}"
                n += 1
            used_names.add(name)

            content = job["text"] if fmt == "txt" else _build_srt(job)
            zf.writestr(name, content)

    if not used_names:
        return jsonify({"error": "Inga klara filer att ladda ner"}), 400

    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"] = "application/zip"
    resp.headers["Content-Disposition"] = 'attachment; filename="transkriberingar.zip"'
    return resp


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Okänt jobb"}), 404
    return jsonify(job)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3005, threaded=True)

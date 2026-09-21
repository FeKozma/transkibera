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

# "Cleanup" step: turns a finished transcript into a formal meeting protocol via
# the Anthropic API. Off unless ANTHROPIC_API_KEY is set.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
ANTHROPIC_HINT_MODEL = os.environ.get("ANTHROPIC_HINT_MODEL", "claude-haiku-4-5-20251001")
# Safety cap on transcript size sent per API call.
MAX_TRANSCRIPT_CHARS = 120_000

_anthropic_client = None

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


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic  # imported lazily so the app still starts if unused

        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _cap_text(text: str, limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…[avkortat]"


def _named_transcript(job: dict, speaker_names: dict) -> str:
    lines = []
    for seg in job.get("segments", []):
        speaker = seg.get("speaker")
        if speaker:
            name = (speaker_names.get(speaker) or "").strip() or speaker
            lines.append(f"{name}: {seg['text']}")
        else:
            lines.append(seg["text"])
    return "\n".join(lines)


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


@app.route("/speaker-hints/<job_id>", methods=["POST"])
def speaker_hints(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Jobbet är inte klart"}), 404
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "AI-städning är inte konfigurerad på servern (ANTHROPIC_API_KEY saknas)"}), 400
    if not job.get("speaker_colors"):
        return jsonify({"error": "Talaridentifiering användes inte för den här filen"}), 400

    if job.get("speaker_hints"):
        return jsonify({"speakers": job["speaker_hints"]})

    transcript = _cap_text(job["text"])
    prompt = (
        "Nedan är ett mötestranskript där talarna bara är märkta generiskt "
        "(Talare 1, Talare 2, osv) eftersom de inte kunde kännas igen automatiskt. "
        "Hjälp en mötessekreterare att lista ut vem som är vem: för varje unik "
        "talaretikett i transkriptet, sammanfatta i en kort mening vad den personen "
        "huvudsakligen pratade om eller bidrog med, och plocka ut 1-2 korta ordagranna "
        "citat som hjälper någon att känna igen personen. Svara på svenska.\n\n"
        f"Transkript:\n{transcript}"
    )

    try:
        client = get_anthropic_client()
        response = client.messages.create(
            model=ANTHROPIC_HINT_MODEL,
            max_tokens=2000,
            tools=[{
                "name": "report_speaker_hints",
                "description": "Report identifying info for each speaker label found in the transcript.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "speakers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "description": "Exact speaker label as it appears in the transcript, e.g. 'Talare 1'",
                                    },
                                    "topic_summary": {
                                        "type": "string",
                                        "description": "One short Swedish sentence describing what this person mainly talked about or contributed.",
                                    },
                                    "quotes": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "1-2 short verbatim Swedish quotes from this speaker.",
                                    },
                                },
                                "required": ["label", "topic_summary", "quotes"],
                            },
                        },
                    },
                    "required": ["speakers"],
                },
            }],
            tool_choice={"type": "tool", "name": "report_speaker_hints"},
            messages=[{"role": "user", "content": prompt}],
        )
        tool_use = next(b for b in response.content if b.type == "tool_use")
        speakers = tool_use.input.get("speakers", [])
    except Exception as e:
        return jsonify({"error": f"AI-anropet misslyckades: {e}"}), 502

    job["speaker_hints"] = speakers
    return jsonify({"speakers": speakers})


@app.route("/summarize/<job_id>", methods=["POST"])
def summarize(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Jobbet är inte klart"}), 404
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "AI-städning är inte konfigurerad på servern (ANTHROPIC_API_KEY saknas)"}), 400

    data = request.json or {}
    speaker_names = data.get("speaker_names") or {}
    meeting_date = (data.get("meeting_date") or "").strip()
    meeting_type = (data.get("meeting_type") or "").strip() or "Styrelsemöte"
    chairperson = (data.get("chairperson") or "").strip()
    secretary = (data.get("secretary") or "").strip()
    extra_context = (data.get("extra_context") or "").strip()

    transcript = _cap_text(_named_transcript(job, speaker_names))

    details = [f"Mötestyp: {meeting_type}"]
    if meeting_date:
        details.append(f"Mötesdatum: {meeting_date}")
    if chairperson:
        details.append(f"Mötesordförande: {chairperson}")
    if secretary:
        details.append(f"Sekreterare: {secretary}")
    if extra_context:
        details.append(f"Övrig information från sekreteraren: {extra_context}")

    prompt = (
        "Du är sekreterare för en ideell förening (t.ex. en löparförening) i Sverige "
        "och ska skriva ett formellt mötesprotokoll utifrån mötestranskriptet nedan.\n\n"
        + "\n".join(details) + "\n\n"
        "Skriv protokollet med följande struktur och i den ordningen:\n"
        "1. Rubrik med \"Mötesprotokoll\", mötestyp och datum.\n"
        "2. Närvarande: lista de personer som förekommer i transkriptet (namngivna, "
        "eller \"Talare N\" om de inte namngetts) samt eventuella extra deltagare från "
        "övrig information.\n"
        "3. §1 Mötets öppnande.\n"
        "4. Numrerade paragrafer (§2, §3, …) för varje sakfråga som togs upp, med en "
        "kort saklig sammanfattning av diskussionen. Skriv en tydlig rad som börjar med "
        "\"Beslut:\" under en paragraf om ett beslut fattades där.\n"
        "5. Ett avsnitt \"Åtgärdspunkter\" som listar vem som ska göra vad, och eventuell "
        "deadline om den nämndes.\n"
        "6. Sista paragrafen: Mötets avslutande.\n\n"
        "Regler: Skriv på svenska. Använd bara information som faktiskt förekommer i "
        "transkriptet eller uppgifterna ovan — hitta inte på beslut, namn eller datum. "
        "Var koncis men fullständig; det här protokollet ska kunna användas som det "
        "riktiga, officiella protokollet för mötet.\n\n"
        f"Transkript:\n{transcript}"
    )

    try:
        client = get_anthropic_client()
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = "".join(b.text for b in response.content if b.type == "text").strip()
    except Exception as e:
        return jsonify({"error": f"AI-anropet misslyckades: {e}"}), 502

    job["summary"] = summary
    return jsonify({"summary": summary})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Okänt jobb"}), 404
    return jsonify(job)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3005, threaded=True)

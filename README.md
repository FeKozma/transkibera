# Transkibera

A self-hosted web app for transcribing audio and video files, built on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Upload a file,
get a transcript back in the browser, and export it as SRT.

## Features

- Drag-and-drop upload of common audio/video formats (mp3, mp4, wav, m4a,
  ogg, flac, webm, mkv, avi, mov, wma, aac, and more)
- Choice of Whisper model size (`base`, `medium`, `large-v3-turbo`)
- Optional speaker diarization via
  [pyannote.audio](https://github.com/pyannote/pyannote-audio) — labels
  each segment with the speaker it overlaps most, color-coded consistently
  in both the web preview and the exported SRT
- Idle model eviction to keep memory usage low when the app isn't in use
- Runs in Docker, capped to a configurable CPU/RAM budget

## Running

```bash
docker compose up -d
```

The app listens on port 3005 by default (see `docker-compose.yml`).

### Speaker diarization

Diarization is optional and off by default. To enable it:

1. Accept the gated model terms on Hugging Face for
   [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0).
2. Create a Hugging Face access token.
3. Set it as `HF_TOKEN` in your environment before starting the container:

   ```bash
   HF_TOKEN=hf_xxx docker compose up -d
   ```

`pyannote`/`torch` are imported lazily, so idle memory use is unaffected
until diarization is actually used.

## Configuration

Environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `medium` | Default Whisper model to load |
| `WHISPER_COMPUTE` | `int8` | faster-whisper compute type |
| `WHISPER_CPU_THREADS` | `1` | CPU threads per transcription job |
| `WHISPER_IDLE_UNLOAD_SECONDS` | `600` | Seconds of inactivity before unloading a model |
| `HF_TOKEN` | unset | Hugging Face token, required for diarization |

## License

MIT

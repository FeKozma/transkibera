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
- Optional "Skapa protokoll" button that turns a finished transcript
  into a formal Swedish meeting protocol (mötesprotokoll) via the
  [Anthropic API](https://www.anthropic.com/api). When diarization was
  used, it first asks Claude to summarize what each anonymous speaker
  talked about and pulls a couple of representative quotes, so you can
  tell it "oh, that's Maria" instead of guessing from a bare speaker
  number — then writes up the protocol with those names attached
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

### Meeting protocol generation

The "Skapa protokoll" button is optional and off by default. To enable it, set
an [Anthropic API key](https://console.anthropic.com/) as `ANTHROPIC_API_KEY`
in your environment before starting the container:

```bash
ANTHROPIC_API_KEY=sk-ant-xxx docker compose up -d
```

Note that the transcript text (and, briefly, meeting details you type in
— chairperson, secretary, agenda notes) is sent to Anthropic's API to
generate the protocol.

## Configuration

Environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `medium` | Default Whisper model to load |
| `WHISPER_COMPUTE` | `int8` | faster-whisper compute type |
| `WHISPER_CPU_THREADS` | `1` | CPU threads per transcription job |
| `WHISPER_IDLE_UNLOAD_SECONDS` | `600` | Seconds of inactivity before unloading a model |
| `HF_TOKEN` | unset | Hugging Face token, required for diarization |
| `ANTHROPIC_API_KEY` | unset | Anthropic API key, required for the "Skapa protokoll" button |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | Model used to write the meeting protocol |
| `ANTHROPIC_HINT_MODEL` | `claude-haiku-4-5-20251001` | Cheaper model used to generate per-speaker hints |

## License

MIT

# MLI Synthetics — Phase 1

Synthetic data generation pipeline for the MLI lighting AI.

The MLI project trains a neural network to map **audio → stage lighting video**.
The real-data approach (harvesting YouTube concert footage) has hit a practical
ceiling at ~96 training samples. This pipeline produces *thousands* of
synthetic `(audio, lighting design)` pairs so the network can actually be
trained.

## Phase 1 scope

Three components in one repo:

1. **Stage Generator** — rule-based, randomized stage layouts (fixtures,
   positions, venue dimensions) driven by genre.
2. **Song Analyzer** — `librosa` numerical features + a local LLM that
   interprets them into a structured song analysis (genre, mood, structure,
   key moments, lighting hints).
3. **Designer LLM** — a local Ollama LLM acting as a senior lighting
   designer; produces a JSON cue list given the stage and song analysis.

Out of scope for Phase 1: rendering, MA3 integration, training pipeline
glue. The data structures are designed so Phase 2 can plug those in cleanly.

## Architecture

```
                              +---------------------+
audio file (.wav/.mp3) -----> | SongAnalyzer        |
                              |  librosa (numerical)|
                              |  Ollama (interpret) |
                              +----------+----------+
                                         | SongAnalysis (genre, mood, structure...)
                                         v
              +------------------+   +------------------+
              | StageGenerator   |-->| Phase1Pipeline   |
              |  (rule-based)    |   |  (orchestrator)  |
              +------------------+   +---------+--------+
                  | StageLayout                |
                  v                            v
              +---------------------------------------+
              | LightingDesignerLLM                   |
              |  knowledge base (PDFs/MD/TXT) +       |
              |  stage JSON + analysis JSON --> Ollama|
              +-----------------+---------------------+
                                | CueList (validated)
                                v
                          outputs/run_<ts>/
                            song_analysis.json
                            stage_layout.{json,txt}
                            cue_list.json
                            summary.{md,json}
```

## Setup

```powershell
# 1) Install Python deps via uv
uv sync

# 2) Pull the LLM via Ollama
ollama pull mistral-nemo:12b-instruct-2407-q4_K_M

# 3) Optional: copy .env.example -> .env and tweak if needed
copy .env.example .env

# 4) Drop knowledge files into src/data/knowledge_base/
#    Any combination of *.pdf, *.md, *.txt - filenames don't matter.
#    fixture_library.md is special (used directly by the stage generator).
```

## Usage

All scripts live under `src/scripts/` and run with `uv run python`.

### Generate a random stage layout

```powershell
uv run python src/scripts/generate_stage.py --genre edm --venue-size medium_club
uv run python src/scripts/generate_stage.py --random --seed 42
uv run python src/scripts/generate_stage.py --genre rock --output stage.json
```

### Analyze a song

```powershell
# Numerical features only (no Ollama needed)
uv run python src/scripts/analyze_song.py mysong.wav --skip-llm --output analysis.json

# Full analysis (requires Ollama running + model pulled)
uv run python src/scripts/analyze_song.py mysong.wav --output analysis.json
```

### Design a show from existing stage + analysis

```powershell
uv run python src/scripts/design_show.py --stage stage.json --analysis analysis.json --output cues.json
```

### Run the full pipeline end-to-end

```powershell
uv run python src/scripts/run_pipeline.py mysong.wav
uv run python src/scripts/run_pipeline.py mysong.wav --force-genre edm --force-venue large_venue
```

Outputs land in `src/data/outputs/run_<timestamp>/`.

## Adding knowledge files

Drop any `.pdf`, `.md`, or `.txt` into `src/data/knowledge_base/`. The loader
walks the directory recursively and bundles everything into the designer
LLM's system prompt (capped at ~30k tokens, truncated proportionally per
source). Cached at `src/data/outputs/knowledge_context.txt`.

`fixture_library.md` is also parsed by the stage generator for
genre-distribution + position rules. If it is missing, a hardcoded minimal
fallback library is used.

## Configuration

Edit `.env` to override:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP endpoint |
| `OLLAMA_MODEL_DESIGNER` | `mistral-nemo:12b-instruct-2407-q4_K_M` | Designer model |
| `OLLAMA_MODEL_ANALYZER` | `mistral-nemo:12b-instruct-2407-q4_K_M` | Analyzer model |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Per-request timeout |
| `DESIGNER_TEMPERATURE` | `0.7` | Designer creativity |
| `ANALYZER_TEMPERATURE` | `0.3` | Analyzer determinism |

## Testing

```powershell
uv run pytest
```

19 unit tests cover stage generation invariants, audio feature extraction
on synthetic WAV, and cue-list validation with a stubbed Ollama client.

## Known limitations

- **No splitting for long songs.** Songs over ~6 minutes may exceed the
  practical generation budget at `max_tokens_designer=4000`. A chunked
  generation path is sketched out in `LightingDesignerLLM` but not yet
  wired in.
- **MD parsing of position rules is heuristic.** The fixture library
  parser uses keyword matching for "front truss", "back truss", etc.
  Edge cases fall back to hardcoded defaults.
- **Drop detection is rough.** It uses RMS + onset percentile heuristics;
  the LLM stage cleans this up downstream.
- **JSON resilience depends on the model.** Mistral-Nemo is fairly
  reliable in JSON mode, but the client strips ` ```json ` fences and
  prose around the object as a safety net.

## Phase 2 roadmap

- Renderer: stage layout + cue list → video frames (Blender or three.js
  headless).
- MA3 integration: emit `.show` files or OSC streams.
- Training pipeline glue: package `(audio, video_frames)` pairs into the
  MLI dataset format.
- Quality control: an automated reviewer LLM that scores generated shows
  before they enter the training set.

## Project layout

```
src/
  mli_synthetics/
    settings.py       # Pydantic settings (.env)
    errors.py         # Exception hierarchy
    logging_config.py # loguru setup
    stage/            # rule-based stage generator
    audio/            # librosa + LLM song analysis
    llm/              # Ollama client, prompts, knowledge loader, designer
    pipeline/         # Phase1Pipeline orchestrator
  scripts/            # Typer CLIs
  data/
    knowledge_base/   # PDFs/MD/TXT (user-provided)
    outputs/          # generated runs
tests/                # pytest suite
```

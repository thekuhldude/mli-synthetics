# MLI Synthetics — Tutorial

This tutorial walks through what each piece of the Phase 1 pipeline does,
how the data flows between them, and how to extend or debug each
component. Read top-to-bottom for the big picture, or jump to a section.

> **Goal of Phase 1:** produce `(song, lighting design)` pairs that can
> later be rendered into video frames and fed to the MLI training pipeline.
> The "design" here is a structured JSON cue list — not pixels yet.

---

## 1. The 30-second mental model

Three independent components, glued together by an orchestrator:

```
┌────────────┐    ┌────────────┐    ┌──────────────┐
│ Stage Gen  │    │ Audio      │    │ Designer LLM │
│ (random)   │    │ Analyzer   │    │ (Mistral-Nemo│
│            │    │ librosa+LLM│    │  via Ollama) │
└─────┬──────┘    └─────┬──────┘    └──────┬───────┘
      │ StageLayout     │ SongAnalysis     │ CueList
      └────────────┬────┴───────────┬──────┘
                   ▼                ▼
              Phase1Pipeline.generate_show()
                   │
                   ▼
           outputs/run_<timestamp>/
              song_analysis.json
              stage_layout.json
              stage_layout.txt   (ASCII viz)
              cue_list.json
              summary.md
```

Each component can be used **on its own** (separate CLIs), or together
through `run_pipeline.py`.

---

## 2. The pieces

### 2.1 Settings, errors, logging — the boring foundation

- [`settings.py`](src/mli_synthetics/settings.py) — Pydantic settings,
  reads `.env`. Holds Ollama URL, model names, default temperatures,
  fixture-count bounds, and resolved paths to `knowledge_base/` and
  `outputs/`.
- [`errors.py`](src/mli_synthetics/errors.py) — single exception
  hierarchy. Catch `MLISyntheticsError` to catch everything we raise.
- [`logging_config.py`](src/mli_synthetics/logging_config.py) — loguru,
  configured on first use. INFO to stderr, DEBUG to rotating files when
  a log dir is passed.

You normally never touch these directly. They are imported lazily by
the other modules.

### 2.2 Stage Generator — `src/mli_synthetics/stage/`

Produces a random but **musically-coherent** stage layout.

- [`fixtures.py`](src/mli_synthetics/stage/fixtures.py) — Pydantic
  models. `FixtureCategory` (12 types), `Position` (8 placement zones),
  `Fixture` (id + category + position + xyz + rotation), and
  `StageLayout` (the whole rig + venue metadata).
- [`library.py`](src/mli_synthetics/stage/library.py) — parses
  `fixture_library.md` into structured rules. Extracts per-category
  typical positions, music pairing notes, etc. Falls back to hardcoded
  defaults if the MD file is missing or unparseable. Also owns the
  genre→category distribution table (e.g. EDM = 40 % beams, acoustic =
  40 % LED pars).
- [`generator.py`](src/mli_synthetics/stage/generator.py) — the
  algorithm itself:

```
1. Pick a venue size (small_club → festival, weighted).
2. Draw stage dimensions from that venue's range.
3. Draw a target fixture count from that venue's range.
4. Look up the genre distribution → category counts.
5. For each category, look up allowed positions, then place fixtures
   within each position's "zone" of (x, y, z) bounds.
   - Truss zones use mirrored pairs for symmetry.
   - Floor / side zones use even spacing.
   - All placements get small (-5cm, +5cm) jitter to avoid grid look.
6. If beams / spots / lasers exist but no hazer → inject one hazer.
7. Validate: no fixtures below floor, outside width, etc.
```

Run it standalone:

```powershell
uv run python src/scripts/generate_stage.py --genre edm --venue-size medium_club
```

You'll see both the JSON layout and an ASCII top-down view like:

```
============================================================
 Stage 12.5m x 5.1m x 4.5m  | venue=medium_club | genre=edm
============================================================
 back_truss      : [I] [I]
 mid_truss       : [B] [B] [I] [B]
 stage_floor     : [B] [W] [W] [W] [I] [B] [B] [W]
 side_left       : [X] [W] [L] [X]
 side_right      : [X] [W] [L] [X]
 front_truss     : [B] [B] [B] [B] [B]
============================================================
 Legend: B=Beam, W=Wash, L=Blinder, X=Strobe, I=Pixel Bar
```

### 2.3 Audio Analyzer — `src/mli_synthetics/audio/`

Two-stage: numerical features first, then an LLM that interprets them
into something a lighting designer can act on.

#### Stage A — numerical (librosa)

[`features.py`](src/mli_synthetics/audio/features.py) extracts:

| Field | What it is |
|---|---|
| `tempo_bpm`, `beat_times` | from `librosa.beat.beat_track` |
| `onset_times`, `onset_density_per_s` | `librosa.onset.onset_detect` |
| `rms_curve`, `rms_times` | RMS energy, downsampled to ~1 Hz |
| `spectral_centroid_curve` | timbral brightness, ~1 Hz |
| `structural_segments` | clustered chroma boundaries from `librosa.segment.agglomerative` |
| `key` | dominant pitch from chroma CQT |
| `energy_peaks` | timestamps where RMS > 80th percentile |
| `drop_candidates` | high-RMS + high-onset regions ≥ 4 s |

These are all pure numbers — no LLM, no Ollama dependency. The
`--skip-llm` flag stops here.

#### Stage B — LLM interpretation

[`analyzer.py`](src/mli_synthetics/audio/analyzer.py)'s
`SongAnalyzer._llm_interpret` formats the numerical features into a
text summary and sends them to the analyzer LLM. The system prompt
([`prompts.py`](src/mli_synthetics/llm/prompts.py) → `ANALYZER_SYSTEM_PROMPT`)
asks for a structured JSON:

```json
{
  "genre_estimate": "edm | rock | pop | acoustic | hip_hop | metal | other",
  "energy_profile": "low | medium | high | variable",
  "mood": "energetic | melancholic | aggressive | uplifting | dark | euphoric",
  "structure": [
    {"start_s": 0.0, "end_s": 16.0, "label": "intro", "energy": 0.3, "description": "..."}
  ],
  "key_moments": [{"time_s": 48.5, "type": "drop"}],
  "lighting_hints": {
    "color_palette": "warm | cool | neutral | contrast",
    "movement_intensity": "static | subtle | moderate | high | chaotic",
    "strobe_appropriate": true,
    "beam_appropriate": true
  }
}
```

The output is validated against [`structure.py`](src/mli_synthetics/audio/structure.py)'s
`SongAnalysis` Pydantic model. If the LLM call fails (Ollama down, JSON
parse error, model not pulled), `SongAnalyzer.analyze` returns
`SongAnalysisResult(numerical=..., interpreted=None)` — **graceful
degradation**, not a crash. The pipeline orchestrator does require the
interpreted analysis, so the pipeline will fail loudly there, but the
analyzer used on its own keeps working.

Run it:

```powershell
# Numerical only — fast, no Ollama
uv run python src/scripts/analyze_song.py mysong.wav --skip-llm --output analysis.json

# Full
uv run python src/scripts/analyze_song.py mysong.wav --output analysis.json
```

### 2.4 LLM stack — `src/mli_synthetics/llm/`

Four files, each with a single job.

- **[`ollama_client.py`](src/mli_synthetics/llm/ollama_client.py)** —
  async httpx wrapper around Ollama's `/api/generate`. Tenacity retries
  HTTP errors (3 attempts, exponential backoff). On 404 it raises
  `OllamaModelNotFoundError` with a `ollama pull <model>` hint. In JSON
  mode it sets `format=json` on the request, strips ` ```json ` fences
  from the response, trims any prose around the outermost `{...}`, and
  retries once if `json.loads` still fails. Every exchange (model,
  truncated system + prompt, response preview) is logged to
  `outputs/llm_logs/<timestamp>_<model>.json` — invaluable for
  debugging weird outputs.

- **[`prompts.py`](src/mli_synthetics/llm/prompts.py)** — every prompt
  the project sends, as Python constants. **No prompts live elsewhere.**
  If you want to tune the designer's behavior, this is the only file
  you edit. The designer's system prompt has a `{knowledge_context}`
  placeholder that gets filled in from the knowledge base.

- **[`knowledge.py`](src/mli_synthetics/llm/knowledge.py)** — walks
  `src/data/knowledge_base/` recursively for `*.pdf`, `*.md`, `*.txt`.
  PDFs go through `pypdf` page-by-page (pages with < 50 chars are
  skipped — likely diagrams). Everything is wrapped with
  `====== SOURCE: <filename> ======` separators and capped at ~30k
  tokens (chars / 4). The result is cached at
  `outputs/knowledge_context.txt`; the cache key is a SHA-256 of
  filenames + mtimes + sizes, so dropping a new PDF in invalidates it
  automatically.

- **[`designer.py`](src/mli_synthetics/llm/designer.py)** — the heart
  of Phase 1. `LightingDesignerLLM.design_show(stage, song_analysis)`
  builds the system prompt (with knowledge context) and user prompt
  (with stage + analysis JSON), calls Ollama in JSON mode, parses, and
  runs `_validate_constraints`:

  - Non-empty cue list.
  - `time_s` monotonically increasing.
  - `duration_s > 0`.
  - **Every fixture from the stage appears in every cue**
    (intensity 0 if off — this is the contract the renderer will rely
    on).
  - No unknown fixture IDs.
  - First cue near t=0, last cue covers within 5 s of song duration.

  On the first failure the user prompt is re-sent with
  "Previous attempt failed because: …" appended; on the second failure
  it raises `DesignerError`. This is deliberate — the LLM gets one
  guided retry, not infinite ones.

#### CueList schema (what Phase 2 will render)

```json
{
  "metadata": {
    "designer_notes": "...",
    "color_palette": ["amber", "deep_blue", "magenta"],
    "intensity_arc": "..."
  },
  "cues": [
    {
      "time_s": 0.0,
      "duration_s": 4.0,
      "section_label": "intro",
      "description": "Slow amber wash builds from upstage",
      "fixture_states": [
        {
          "fixture_id": "moving_beam_back_truss_1",
          "intensity": 0.0,
          "color": [255, 180, 80],
          "movement": {"pan": 0.0, "tilt": 0.5, "movement_type": "static", "speed": "slow"},
          "effect": "none",
          "effect_speed_hz": 0.0
        }
      ]
    }
  ]
}
```

### 2.5 Pipeline orchestrator — `src/mli_synthetics/pipeline/`

[`orchestrator.py`](src/mli_synthetics/pipeline/orchestrator.py) is the
glue:

1. `SongAnalyzer.analyze(audio)` → `SongAnalysisResult`
2. Pick a genre (`force_genre` or `interpreted.genre_estimate`)
3. `StageGenerator(...).generate(genre, venue_size)` → `StageLayout`
4. `LightingDesignerLLM.design_show(stage, analysis)` → `CueList`
5. Write everything + a `summary.md` + a `summary.json` with timings.

Each step is independently timed (`timings_s.analyze`,
`timings_s.stage`, `timings_s.design`) so you can profile.

Run it:

```powershell
uv run python src/scripts/run_pipeline.py mysong.wav
```

---

## 3. End-to-end run, narrated

Let's trace what happens when you run:

```powershell
uv run python src/scripts/run_pipeline.py "C:\path\to\song.wav" --force-genre edm
```

1. **Logging configured** (loguru → stderr at INFO).
2. **`Phase1Pipeline()` created.** Settings loaded; `outputs/` and
   `knowledge_base/` directories ensured.
3. **`analyzer.analyze(audio)` runs:**
   - `librosa.load` → mono 22050 Hz waveform.
   - Tempo + beats + onsets + RMS + centroid + chroma + segments + key.
   - `_format_features_summary` produces a ~15-line text block.
   - `OllamaClient.generate(model=mistral-nemo..., json_mode=True)`
     fires off the analyzer prompt. Response is parsed into
     `SongAnalysis`.
   - Saved to `song_analysis.json`.
4. **Genre resolved.** Since we passed `--force-genre edm`, we ignore
   the LLM's `genre_estimate` and use `"edm"`.
5. **`StageGenerator.generate(target_genre='edm')` runs.** Picks a
   venue size (random weighted), draws dimensions and fixture count,
   distributes by category, places fixtures, validates. Returns a
   `StageLayout`. Saved to `stage_layout.json` + ASCII viz to
   `stage_layout.txt`.
6. **`LightingDesignerLLM.design_show(stage, analysis)` runs.**
   - `build_knowledge_context()` reads PDFs/MD/TXT from
     `knowledge_base/` (cached). At ~28k tokens you'll see a `INFO`
     line confirming.
   - System prompt = `DESIGNER_SYSTEM_PROMPT.format(knowledge_context=...)`.
   - User prompt = stage JSON + analysis JSON.
   - One Ollama call, JSON mode, temperature 0.7,
     `num_predict=max_tokens_designer (4000)`.
   - Response parsed → `CueList` Pydantic model →
     `_validate_constraints`. On failure, one retry with explicit
     error feedback.
   - Saved to `cue_list.json`.
7. **`summary.md` + `summary.json` written.**

Look in `src/data/outputs/run_<timestamp>/` for everything. Look in
`src/data/outputs/llm_logs/` for the raw LLM exchanges (helpful when
something looks off).

---

## 4. Extending the pipeline

### Add a new genre

1. Add a row to `_DEFAULT_GENRE_DISTRIBUTIONS` in
   [`library.py`](src/mli_synthetics/stage/library.py).
2. Add an entry to `_GENRE_WEIGHTS` in
   [`generator.py`](src/mli_synthetics/stage/generator.py) if you want
   it to appear in random selection.
3. Add the literal to `genre_estimate` in
   [`structure.py`](src/mli_synthetics/audio/structure.py) so the
   analyzer LLM can emit it.

### Add a new fixture category

1. Add to `FixtureCategory` in
   [`fixtures.py`](src/mli_synthetics/stage/fixtures.py).
2. Add a section to `fixture_library.md` under
   `src/data/knowledge_base/`. Use the existing format
   (`## N. NAME`, `**Description:**`, `**Typical Position:**`, etc.).
3. Add to `_HEADING_TO_CATEGORY`, `_DEFAULT_POSITION_RULES`, and the
   legend tables (`_LEGEND_SYMBOLS`, `_LEGEND_NAMES`).

### Tune designer behavior

Edit `DESIGNER_SYSTEM_PROMPT` in
[`prompts.py`](src/mli_synthetics/llm/prompts.py). The "Hard rules"
list is where you encode global behavioral constraints (no strobe over
8 bars, no lasers without haze, etc.). The schema portion in the user
prompt is what locks the output shape — if you change it, also update
the matching Pydantic models in
[`designer.py`](src/mli_synthetics/llm/designer.py).

### Swap the LLM

Set `OLLAMA_MODEL_DESIGNER` / `OLLAMA_MODEL_ANALYZER` in `.env`. Any
Ollama model that supports `format=json` will work. Smaller models are
faster but worse at the strict per-cue-per-fixture coverage rule —
expect to lower `max_tokens_designer` and shorten the prompt.

---

## 5. Debugging

| Symptom | Where to look |
|---|---|
| `OllamaConnectionError` | `ollama serve` running? `OLLAMA_BASE_URL` correct? |
| `OllamaModelNotFoundError` | run `ollama pull <name>`. The error tells you which. |
| `DesignerError: missing fixtures` | LLM didn't emit a state for every fixture. Lower fixture count, raise `max_tokens_designer`, or shorten the song. Look at the actual response in `outputs/llm_logs/`. |
| `DesignerError: not monotonic` | LLM emitted cues out of order. The auto-retry usually fixes this. |
| `AudioAnalysisError` | unsupported audio format or corrupted file. librosa relies on `soundfile` / `audioread`. |
| Stage gen looks weird (everything on one side) | the fixture library MD's "Typical Position" text was parsed as side-left only — generator now mirrors that into both sides, but if you see a regression, check `_infer_positions` in `library.py`. |
| Designer outputs prose instead of JSON | check `llm_logs/<timestamp>_*.json`. The client strips fences and prose, but truly off-format outputs trigger one retry then raise. Lower temperature in `.env`. |

LLM logs are the single best debugging tool. Every call writes a JSON
file containing the model, options, truncated prompt, and a 2000-char
response preview.

---

## 6. Testing

```powershell
uv run pytest
```

What's covered:

- **`test_stage_generator.py`** — fixture-count bounds, in-bounds
  placement, hazer-when-beams invariant, genre-distribution dominance,
  rejected bad venue sizes.
- **`test_audio_analyzer.py`** — synthetic WAV generation, numerical
  feature extraction sanity, missing-file errors, JSON-serializable
  summary, `--skip-llm` returns numerical-only, broken Ollama degrades
  to numerical-only.
- **`test_designer_llm.py`** — cue-list validation accepts a valid
  list, rejects missing fixtures / unknown fixtures / non-monotonic
  times / empty list / short coverage; retry succeeds when first
  attempt fails; raises after exhausting retries.

All tests run offline (synthetic WAV, stubbed Ollama).

---

## 7. What Phase 2 needs from Phase 1

Phase 2's renderer will consume:

- `stage_layout.json` to instantiate fixtures in 3D space.
- `cue_list.json` to drive per-frame fixture state interpolation.
- `summary.json` for run-level metadata (genre, mood, color palette).

The contracts to preserve in any future Phase 1 refactor:

1. **Every fixture in `stage_layout.fixtures` appears in every cue's
   `fixture_states`** (intensity 0 if off). The renderer expects to
   never have to invent state.
2. **`time_s` is monotonic and starts ≈ 0.** No gaps; durations
   describe the time the cue is "active" until the next cue.
3. **Colors are RGB 0-255 ints**, intensities are floats 0.0-1.0,
   pan/tilt are normalized -1.0 to 1.0. Renderer will denormalize per
   fixture type.

Both `LightingDesignerLLM._validate_constraints` and the test suite
enforce these.

---

## 8. Quick reference

```powershell
# All four CLIs
uv run python src/scripts/generate_stage.py --help
uv run python src/scripts/analyze_song.py --help
uv run python src/scripts/design_show.py --help
uv run python src/scripts/run_pipeline.py --help

# Tests
uv run pytest

# Health-check Ollama from Python
uv run python -c "import asyncio; from mli_synthetics.llm.ollama_client import OllamaClient; print(asyncio.run(OllamaClient().list_models()))"
```

Happy generating.

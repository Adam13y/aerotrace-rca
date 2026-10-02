# AeroTrace-RCA

**Multi-agent AI system for rocket telemetry anomaly detection and Root Cause Analysis (RCA).**

AeroTrace-RCA ingests launch-vehicle telemetry (CSV, JSON or FITS), detects anomalies with robust statistics,
triages them with a local GPU model (Ollama / Qwen), reasons about root causes with Claude, and produces a
professional Markdown RCA report with a timeline, ranked hypotheses, recommendations and plots.

Every LLM step degrades gracefully. If Ollama is down, local work goes to Claude. If Claude is not configured,
supervisor reasoning runs on the local model. With no LLM at all, deterministic, knowledge-base-driven agents
still produce a complete report.

---

## Architecture

```
                         ┌──────────────────────────────────────────────────────┐
  telemetry file ──────► │        SUPERVISOR  (src/agents/supervisor.py)        │
  (.csv .json .fits)     │  orchestration · retries · fallbacks · decision log  │
                         └──────────────────────────────────────────────────────┘
                                │
     ┌──────────────┬───────────┼─────────────────┬────────────────────┬──────────────────┐
     ▼              ▼           ▼                 ▼                    ▼                  ▼
 ┌─────────┐  ┌───────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ ┌──────────────┐
 │Ingestion│  │ Telemetry │ │   Anomaly    │ │   Anomaly    │ │   Hypothesis     │ │    Report    │
 │CSV/JSON/│─►│  Analyst  │►│  Detector    │►│   Triage     │►│   Generator      │►│    Writer    │
 │  FITS   │  │ stats,    │ │ z-score, MA, │ │ LOCAL Ollama │ │ CLOUD Claude     │ │ Markdown,    │
 │normalise│  │ phases,   │ │ derivative,  │ │ (Qwen 2.5)   │ │ (+ KB context)   │ │ JSON, plots, │
 └─────────┘  │ correlate │ │ limits       │ │ ↳ Claude     │ │ ↳ Ollama         │ │ quick-look   │
              └───────────┘ └──────────────┘ │ ↳ rules      │ │ ↳ KB rules       │ └──────────────┘
                                             └──────────────┘ └────────┬─────────┘
                                                                       │ similarity search / store
                                                              ┌────────▼─────────┐
                                                              │  Knowledge Base  │
                                                              │ ChromaDB vectors │
                                                              │ failure modes +  │
                                                              │ past anomalies   │
                                                              └──────────────────┘
```

| Agent | Where it runs | Responsibility |
|---|---|---|
| **Telemetry Analyst** | local, deterministic | Per-sensor stats, data-quality notes (gaps, imputation, stuck sensors), flight-phase tagging, cross-sensor correlation |
| **Anomaly Detector** | local, deterministic + **Ollama** | Robust z-score / moving-average / derivative / limit detection, severity grading, LLM triage (category, subsystem, sensor-fault screening) |
| **Hypothesis Generator** | **Claude** (Ollama fallback) | Fault-tree style root-cause hypotheses with confidence, evidence, counter-evidence and actions, grounded in knowledge-base matches |
| **Report Writer** | local (+ Claude executive summary) | Full RCA report, quick-look summary, JSON export, telemetry plot |
| **Supervisor** | local | Runs the pipeline, routes between local and cloud models, retries with back-off, applies fallbacks, writes an auditable decision log |

---

## Installation

Requirements: **Python 3.11+**. Ollama and an Anthropic API key are optional but recommended.

```bash
git clone <your-repo-url> AeroTrace-RCA
cd AeroTrace-RCA
python scripts/setup_env.py            # creates .venv, installs requirements, creates .env
```

Or manually:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                   # then add ANTHROPIC_API_KEY
```

Local model (optional, ~9 GB, needs a GPU with about 12 GB VRAM for good speed):

```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
```

> The agent orchestration is built into the supervisor; no agent framework (CrewAI, LangChain) is required.

---

## Docker (zero-setup)

Everything runs locally: the app container plus an Ollama container with **Qwen 2.5 14B and nomic-embed-text baked
into the image**, so nothing is downloaded at run time and no API keys are needed.

```bash
docker compose build                       # once: pulls ~9.5 GB of models into the Ollama image
docker compose run --rm aerotrace          # full RCA on data/raw/ (a sample flight is generated if empty)
docker compose run --rm aerotrace python -m src.main data/raw/my_flight.csv --report
docker compose run --rm aerotrace python scripts/run_pipeline.py --kb-stats
```

- **Input and output:** put telemetry files in `./data/raw/`; reports, plots and decision logs appear in `./data/reports/`.
- **First run:** the app container builds the RAG vector store (about 1 minute) into a persistent volume. It is rebuilt
  automatically when `data/historical_failures/` changes.
- **GPU:** needs an NVIDIA GPU with the NVIDIA Container Toolkit (Linux) or Docker Desktop with WSL2 GPU support
  (Windows). Qwen 14B Q4 with a 10,240-token context fits in 12 GB of VRAM.
- **CPU only:** `docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm aerotrace` (much slower).
- **Settings:** `OLLAMA_REPRODUCIBLE`, `OLLAMA_NUM_CTX`, `OLLAMA_VERSION` (Ollama image tag, pinned to 0.34.4 by default) and the optional
  `ANTHROPIC_API_KEY` can be set in the shell or in `.env` next to `docker-compose.yml`.

## Usage

```bash
# One-command run: checks Ollama/Claude, generates sample data if needed, runs everything
python scripts/run_pipeline.py

# Generate the synthetic 300 s ascent dataset (NASA-style CSV, SpaceX-style CSV, channel JSON)
python scripts/download_sample_data.py

# Full RCA on a file (default mode is --report)
python -m src.main data/raw/sample_flight_nasa_long.csv --report

# Detection and triage only
python -m src.main data/raw/sample_flight_spacex_wide.csv --analyze

# Markdown + JSON, no LLM calls, debug logging with the supervisor decision log
python -m src.main data/raw/sample_flight_channels.json --format both --offline -v

# Offline end-to-end self-test on synthetic data with injected faults
python -m src.main --test

# Unit and integration tests
pytest
```

### Outputs (in `data/reports/`)

| File | Content |
|---|---|
| `rca_report_<stamp>.md` | Full RCA: summary, conclusion, data overview, timeline, findings, hypotheses, recommendations, methodology |
| `rca_summary_<stamp>.md` | One-page quick-look for fast review |
| `rca_report_<stamp>.json` | Machine-readable `RCAReport` (with `--format json` / `both`) |
| `rca_plots_<stamp>.png` | Per-sensor telemetry with shaded anomaly spans |
| `decision_log_<stamp>.json` | Supervisor decisions, provider routing, LLM call counts, errors |

The normalised input (ISO-8601 timestamps, long format) is written to `data/processed/`.

### Synthetic sample flight

`download_sample_data.py` generates a 300 s, 10 Hz ascent with engine start, a throttle bucket around Max-Q,
MECO at ~285 s, sensor noise and 0.4 % missing samples. It also injects these known faults:

| MET | Sensor | Fault |
|---|---|---|
| T+142 s | `lox_tank_pressure` | Pressurisation drop of 6 psi with slow recovery |
| T+143.5 s | `turbopump_vibration` | +6 g vibration burst (cascade) |
| T+145 s | `engine_bay_temp` | +35 °C thermal excursion (cascade) |
| T+210 s | `avionics_bus_voltage` | Under-voltage sag below the 26.5 V limit |
| T+230 s | `fuel_tank_pressure` | Single-sample spike, i.e. an instrumentation glitch |

---

## Configuration

### Environment (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Enables Claude (cloud supervisor reasoning) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen2.5:14b-instruct-q4_K_M` | Local triage model |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Claude model ID |
| `DATA_DIR` / `REPORT_DIR` | `./data`, `./data/reports` | Paths are relative to the project root |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, ... |
| `OLLAMA_TIMEOUT`, `CLAUDE_MAX_TOKENS` | `120`, `4096` | Request tuning |

### Agents (`config/agents.yaml`)

- `pipeline`: retries, back-off, global `use_llm` switch.
- `telemetry_analyst`: `correlation_window_seconds` and named `flight_phases` (MET seconds).
- `anomaly_detector`:
  - detector thresholds (`zscore_threshold`, `derivative_threshold`, `moving_average_threshold`)
  - severity cut-offs in sigma (`info_score`, `warning_score`, `critical_score`)
  - `window_seconds`, `merge_gap_seconds`
  - `sensor_limits`: hard limits, where any breach is CRITICAL
  - `planned_events`: known transients, which are downgraded to INFO and marked expected. Sensor names accept glob patterns.
  - `provider_preference`: `local` by default.
- `hypothesis_generator`: `provider_preference` (`cloud` by default), number of hypotheses, knowledge-base matches.
- `report_writer`: mission and vehicle names, plots, whether to use an LLM-written executive summary.

---

## Historical failure RAG

Before Qwen writes hypotheses, the pipeline retrieves the 3 most similar past aerospace failures and adds them
to the prompt as **Historical Context**. Qwen treats them as prior art, not as evidence about the current flight.

```bash
python scripts/acquire_aerospace_data.py     # renders the 102-case knowledge base from scripts/kb_sources/*.json
ollama pull nomic-embed-text                 # local embedding model (~270 MB)
python scripts/build_knowledge_base.py       # embeds data/historical_failures/**.txt into data/vector_store/
python scripts/run_pipeline.py --kb-stats    # cases by category and source type, on disk vs. indexed
```

- **Contents:** 82 *synthetic* failure-mode summaries (10 per category, 12 for propulsion: propulsion, pressurization, structural,
  thermal, electrical, avionics, separation, aerodynamic) and 20 *historical* summaries of real incidents
  (`historical/`). Synthetic cases are generic engineering write-ups, not records of real events. They name a real
  incident only where one genuinely exhibited the failure mode, and Qwen's prompt labels them as synthetic.
- **Retrieval:** the top 5 cases. Cases from the detected anomalies' failure categories get a similarity bonus
  (`historical_rag.category_boost` in `agents.yaml`), so they rank ahead of comparably similar cases from other categories.
- **Add cases:** edit `scripts/kb_sources/*.json` and re-run the acquisition script, or drop your own `.txt` files into
  `data/historical_failures/<category>/`. Start each file with `Title:`/`Category:`/`Failure mode:`/`Source type:` header lines
  and a blank line, then re-run the build script. Hand-added files are never removed by the acquisition script.
  Files are chunked at ~500 tokens with 50 tokens of overlap.
- **Graceful degradation:** if the store is missing or the embedding model is unavailable, the pipeline logs
  "Historical context skipped" and continues without RAG.
- **Context window:** the hypothesis prompt is about 7k tokens with RAG. Ollama's default 4,096-token window truncates
  it silently, so `OLLAMA_NUM_CTX` defaults to 10240, the largest window that keeps Qwen 14B Q4 fully on a 12 GB GPU.
  A warning is logged if a prompt comes close to filling the window.
- **Reproducibility:** even at temperature 0 with a fixed seed, Ollama's output depends on its KV-cache state.
  A prompt it has already cached can come back worded differently from a freshly computed one. With
  `OLLAMA_REPRODUCIBLE=true` (the default), Qwen is reloaded for every call, so identical input always gives identical
  output. Each call takes about 25-30 s longer. Set it to `false` for faster, slightly variable runs.

## Adding your own telemetry

Put files in `data/raw/` (or pass any path). The format is detected automatically.

**Long / NASA-style CSV:** one row per sample. Aliases such as `mnemonic`, `channel`, `eng_value` and `units` are accepted.

```csv
timestamp,sensor_id,value,unit
2026-03-01T14:00:00.000Z,lox_tank_pressure,50.12,psi
```

**Wide / SpaceX-style CSV:** one column per sensor, with units in the header.

```csv
T+ (s),Chamber Pressure (psi),Bus Voltage (V)
0.0,0.4,28.01
```

**JSON:** a list of records, `{"records": [...]}`, JSON-Lines, or channel documents:

```json
{"base_time": "2026-03-01T14:00:00Z",
 "sensors": {"lox_tank_pressure": {"unit": "psi", "samples": [[0.0, 50.1], [0.1, 50.0]]}}}
```

**FITS:** the first binary-table HDU, a `TIME`/`MET`/`TIMESTAMP` column, units from `TUNITn`, and the epoch from `DATE-OBS`.

Timestamps can be ISO-8601, epoch seconds/ms/µs/ns, or relative seconds (`12.5`, `T+12.5`). Relative times are
offset from `CSVParserConfig.base_time` or the file's own epoch. Missing values are interpolated in time by
default (`missing_strategy`: `interpolate`, `ffill` or `drop`) and flagged `imputed` in reading metadata.

After adding a new vehicle, update `flight_phases`, `planned_events` and `sensor_limits` in `agents.yaml`.
Otherwise engine start, throttling and cutoff will show up as anomalies.

---

## Customising agents

- **Prompts:** `CLASSIFIER_SYSTEM_PROMPT` (`anomaly_detector.py`), `SYSTEM_PROMPT` / `HYPOTHESIS_PROMPT_TEMPLATE`
  (`hypothesis_generator.py`), `EXEC_SUMMARY_SYSTEM` (`report_writer.py`).
- **Model routing:** set `provider_preference` per agent in `agents.yaml`, or build an `LLMRouter` with your own
  clients. Any object with `model`, `available()` and `chat(system, prompt, json_mode=..., temperature=...)` works.
- **Knowledge base:** add entries to `SEED_FAILURE_MODES` in `src/memory/knowledge_base.py`. They are upserted on
  start-up. Past anomalies are stored automatically after every report run (ChromaDB in
  `data/processed/knowledge_base/`, with a JSON fallback).
- **New detectors:** extend `AnomalyDetector._score_sensor` / `_detect_sensor` and add the method name to
  `detection_methods`.
- **New pipeline steps:** add a `self._step("name", func, attempts=..., fallback=...)` call in `Supervisor.run`.
  Retries, fallbacks and decision logging come for free.

---

## Project layout

```
config/        settings.py (pydantic-settings), agents.yaml
src/agents/    telemetry_analyst, anomaly_detector, hypothesis_generator, report_writer, supervisor
src/ingestion/ csv_parser (+ synthetic data generator), json_parser, fits_reader
src/memory/    vector_store (ChromaDB / JSON), knowledge_base
src/models/    telemetry_schema (Pydantic models)
src/utils/     logger (Rich), helpers (timestamps, retries, JSON extraction, Ollama/Claude clients, router)
scripts/       setup_env.py, download_sample_data.py, run_pipeline.py
tests/         pytest suite (fake LLM clients cover the fallback paths)
```

---

## Disclaimer

AeroTrace-RCA is an engineering decision-support tool. Its reports are preliminary and must be reviewed by
qualified engineers before any flight-safety or hardware disposition decision.

## License

MIT License

Copyright (c) 2026 AeroTrace-RCA contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

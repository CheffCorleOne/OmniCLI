# OmniCLI – Offline AI Terminal Assistant

OmniCLI is an adaptive AI agent that lives inside a Linux terminal and helps systems administrators translate natural-language intents into safe, optimized shell workflows. The project follows the OmniCLI v1.0 specification (02 Dec 2025) and is delivered as an offline-first CLI application that automatically adapts to the host hardware, understands the surrounding system context, and continuously learns from prior interactions.

## Vision

- **Hardware-aware** – detects CPUs, GPUs, accelerators, and tunes inference accordingly.  
- **Context-centric** – captures live system snapshots (processes, services, mounts, topology) before planning actions.  
- **Safe-by-design** – multi-layer validation, sandboxed dry-runs, and auditable execution trails.  
- **Self-improving** – learns from every command, surfacing better plans over time without leaving the host.  
- **Privacy-first** – operates fully offline, storing knowledge locally.

## Repository Layout

```
src/omnicli
├── app.py                    # High-level composition root
├── cli/                      # User-facing entry points (Typer-based)
├── core/                     # Adaptive core, context capture, orchestration
├── hardware/                 # Detection, optimization, allocation
├── ai/                       # Model registry, inference, and training hooks
├── knowledge/                # Vector stores, indexing, retrieval, data sources
├── nlp/                      # Intent understanding and command generation
├── execution/                # Safety layers, shell executors, monitors
├── data_pipeline/            # Collection, processing, storage, analytics
├── continuous_learning/      # Feedback loops and personalization engines
├── system_integration/       # File system, network, process, and package adapters
├── security/                 # Threat detection, privacy, compliance
├── packaging/                # Build, distribution, installation helpers
├── testing/                  # Harnesses, fixtures, QA tooling
├── documentation/            # Auto-docs, tutorials, helpers
└── utils/                    # Shared logging/error-handling abstractions
```

Each subpackage mirrors the technical specification and is intentionally modular so the system can evolve iteratively. Early commits focus on establishing interfaces, dependency flows, and mocked implementations that can be swapped for real hardware/model integrations later.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate          # (or .venv\Scripts\activate on Windows)
pip install --upgrade pip
pip install -e ".[ai,vector,security,testing]"
```

Run the CLI:

```bash
omnicli --help
omnicli run "сделай резервную копию проектов и отправь на сервер"
```

During early development most subsystems emit structured traces rather than executing destructive commands. The `--dry-run` flag (default) ensures only simulated plans are produced.

## Execution Stages & Logging

Every `omnicli run "<intent>"` request проходит через явные стадии:

1. **Hardware discovery** – сбор краткого профиля железа (`HardwareProfile`).  
2. **System snapshot** – снимок окружения (`SystemSnapshot`: пользователи, env, далее процессы/сервисы и т.д.).  
3. **Intent classification** – упрощённая классификация намерения пользователя.  
4. **Plan generation** – построение плана команд (пока rule‑based, позже модель + RAG).  
5. **Safety validation** – базовая проверка плана на опасные конструкции.  
6. **Execution** – сейчас *по умолчанию выключено* (режим dry‑run).

Все значимые стадии и изменения логируются через единый логгер `omnicli`:

- **INFO‑уровень**: отметки стадий (`discover_hardware`, `capture_system_snapshot`, `classify_intent`, `generate_plan`, `validate_plan`, `execute_plan/dry-run`).  
- **DEBUG‑уровень**: детали профиля/снапшота и сгенерированного плана.

Примеры:

```bash
omnicli run "сделай резервную копию проектов и отправь на сервер"
omnicli run "почему тормозит система" --log-level debug
```

Логи выводятся в человеко‑читаемом виде через `rich` и помогают отслеживать, на какой стадии находится запрос и что именно изменилось.

## Contributing

1. Create feature branches per subsystem (e.g., `feature/context-scanner`).  
2. Add or update unit tests in `tests/`.  
3. Run `ruff check .` and `pytest` before opening a PR.  
4. Document major changes in `CHANGELOG.md` (to be added).

## Roadmap Snapshot

- ✅ Project scaffolding with modular packages and type-safe interfaces.  
- 🚧 Hardware detection adapters + context snapshot pipeline.  
- 🔜 Model selection, safety sandbox, and experience database.  
- 🔜 Packaging targets (.deb, Snap, Docker) and offline installer.

Refer to the specification in `docs/specs/` (to be imported next) for the exhaustive plan.


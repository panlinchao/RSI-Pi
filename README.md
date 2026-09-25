# RSI-Pi

Source code for a recursive self-improvement experiment with the pi coding agent.
The driver compares two editable extension surfaces: a tool extension and an
execution-strategy extension. Pi proposes changes from development-task evidence;
the harness evaluates each version before deciding whether to keep it.

This repository contains the implementation, tests, and two example extensions.
Task transcripts, experiment results, and cost records are generated locally
when you run it.

## Code map

| Path | Role |
| --- | --- |
| `rsi/run_experiment.py` | Run a shared baseline, both improvement loops, and final evaluation. |
| `rsi/evolve.py` | Stage immutable versions, evaluate candidates, and select retained versions. |
| `rsi/evolver.py` | Let pi inspect evidence and propose a candidate in a container. |
| `rsi/harness.py`, `adapter/pi_pier_agent.py` | Run isolated task trials and record scores and usage. |
| `bench/fetch_ccbench.py` | Fetch a pinned CCBench revision and create a local task split. |
| `modules/` | Initial extension surfaces and fixed budget guard. |
| `examples/tools/index.ts`, `examples/exec/index.ts` | Example extensions produced by one run. |

## Set up

The Python imports currently use `open_research_RSI` as the package name, so
clone the repository into a directory with that name and run commands from its
parent directory. A new run needs Python 3.12, `uv`, Git, Docker, and a DeepSeek
API key.

```bash
git clone https://github.com/panlinchao/RSI-Pi.git open_research_RSI
uv venv --python python3.12 open_research_RSI/.venv
uv pip install --python open_research_RSI/.venv/bin/python -r open_research_RSI/requirements.txt
open_research_RSI/.venv/bin/python open_research_RSI/bench/fetch_ccbench.py
docker build -t rsi-evolver:0.86.1 open_research_RSI/bench/evolver
```

The fetch command writes `bench/manifest.json` and materializes the selected
tasks locally. Keep that manifest for the whole run: it fixes the development
and test split. The benchmark data and run records are ignored by Git.

Run local tests before using the model API:

```bash
open_research_RSI/.venv/bin/python -m unittest discover -s open_research_RSI/bench -p 'test_*.py' -v
open_research_RSI/.venv/bin/python -m unittest discover -s open_research_RSI/rsi -p 'test_*.py' -v
open_research_RSI/.venv/bin/python -m unittest discover -s open_research_RSI/adapter -p 'test_*.py' -v
```

After setting `DEEPSEEK_API_KEY`, run from the parent directory:

```bash
export PYTHONPATH="$PWD"
open_research_RSI/.venv/bin/python -m open_research_RSI.rsi.run_experiment --tag my-rsi-run
```

This command makes paid API calls. Each run has a tag and records its source
commit, container image, task split, and limits. Use a new tag when any of those
change. The implementation requires a clean committed Git checkout at run time.

## License

MIT. See [LICENSE](LICENSE).

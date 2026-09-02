# Progress Notes

Measurements and environment findings recorded as the build proceeds. Facts, not plans —
the plan lives in `BUILD_ORDER.md`.

---

## Step 1 — Repo scaffold (2026-09-02)

**The local toolchain assumed by `SETUP_CHECKLIST.md` §6a was not present.** The Mac had
no Homebrew, no Python 3.11 (system Python was 3.9.6), no Poetry, and no SSH key for
GitHub. Resolved without sudo:

- `uv` installed to `~/.local/bin`, and used to install CPython 3.11.16
- Poetry 2.4.2 installed with the official installer, using the uv interpreter
  (note: the installer must be pointed at the *real* interpreter path under
  `~/.local/share/uv/python/...`, not the `~/.local/bin/python3.11` symlink — a venv built
  from the symlink fails with `ModuleNotFoundError: No module named 'encodings'`)
- `export PATH="$HOME/.local/bin:$PATH"` appended to `~/.zshrc`
- The user generated an ed25519 key and added it to GitHub

**Resolved dependency versions** (from `poetry.lock`): darts 0.46.1,
chronos-forecasting 2.3.1, statsmodels 0.15.0, arch 8.0.0, torch 2.14.0,
transformers 5.16.1.

**A `.gitignore` bug caught before the first commit.** The pattern `data/` is unanchored
and therefore also matched `forecast_bench/data/`, which would have silently excluded the
entire data module from version control. Changed to `/data/`.

## Step 4 — Credential verification (2026-09-02)

Output of `poetry run python -m scripts.verify_setup`:

| Check | Result |
|---|---|
| FRED (`DGS10`) | PASS — last observation 2026-08-31 = 4.75 |
| Yahoo (SPY OHLC) | PASS — 10 bars, O/H/L/C all present |
| Chronos-2 on CPU | PASS — see latency below |
| Hugging Face access | PASS — authenticated as `rohanjain2312`, all 3 repos reachable |
| Space configuration | REVIEW — see hardware below |

### Chronos-2 CPU latency: **0.85 s** per forecast

Measured on Apple Silicon, `device_map="cpu"`, 512-step context to a 21-step quantile
forecast. Model load (cold, from local cache) takes a further 16.9 s and happens once at
Space startup, not per request.

**Consequence for the demo (DECISIONS.md D12):** comfortably under the 5 s threshold, so
the Space runs **live inference**. No caching layer and no pre-computed forecast grid are
needed. The Space's 2 vCPU will be slower than this machine, but there is roughly 6× of
headroom before the decision would change.

### Space hardware: **`zero-a10g`** (ZeroGPU) — needs changing to CPU Basic

The Space is on ZeroGPU, not CPU Basic. This is the one preflight item that was recorded
as unconfirmed, and it turns out to be wrong for this project.

Reading it required care. `SpaceRuntime.hardware` is `None` for a Space that has never
built (`stage=NO_APP_FILE`), because "current" hardware only exists once something has
run. The live setting is in `requested_hardware`. A check reading only `hardware` reports
`unknown` and passes the exact case it exists to catch, so `check_space_config` reads
`hardware or requested_hardware`.

Why this matters (DECISIONS.md D12): ZeroGPU charges GPU quota to the *visitor*, not the
owner. An unauthenticated visitor gets 2 GPU-minutes per day and queues behind PRO users.
Given the measured 0.85 s CPU latency, there is nothing to gain from a GPU and a real
failure mode to lose.

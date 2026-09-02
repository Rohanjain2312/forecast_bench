# forecast_bench — Manual Setup Checklist

> **Status, verified live 2026-09-02.** Steps 1–5 are **already complete** — HF PRO is
> active, the write token and FRED key exist, and all three Hugging Face repos plus the
> GitHub repo have been created. **Do not redo them.** They are kept below as reference for
> how each was set up and how to recover if a key is ever lost or revoked.
>
> The only steps still ahead are 6 (fill `.env`), 8 (Colab secrets), and 10 (Space
> hardware/variables). `docs/planning/MANUAL_TASKS.md` is the short, current version of
> what you actually have left to do — read that one first. This file is the detailed
> appendix.

Everything you have to click, generate, or paste before any code runs. Ordered by
dependency — do them top to bottom. Nothing here is done by code; all of it is done by you
in a browser or a terminal.

**The one rule that governs all of it:** a secret exists in exactly three places, and never
anywhere else. Local `.env` (gitignored), Colab Secrets, HF Space Secrets. Never in a
notebook cell, never in a config file, never in a commit.

---

## Step 1 — Hugging Face PRO subscription ✅ DONE

**Why first:** since July 2026, creating a Gradio Space requires a paid plan. Without this,
the required demo deliverable cannot exist. Every other HF step depends on the account
being in the right state.

1. Go to `https://huggingface.co/settings/billing/subscription`
2. Subscribe to **PRO** ($9/month)
3. Confirm at `https://huggingface.co/settings/billing` that the plan shows as active

**Verify:** go to `https://huggingface.co/new-space`. The **Gradio** SDK option should be
selectable and not marked as paid/locked. If it is still locked, the subscription has not
propagated — wait a few minutes and reload.

---

## Step 2 — Hugging Face access token (write scope) ✅ DONE

**Used for:** pushing the fine-tuned checkpoint, pushing the dataset, deploying the Space.

1. Go to `https://huggingface.co/settings/tokens`
2. **New token** → token type **Fine-grained** (preferred over Classic-write; it lets you
   scope the token to only the three repos this project touches)
3. Name: `forecast-bench-write`
4. Permissions — enable exactly these:
   - Repositories: **Write access to contents/settings** for
     `rohanjain2312/forecastbench-chronos`, `rohanjain2312/forecastbench-data`,
     `rohanjain2312/forecastbench-demo`
   - **Read access to contents of all public repos** (needed to pull `amazon/chronos-2`)
5. Create, then **copy the value immediately** — HF shows it once

**Where it goes:** three places, in Steps 6, 8, and 10 below. Copy it somewhere temporary
(a password manager entry, not a text file on the Desktop) until all three are done, then
delete the temporary copy.

**If you ever paste it into a notebook by accident:** revoke it at
`https://huggingface.co/settings/tokens` and regenerate. HF also runs a secrets scanner on
Spaces and will warn you, but do not rely on that.

---

## Step 3 — FRED API key ✅ DONE

**Used for:** `DGS10`, `DGS3MO`, `T10Y2Y`, `VIXCLS`, `DFF`.

1. Create a free FRED account at `https://fredaccount.stlouisfed.org/apikeys`
   (or start from `https://fred.stlouisfed.org/docs/api/api_key.html`)
2. Request an API key — it is issued instantly, no approval wait
3. Copy the 32-character key

**Where it goes:** local `.env` (Step 6) and Colab Secrets (Step 8). **Not** the Space —
the Space reads pre-computed data from the Hub and never calls FRED.

---

## Step 4 — GitHub authentication — handled by Claude Code in build Step 1

**Used for:** pushing commits from the Mac to `Rohanjain2312/forecast_bench`.

You almost certainly already have this configured from your other repos. Verify rather than
redo:

```bash
ssh -T git@github.com
# Expected: "Hi Rohanjain2312! You've successfully authenticated..."
```

If that fails, generate a key and add it:

```bash
ssh-keygen -t ed25519 -C "rohanjain2312@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
pbcopy < ~/.ssh/id_ed25519.pub
```

Then paste at `https://github.com/settings/keys` → **New SSH key**.

---

## Step 5 — Create the three Hugging Face repos ✅ DONE (all three exist and are empty)

Create them empty now, before any code, so the identifiers in `config.py` resolve from the
first commit.

**5a. Model repo**
1. `https://huggingface.co/new`
2. Owner `rohanjain2312`, name `forecastbench-chronos`, type **Model**, **Public**,
   license **Apache 2.0** (matching the Chronos base model's license — required, since
   fine-tuned derivatives inherit it)

**5b. Dataset repo**
1. `https://huggingface.co/new-dataset`
2. Owner `rohanjain2312`, name `forecastbench-data`, **Public**, license **MIT**

**5c. Space**
1. `https://huggingface.co/new-space`
2. Owner `rohanjain2312`, name `forecastbench-demo`, **Public**
3. SDK: **Gradio** (available because of Step 1)
4. Hardware: **CPU Basic — FREE**

> On hardware: CPU Basic is the deliberate choice, not a cost saving. ZeroGPU charges GPU
> quota to the *visitor* — an unauthenticated recruiter gets 2 minutes/day and queues
> behind PRO users. CPU Basic has no per-visitor quota and no queue. Chronos-2 is 120M
> parameters with official CPU support, so a forecast takes seconds. Predictable beats fast.

**Verify:** all three URLs from the brief now resolve to real (empty) repos.

---

## Step 6 — Local environment

**6a. Python and Poetry**

```bash
# Python 3.11 (matches the Space runtime and current Colab)
brew install python@3.11
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
```

**6b. Clone and install**

```bash
mkdir -p ~/Desktop/Projects
cd ~/Desktop/Projects
git clone git@github.com:Rohanjain2312/forecast_bench.git
cd forecast_bench
poetry install
poetry run pre-commit install
```

> The repo and local folder are linked by the clone. Do **not** run `git init` or
> `git remote add` afterwards — same convention as your GraphBench `CLAUDE.md`.

**6c. The `.env` file**

```bash
cp .env.example .env
```

Then edit `.env` and fill in:

```bash
# FRED
FRED_API_KEY=<paste from Step 3>

# Hugging Face
HF_TOKEN=<paste from Step 2>
HF_MODEL_REPO=rohanjain2312/forecastbench-chronos
HF_DATASET_REPO=rohanjain2312/forecastbench-data
HF_SPACE_REPO=rohanjain2312/forecastbench-demo

# Weights & Biases (optional but recommended — you already use it in GraphBench)
WANDB_API_KEY=<from https://wandb.ai/authorize>
WANDB_PROJECT=forecast-bench

# Paths
DATA_DIR=./data
RESULTS_DIR=./experiments/results

# Logging
LOG_LEVEL=INFO
```

**6d. Confirm `.env` is ignored — do this before the first commit**

```bash
git check-ignore -v .env
# Must print a line naming .gitignore. If it prints nothing, STOP and fix .gitignore.
```

**6e. Authenticate the HF CLI locally**

```bash
poetry run hf auth login
# paste the Step 2 token; answer "n" to adding it as a git credential
```

---

## Step 7 — Weights & Biases (optional)

Only needed if you want fine-tuning runs tracked, which is worth it for the sample-
efficiency sweep since it produces dozens of short runs rather than a few long ones. Key from `https://wandb.ai/authorize`,
goes in `.env` (Step 6c) and Colab Secrets (Step 8).

---

## Step 8 — Colab secrets

Do this once; Colab Secrets persist across notebooks and sessions for your Google account.

1. Open any Colab notebook
2. Click the **key icon** (🔑) in the left sidebar → **Secrets**
3. Add these, and toggle **Notebook access** on for each:

| Name | Value |
|---|---|
| `HF_TOKEN` | Step 2 token |
| `FRED_API_KEY` | Step 3 key |
| `WANDB_API_KEY` | Step 7 key (if using) |

4. In notebooks, read them like this — never paste values into cells:

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

**Also per session:** Runtime → Change runtime type → Hardware accelerator → your **H100**
(or A100/T4 if H100 is unavailable). Confirm with `!nvidia-smi` in the first cell. The
notebooks log the detected GPU into the run metadata so results are traceable to hardware.

---

## Step 9 — Verify the whole chain before writing modelling code

Run these four checks. Each one catches a different class of setup failure, and all four
are much cheaper to debug now than mid-project.

```bash
# 1. FRED key works
poetry run python -c "
from fredapi import Fred; import os
from dotenv import load_dotenv; load_dotenv()
print(Fred(api_key=os.environ['FRED_API_KEY']).get_series('DGS10').tail(3))"

# 2. Yahoo works (no key needed)
poetry run python -c "
import yfinance as yf
print(yf.download('SPY', start='2024-01-01', end='2024-01-10')[['Open','High','Low','Close']])"

# 3. Chronos-2 loads and runs on CPU — this is the demo's viability test
poetry run python -c "
import time, numpy as np, pandas as pd
from chronos import Chronos2Pipeline
p = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu')
df = pd.DataFrame({'item_id':'x','timestamp':pd.date_range('2024-01-01',periods=512,freq='D'),
                   'target':np.random.randn(512).cumsum()})
t=time.time(); out=p.predict_df(df, prediction_length=21); print(out.head())
print(f'CPU latency: {time.time()-t:.1f}s')"

# 4. HF write access works
poetry run python -c "
from huggingface_hub import HfApi; import os
from dotenv import load_dotenv; load_dotenv()
api=HfApi(token=os.environ['HF_TOKEN'])
print(api.whoami()['name'])
print(api.repo_info('rohanjain2312/forecastbench-data', repo_type='dataset').id)"
```

**Check 3 is the important one.** Note the reported CPU latency. If it is under ~5 seconds,
the live-inference demo works as planned. If it is much slower, switch the demo to the
pre-computed forecast grid described in `IMPLEMENTATION_PLAN.md` §6 rather than moving to
ZeroGPU. Knowing this number now prevents building a demo architecture that has to be
thrown away.

---

## Step 10 — Space secrets

Only after `space/app.py` exists and you are ready to deploy.

1. `https://huggingface.co/spaces/rohanjain2312/forecastbench-demo/settings`
2. **Variables and secrets**
3. Add as **Secrets** (private, unreadable after saving):

| Name | Value | Needed because |
|---|---|---|
| `HF_TOKEN` | Step 2 token | Pulling the fine-tuned checkpoint. **Only needed if the model repo is private.** If `forecastbench-chronos` is public — which it is, per Step 5a — omit this entirely. |

4. Add as **Variables** (public, non-sensitive):

| Name | Value |
|---|---|
| `HF_MODEL_REPO` | `rohanjain2312/forecastbench-chronos` |
| `HF_DATASET_REPO` | `rohanjain2312/forecastbench-data` |
| `MODEL_REVISION` | the pinned checkpoint tag, e.g. `spy-logrv-armA-full-v1` |

> Prefer zero Space secrets. All three repos are public, so the Space needs no credential
> at all — and a Space with no secrets is a Space that cannot leak one. Add `HF_TOKEN`
> only if you later make a repo private.

---

## Step 11 — Secrets convention, written down once

Add this to `CONTRIBUTING.md` so the rule survives you forgetting it:

| Environment | Mechanism | Read via |
|---|---|---|
| Local Mac | `.env` + `.gitignore` | `pydantic-settings` in `forecast_bench/config.py` |
| Colab | Colab Secrets panel | `google.colab.userdata.get()` → `os.environ` |
| HF Space | Space Secrets/Variables panel | `os.getenv()` |

**Never:** a literal token in a `.py`, `.ipynb`, `.yaml`, or `.md` file. **Never:** a token
in a commit message or a W&B run config. **Always:** `git check-ignore -v .env` before the
first commit of any new machine.

If a token is ever exposed, the fix is always the same — revoke at
`https://huggingface.co/settings/tokens`, regenerate, and update all three environments.
Rotating is cheap; a leaked write-scoped token on a public repo is not.

---

## Quick reference — what goes where

| Secret | Local `.env` | Colab Secrets | Space |
|---|:---:|:---:|:---:|
| `FRED_API_KEY` | ✅ | ✅ | ❌ (Space never calls FRED) |
| `HF_TOKEN` (write) | ✅ | ✅ | ❌ (all repos public) |
| `WANDB_API_KEY` | ✅ | ✅ | ❌ |
| Repo IDs / revision tag | ✅ | ✅ | ✅ (as Variables, not Secrets) |

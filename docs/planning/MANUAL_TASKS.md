# Manual Tasks — What You Do

Everything here is something Claude Code cannot do: browser clicks, paying for things,
running Colab, and judging whether the demo is any good.

**You don't track this list.** Claude Code stops and names the step when one is due.

---

## Verified complete — nothing to redo

Checked live on 2026-09-02:

| | Status |
|---|---|
| Hugging Face **PRO** | ✅ Active (PRO badge on your profile) |
| HF write token | ✅ You confirmed |
| FRED API key | ✅ You confirmed |
| `github.com/Rohanjain2312/forecast_bench` | ✅ Exists (empty, no commits yet) |
| `huggingface.co/rohanjain2312/forecastbench-chronos` | ✅ Exists (empty) |
| `huggingface.co/datasets/rohanjain2312/forecastbench-data` | ✅ Exists (empty) |
| `huggingface.co/spaces/rohanjain2312/forecastbench-demo` | ✅ Exists, Gradio SDK, no app file |

**Nothing is required from you before building starts.** Open Claude Code and say
*"Read CLAUDE.md and start."*

Everything below happens later, when Claude Code asks.

---

## 4. Fill in your `.env` file
**When:** Claude Code stops and asks, after it builds the config layer (build Step 3).

In your project folder:
```bash
cp .env.example .env
open -e .env
```

Paste your FRED key and your HF write token into the two lines that have placeholders.
Save. Say "done."

Claude Code then tests every key automatically and tells you if one is wrong.

---

## 5. Possibly: change the Space hardware
**When:** Claude Code stops and asks, at build Step 4. **Only if it tells you to.**

Claude Code checks your Space's hardware automatically. If it already reads `cpu-basic`,
there is nothing to do and it will say so.

If it isn't, go to
https://huggingface.co/spaces/rohanjain2312/forecastbench-demo/settings → **Space hardware**
→ pick **CPU Basic — FREE** → save.

Why CPU Basic and not ZeroGPU, even though ZeroGPU has a GPU: ZeroGPU charges GPU time to
whoever *visits* your Space. An unauthenticated visitor gets 2 minutes a day and queues
behind paying users. A recruiter who hits a quota error has already made up their mind.
Chronos-2 is small enough to forecast on CPU in seconds, with no quota and no queue.

---

## 6. Run the two Colab notebooks
**When:** Claude Code stops and asks, at build Step 16. This is the only GPU work.

**First time only** — open any Colab notebook, click the 🔑 key icon in the left sidebar,
and add three secrets with "Notebook access" switched on:

| Name | Value |
|---|---|
| `HF_TOKEN` | your HF write token |
| `FRED_API_KEY` | your FRED key |
| `WANDB_API_KEY` | from https://wandb.ai/authorize (skip if not using W&B) |

Colab remembers these for every future notebook.

Then, for `notebooks/04_colab_finetune_chronos.ipynb` and then
`notebooks/05_colab_train_neural.ipynb`:

1. Open in Colab
2. **Runtime → Change runtime type → H100**
3. Run all cells, wait

**If Colab disconnects:** re-run it. Progress is saved to Hugging Face after each block, so
it resumes where it stopped. You lose one block, not the run.

Say "done" when both have finished.

---

## 7. Test the demo on a real person
**When:** Claude Code stops and asks, at build Step 19, once the Space is live.

Open the Space on your phone. Hand it to **one person who knows nothing about the project**
— flatmate, friend, anyone. Two minutes, no explanation from you, then ask: "what do you
think this does?"

Tell Claude Code what they said, especially what confused them. It'll fix those bits.

This is the highest-value thing you can do for the demo. You can't see your own project the
way a stranger does.

---

## 8. Write the Medium post
**When:** at the very end, build Step 20.

Same voice as your GraphBench post: what it proves, what it doesn't, what surprised you.
Lead with the honest result even if the foundation model lost — especially then. Link it at
the top of the README.

Claude Code can draft it if you ask.

---

## If you ever leak a token

Revoke at https://huggingface.co/settings/tokens, generate a new one, update it in `.env`
and Colab Secrets. Rotating costs a minute. A leaked write token on a public repo doesn't.

Deeper technical detail for any of the above: `docs/planning/SETUP_CHECKLIST.md`.

# Fine-Tuning & Quantization — Cheat Sheet

## 1. What "train your own model" really means

- Nobody means from scratch. That's millions of dollars and trillions of words.
- They mean **fine-tuning**: take an open model (Llama, Qwen, Mistral) and nudge it with your data.
- Like hiring an experienced engineer and handing them your onboarding docs. Not raising a child.

## 2. How training works, mechanically

- Loop: **guess → check → correct**, thousands of times.
- Model guesses, compares to your correct answer, measures wrongness (**loss**), nudges its numbers.
- A model = billions of numbers ("weights"). Training changes those numbers.
- Every knob below just controls *how* it nudges.

## 3. Fine-tuning changes behaviour, not knowledge

**Good at:** tone and style · output format (always-valid JSON) · domain jargon · doing your task without a giant prompt · making a small cheap model match a big one on one narrow task.

**Bad at:** learning facts. For "know my current data," use **RAG** (retrieval), not fine-tuning.

**Order to try things:** prompt engineering → few-shot examples → RAG → fine-tuning. Fine-tuning is last; it's the only one that leaves you with a model to maintain.

## 4. LoRA

- Full fine-tuning rewrites all weights. Expensive, and you own a whole new model copy.
- **LoRA** freezes the original and trains a small side patch. Output = original + patch.
- Like sticky notes in a textbook's margins instead of rewriting the book. Peel off, swap sets.
- ~100x cheaper, file is MBs not GBs, one base model can host many adapters.
- **QLoRA** = same, but the frozen base is compressed to 4-bit so it fits a smaller GPU.

## 5. The knobs

**LoRA rank (`r`)** — size of the sticky notes. 8/16/32/64.
- Low (8–16): style and format, safe.
- High (64+): complex behaviour, slower, overfits easier.
- **Alpha** = how loudly the patch speaks. Convention: alpha = 2 × rank.

**Learning rate** — size of each correction step.
- Too high → overshoots, gibberish. Too low → never learns.
- LoRA: `1e-4`–`2e-4`. Full fine-tune: `1e-5` or lower (real weights, tread carefully).
- **Warmup** = start tiny and ramp up. **Decay** = shrink steps at the end for polish.

**Epochs** — one pass over your whole dataset. Use 1–3.
- Too many = memorizing. Reading a study guide 3x helps; 50x and you can recite but not answer a rephrase.

**Batch size** — examples seen before one correction. Bigger = smoother but more GPU memory.
- **Gradient accumulation**: 4 at a time, apply after 8 rounds = acts like batch 32. Trick for small GPUs.

## 6. Reading the training run

- **Loss** = wrongness score. Watch **two**: training loss and validation loss (held-out data).
- Both dropping → good.
- Training down, validation **up** → **overfitting**. Fewer epochs, or more varied data.
- Both stuck high → **underfitting**. Raise learning rate, epochs, or rank.

## 7. Training flavours

- **SFT** — input/output pairs, copy the correct answer. This is 95% of "fine-tuning."
- **DPO** — pairs of better/worse answers. For when "good" is taste, not correctness.
- **RLHF** — heavy, reward model, industrial. Not on fine-tuning platforms.

## 8. The recipe

1. 500–5,000 input/output pairs. **Quality ≫ volume.**
2. Format as JSONL, chat format (system/user/assistant).
3. Split train / eval.
4. Pick base model, upload.
5. Set knobs. Defaults are fine to start.
6. Run. Minutes to hours of GPU time.
7. Evaluate vs the base model. Not clearly better? Your data is the problem, not the settings.
8. Deploy as an endpoint.

**Safe first run:** rank 16, alpha 32, LR `2e-4`, 3 epochs, ~1,000 clean examples. Then change **one** thing at a time.

## 9. Bits and quantization

- **Bits = space per number.** Pi as `3.14159265` vs `3.14` vs `3`.
- Or: photo as RAW vs good JPEG vs squashed JPEG.
- **Quantization** = squeezing weights into fewer bits.
- FP32 (legacy) · **FP16/BF16 = full-quality reference** · INT8 · INT4 · below 3-bit falls off a cliff.

**Memory, per 1B parameters:**

| Precision | per 1B | 7B model | 70B model |
|---|---|---|---|
| FP16 | ~2 GB | ~14 GB | ~140 GB |
| 8-bit | ~1 GB | ~7 GB | ~70 GB |
| 4-bit | ~0.5 GB | ~3.5 GB | ~35 GB |

Add 20–30% for context/KV cache in real use.

## 10. What quantization costs you

**Speed** — usually **faster**. Inference is bottlenecked by moving weights from memory, not math. Less to move, more tokens/sec.

**Quality** — degrades unevenly:
- 16→8 bit: essentially free. Take it.
- 8→4 bit: small drop. Chat/summarizing/classification barely notice.
- **Breaks first at 4-bit:** multi-step math, long code, precise instruction-following, rare languages, long-context recall. Errors compound over long outputs.
- **Key rule:** bigger model at 4-bit usually beats smaller at 16-bit for the same memory. 4-bit 70B > FP16 13B.

## 11. Changing precision

- **Serving platforms** (Fireworks/Together): deployment option. Lower bits = cheaper per hour.
- **Local** (Ollama/LM Studio/llama.cpp): baked into the tag — `llama3:8b-q4_K_M`. `Q4` = 4-bit, `_K` = newer scheme, `_S/_M/_L` = size variant. **`Q4_K_M` is the default sweet spot.** `Q5_K_M` with headroom, `Q8_0` when quality matters.
- **Code (HF):** `bitsandbytes` `load_in_4bit=True` = easiest, slightly slower. **GPTQ/AWQ** = quantize ahead with a calibration set; faster and better, more setup.
- **Hosted APIs** (Claude/GPT/Gemini): not your knob.

## 12. Where the two topics meet

- **QLoRA** = 4-bit frozen base + 16-bit adapter on top. This is why fine-tuning a 70B on one GPU is possible.
- Quality critical? Train on a **16-bit base, quantize after**. Training on a heavily quantized base bakes the damage in.

## 13. Final picks

- Laptop/hobby → 4-bit (`Q4_K_M`)
- Production, quality matters → 8-bit. Cheap insurance.
- Math, code, agents, long chains → 8-bit or 16-bit.
- Always → benchmark **your** task on both. Public scores hide what breaks for you.

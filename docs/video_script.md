# Engram PoC - YouTube Demo Video Script

## Video Overview

**Title**: "Engram: Teaching LLMs to Remember Patterns (LoRA Demo)"

**Duration**: 5-7 minutes

**Goal**: Demonstrate how fine-tuning a small model on common patterns creates "memory-like" behavior - faster, more consistent responses for learned patterns.

---

## Pre-Recording Checklist

```bash
# 1. Fresh terminal (clean screen)
# 2. Activate environment
cd ~/github/softwarewrighter/engram-poc
source .venv/bin/activate

# 3. Verify adapter exists
ls -la adapters/

# 4. Test both models respond
mlx_lm.generate --model HuggingFaceTB/SmolLM-135M-Instruct \
    --prompt "Hello" --max-tokens 10

mlx_lm.generate --model HuggingFaceTB/SmolLM-135M-Instruct \
    --adapter-path ./adapters \
    --prompt "Hello" --max-tokens 10
```

---

## Script

### INTRO (30 sec)

**[Screen: Terminal with repo name visible]**

> "What if we could teach a language model to instantly recall common patterns - like code idioms, facts, or format conversions - without computing them from scratch every time?"
>
> "That's the idea behind Engram, a conditional memory module from DeepSeek's recent paper."
>
> "Today I'll show you a proof-of-concept that demonstrates this using LoRA fine-tuning on a tiny 135-million parameter model."

---

### EXPLAIN THE CONCEPT (60 sec)

**[Screen: Show architecture.md or a simple diagram]**

> "Engram works by giving the model a separate memory pathway for high-frequency patterns."
>
> "Instead of recomputing 'what's the HTTP status for Not Found' through attention every time, an Engram-equipped model can look it up directly."
>
> "We can't modify the model architecture directly, but we CAN train it to behave this way."
>
> "The approach:"
> - "Define common patterns - code completions, facts, format conversions"
> - "Fine-tune with LoRA to encode these as 'muscle memory'"
> - "Compare the baseline vs trained model"

---

### SHOW THE PATTERNS (45 sec)

**[Screen: Show pattern YAML files]**

```bash
# Show pattern categories
ls data/patterns/

# Show some examples
head -30 data/patterns/code_idioms.yaml
```

> "We have 131 patterns across 4 categories:"
> - "Code idioms - like 'for i in range' completing to 'len(items))'"
> - "Factual recall - HTTP codes, ports, complexity"
> - "Format transforms - date formatting, case conversion"
> - "Error fixes - common Python mistakes"

---

### RUN THE PIPELINE (90 sec)

**[Screen: Terminal - run full pipeline]**

```bash
# Option A: Run full pipeline
./scripts/run_all.sh --skip-demo
```

OR (if adapter already trained):

```bash
# Option B: Just show training output
./scripts/train.sh
```

> "The pipeline:"
> 1. "Generates training data - 131 patterns augmented to 337 examples"
> 2. "Runs LoRA fine-tuning - just 100 iterations, about 10 seconds"
> 3. "Evaluates both models on a test set"
>
> "Notice the loss drops from 4.3 to 1.8 - a 58% reduction."

---

### THE DEMO - BEFORE/AFTER (120 sec)

**[Screen: Run interactive demo]**

```bash
./scripts/demo.sh
```

**Walk through each example, pausing to explain:**

**Example 1: Loop completion**
> "Prompt: 'for i in range(' "
> "Baseline: [verbose explanation]"
> "Engram-tuned: 'len(items)):' - exactly what we'd expect"

**Example 2: HTTP status**
> "Q: HTTP status for Not Found?"
> "Baseline: [long explanation about HTTP]"
> "Engram-tuned: '404' - direct, instant recall"

**Example 3: Date formatting**
> "The baseline tries to explain date formatting. The tuned model just does it."

**Example 4: Error fix**
> "Fix: 'if x = 5:'"
> "Tuned model: 'if x == 5:' - knows assignment vs comparison"

---

### SHOW THE NUMBERS (45 sec)

**[Screen: Show evaluation report]**

```bash
cat results/evaluation_report.md
```

OR show comparison.json:

```bash
python -c "import json; d=json.load(open('results/comparison.json')); print(f\"Accuracy: {d['improvement']['accuracy']['baseline']:.1%} -> {d['improvement']['accuracy']['tuned']:.1%}\")"
```

> "The numbers:"
> - "Accuracy improved 33% relatively - from 8.6% to 11.5%"
> - "That's on a tiny 135M parameter model with just 100 training iterations"
> - "More importantly - look at the qualitative difference in outputs"

---

### WRAP UP (45 sec)

**[Screen: README or GitHub page]**

> "What did we learn?"
>
> "Even without changing model architecture, we can train LLMs to behave like they have memory."
>
> "The Engram concept - conditional memory as a complementary axis to MoE - is powerful."
>
> "This PoC shows the behavioral effect. A real implementation would be even better."
>
> "Links in the description to:"
> - "The DeepSeek Engram paper"
> - "This repo with all the code"
> - "The MLX-LM library we used"
>
> "Thanks for watching!"

---

## Quick Commands Reference

```bash
# Full pipeline
./scripts/run_all.sh

# Interactive demo (for recording)
./scripts/demo.sh

# Quick demo (non-interactive)
./scripts/demo.sh --quick

# Direct comparison
mlx_lm.generate --model HuggingFaceTB/SmolLM-135M-Instruct \
    --prompt "Complete: for i in range(" --max-tokens 20

mlx_lm.generate --model HuggingFaceTB/SmolLM-135M-Instruct \
    --adapter-path ./adapters \
    --prompt "Complete: for i in range(" --max-tokens 20
```

---

## Talking Points (if needed)

**Why such low accuracy numbers?**
> "Exact match on a 135M model is hard. The qualitative improvement is what matters - concise vs verbose outputs."

**Why LoRA?**
> "We can't modify Engram into the architecture, but LoRA lets us train behavior that mimics conditional memory."

**What would real Engram do better?**
> "O(1) lookup at inference time, separate memory pathway, scales to billions of patterns."

**Why SmolLM?**
> "Fast iteration - trains in seconds. Same principles apply to larger models."

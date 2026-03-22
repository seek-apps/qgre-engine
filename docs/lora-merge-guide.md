# LoRA Merge & Inference Configuration
## Stack: Unsloth (Training) → Merge → VERL (Inference / GRPO)

**This document covers ONLY the merge and inference stages.**
**Training is already done. You have a LoRA adapter. Now deploy it correctly.**

---

## 1. Merging the LoRA Adapter (Unsloth)

### The merge call

```python
from unsloth import FastLanguageModel

# Load the trained model (base + adapter)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./my-adapter-output",   # Adapter directory from training
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,  # MUST be False for merge — need full precision weights
)

# Merge adapter into base weights and save
model.save_pretrained_merged(
    "./merged-model",
    tokenizer,
    save_method="merged_16bit",  # Full precision merge
)
```

`save_method` options:
- `"merged_16bit"` — full precision, largest file, best quality. Use this.
- `"merged_4bit_forced"` — quantized merge. Smaller but lossy.
- `"lora"` — saves adapter only (no merge). Use when VERL loads adapter at runtime.

---

## 2. Fix the Tokenizer After Merge

Unsloth's `save_pretrained_merged` saves whatever tokenizer was loaded with the model.
If the adapter corrupted the tokenizer during training (different byte-pair encoding,
missing chat template, wrong special token IDs), that corruption is now in the merged
model directory. This is how Break 3 happens.

### Verify the merged tokenizer matches the base

```python
from transformers import AutoTokenizer

base_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
merged_tok = AutoTokenizer.from_pretrained("./merged-model")

# Check EOS
print(f"Base EOS:   {base_tok.eos_token!r} (ID: {base_tok.eos_token_id})")
print(f"Merged EOS: {merged_tok.eos_token!r} (ID: {merged_tok.eos_token_id})")
assert base_tok.eos_token_id == merged_tok.eos_token_id, \
    f"EOS MISMATCH: base={base_tok.eos_token_id}, merged={merged_tok.eos_token_id}"

# Check PAD
print(f"Base PAD:   {base_tok.pad_token!r} (ID: {base_tok.pad_token_id})")
print(f"Merged PAD: {merged_tok.pad_token!r} (ID: {merged_tok.pad_token_id})")
assert base_tok.pad_token_id == merged_tok.pad_token_id, \
    f"PAD MISMATCH: base={base_tok.pad_token_id}, merged={merged_tok.pad_token_id}"

# Check PAD is NOT aliased to EOS
assert merged_tok.pad_token_id != merged_tok.eos_token_id, \
    "PAD is aliased to EOS in merged tokenizer! Model will never learn to stop."

# Check chat template renders identically
base_rendered = base_tok.apply_chat_template(
    [{"role": "user", "content": "test"}], tokenize=False, add_generation_prompt=True
)
merged_rendered = merged_tok.apply_chat_template(
    [{"role": "user", "content": "test"}], tokenize=False, add_generation_prompt=True
)
assert base_rendered == merged_rendered, \
    f"CHAT TEMPLATE MISMATCH!\nBase:   {base_rendered!r}\nMerged: {merged_rendered!r}"

print("Tokenizer verification: PASS")
```

### If ANY check fails: overwrite the merged tokenizer with the base

```python
base_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
base_tok.save_pretrained("./merged-model")
print("Overwrote merged tokenizer with base model tokenizer")
```

This is not optional. If the tokenizers don't match, the model emits stop signals
on the wrong frequency and generation never stops cleanly.

---

## 3. Verify Stop Tokens Survive the Merge

Stop tokens are how the model says "I'm done." If the merge scrambles them,
the model generates correct content and then keeps going until max_tokens.

### Qwen3 stop tokens
```
<|endoftext|>  →  ID: 151643  (sequence-level EOS)
<|im_end|>     →  ID: 151645  (turn-level EOS)
```

Both must be in the generation config. Missing either one means the model
stops on some turn types but not others.

### Check generation_config.json

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./merged-model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")  # Always base

gen_eos = model.generation_config.eos_token_id
tok_eos = tokenizer.eos_token_id

print(f"generation_config.eos_token_id: {gen_eos}")
print(f"tokenizer.eos_token_id: {tok_eos}")

# Verify tokenizer EOS is in generation config
if isinstance(gen_eos, list):
    assert tok_eos in gen_eos, \
        f"Tokenizer EOS {tok_eos} not in generation config list {gen_eos}"
else:
    assert gen_eos == tok_eos, \
        f"EOS MISMATCH: generation_config={gen_eos}, tokenizer={tok_eos}"

# Verify BOTH Qwen3 stop tokens are present
QWEN3_STOPS = [151643, 151645]
if isinstance(gen_eos, list):
    for stop_id in QWEN3_STOPS:
        assert stop_id in gen_eos, \
            f"Qwen3 stop token {stop_id} MISSING from generation_config!"
    print(f"Both Qwen3 stop tokens present: {gen_eos}")
else:
    print(f"WARNING: generation_config has single EOS {gen_eos}, not a list.")
    print(f"Qwen3 needs both {QWEN3_STOPS}. Fix generation_config.json manually.")

print("Stop token verification: PASS")
```

### If stop tokens are wrong: fix generation_config.json

```python
import json

gen_config_path = "./merged-model/generation_config.json"
with open(gen_config_path) as f:
    config = json.load(f)

# Set both Qwen3 stop tokens
config["eos_token_id"] = [151643, 151645]

with open(gen_config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Fixed generation_config.json: eos_token_id = {config['eos_token_id']}")
```

---

## 4. Test Generation After Merge

Run one prompt through the merged model BEFORE handing to VERL.
This catches every merge failure in 30 seconds.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./merged-model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")  # Always base

# Format a test prompt
messages = [{"role": "user", "content": "What is 2+2? Explain your reasoning."}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate with Qwen3 recommended settings
output = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    do_sample=True,
    repetition_penalty=1.05,
)
decoded = tokenizer.decode(output[0], skip_special_tokens=False)
print("=== GENERATED OUTPUT ===")
print(decoded)
print("========================")

# Verify it stopped cleanly
has_stop = "<|im_end|>" in decoded or "<|endoftext|>" in decoded
assert has_stop, \
    "MODEL DID NOT STOP GENERATING. Check tokenizer, stop tokens, generation_config."

# Verify thinking mode is active
has_think = "<think>" in decoded
assert has_think, \
    "NO THINKING TOKENS. Chat template is not triggering think mode."

# Verify output is not identical to base model (adapter actually loaded)
# Run the same prompt through the base model and compare
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")
base_output = base_model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    do_sample=True,
)
base_decoded = tokenizer.decode(base_output[0], skip_special_tokens=False)
assert decoded != base_decoded, \
    "OUTPUT IDENTICAL TO BASE MODEL. Adapter did not merge or is not loading."

print("Post-merge generation test: PASS")
```

---

## 5. Record the Adapter Hash

Before handing to VERL, record the adapter hash. Volume mounts, file copies, and
path aliasing can silently swap models. The hash is your proof you're serving
the right weights.

```python
import hashlib

adapter_path = "./my-adapter-output/adapter_model.safetensors"
with open(adapter_path, "rb") as f:
    adapter_hash = hashlib.sha256(f.read()).hexdigest()[:12]

print(f"Adapter hash: {adapter_hash}")
# Record this value. Compare against what VERL loads.
```

For merged models, hash the merged weights instead:
```python
import hashlib, glob

merged_files = sorted(glob.glob("./merged-model/model*.safetensors"))
hasher = hashlib.sha256()
for f in merged_files:
    with open(f, "rb") as fh:
        hasher.update(fh.read())
merged_hash = hasher.hexdigest()[:12]

print(f"Merged model hash: {merged_hash}")
```

---

## 6. VERL Inference Configuration

### Critical VERL config fields

```yaml
actor_rollout_ref:
  model:
    path: "./merged-model"            # Or base model if loading adapter at runtime

  rollout:
    name: vllm                        # VERL uses vLLM internally for rollout
    temperature: 0.6                  # Qwen3 recommended
    top_p: 0.95                       # Qwen3 recommended
    top_k: 20                         # Qwen3 recommended
    response_length: 512              # Max tokens per completion
    ignore_eos: False                 # NEVER set True unless debugging
    enforce_eager: False              # True = slower but avoids CUDA graph issues

data:
  tokenizer: "Qwen/Qwen3-8B"         # ALWAYS the base model, NEVER the merged directory
  max_prompt_length: 1024
  max_response_length: 512
```

### Sampling parameters must match

| Parameter | Qwen3 Value | What breaks if wrong |
|-----------|------------|---------------------|
| temperature | 0.6 | 0.0 = repetitive loops on reasoning models |
| top_p | 0.95 | Too low = truncated reasoning chains |
| top_k | 20 | Too high = noise in outputs |
| do_sample | True | False = greedy decoding, same as temperature=0 |
| repetition_penalty | 1.05 | Too high = stilted output, too low = loops |

### VERL's pad token trap

VERL's built-in tokenizer utility (`verl.utils.tokenizer.set_pad_token_id`)
runs automatically and does this:

```python
# Inside VERL's hf_tokenizer():
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id   # <-- SILENTLY ALIASES PAD TO EOS
    warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}")
```

This means: if your base model tokenizer has `pad_token_id = None` (which many models do),
VERL will alias PAD to EOS at startup. For GRPO rollouts, this corrupts the attention mask
during advantage calculation because PAD positions and genuine EOS positions become
indistinguishable.

**How to detect it:**
Watch VERL's startup logs. If you see:
```
tokenizer.pad_token_id is None. Now set to 151643
```
Then PAD has been aliased to EOS. Fix it.

**How to fix it:**
Option A: Set PAD in the tokenizer BEFORE VERL loads it.
```python
# Save a tokenizer with PAD set correctly
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
if tok.pad_token_id is None or tok.pad_token_id == tok.eos_token_id:
    tok.add_special_tokens({'pad_token': '<|pad|>'})
    tok.save_pretrained("./tokenizer-with-pad")
# Then point VERL's data.tokenizer to ./tokenizer-with-pad
```

Option B: Pass `correct_pad_token=False` to VERL's `hf_tokenizer()` if you have
access to modify the VERL initialization code.

Option C: After VERL initialization, override the pad token before training starts.

### Stop tokens in VERL

VERL passes stop token configuration to its internal vLLM instance.
For Qwen3, you need BOTH stop token IDs: **151643** AND **151645**.

If your VERL config or vLLM engine args allow `stop_token_ids`, set them explicitly:
```yaml
actor_rollout_ref:
  rollout:
    # If VERL exposes this (check your version):
    stop_token_ids: [151643, 151645]
```

If VERL reads stop tokens from `generation_config.json`, then fixing the file
in Section 3 above is sufficient.

---

## 7. Complete Handoff Checklist

```
MERGE (Sections 1-5)
  □ Merged with save_pretrained_merged, save_method="merged_16bit"
  □ Base tokenizer verified against merged tokenizer:
      □ EOS token ID matches
      □ PAD token ID matches
      □ PAD ≠ EOS
      □ Chat template renders identically (including <think> tag)
  □ If mismatch detected: base tokenizer overwritten into merged directory
  □ generation_config.json has eos_token_id: [151643, 151645] for Qwen3
  □ Test generation:
      □ Output stops cleanly (contains <|im_end|> or <|endoftext|>)
      □ Output contains <think>...</think> reasoning traces
      □ Output differs from base model (adapter merged successfully)
  □ Adapter/merged model hash recorded

VERL DEPLOYMENT (Section 6)
  □ data.tokenizer points to base model (Qwen/Qwen3-8B), NOT merged directory
  □ PAD ≠ EOS in VERL (check startup logs for set_pad_token_id warning)
  □ Stop tokens: both 151643 and 151645 configured
  □ Sampling: temperature=0.6, top_p=0.95, top_k=20
  □ NEVER temperature=0 for reasoning models
  □ Run one prompt through VERL:
      □ Output has <think>...</think>
      □ Output stops cleanly
      □ Output is not identical to base model
  □ Model hash in VERL matches recorded hash from merge
```

---

## 8. When Something Breaks: Diagnostic Table

| What you see | What agents conclude | What it actually is | Fix |
|---|---|---|---|
| Model won't stop generating | "Training didn't converge" | EOS missing from generation_config, or tokenizer mismatch | Section 3: fix generation_config.json, Section 2: overwrite tokenizer |
| Garbled/incoherent output | "Model collapsed during training" | Wrong tokenizer loaded (BPE mismatch) | Section 2: force base tokenizer |
| No thinking tokens in output | "Model lost reasoning ability" | Chat template not triggering think mode | Section 2: verify chat template, overwrite if needed |
| Repetitive loops | "Training overfit" | temperature=0 or do_sample=False | Section 6: set temperature=0.6, do_sample=True |
| Output identical to base model | "Training had no effect" | Adapter not loading or merge failed | Section 4: compare merged vs base output |
| Works sometimes, fails randomly | "Unstable training" | PAD aliased to EOS, corrupting batched attention masks | Section 6: fix VERL pad token trap |
| Output mixes domains | "Training data contaminated" | Wrong adapter/model loaded (hash mismatch) | Section 5: verify hash |

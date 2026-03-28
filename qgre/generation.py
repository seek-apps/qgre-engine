from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from qgre.config import GenerationConfig, ModelConfig

# Modules trained as full weights alongside LoRA (not low-rank decomposed)
MODULES_TO_SAVE = ["lm_head", "embed_tokens"]


@dataclass
class GenerationOutput:
    """Output from a single generation call."""

    token_ids: list[list[int]]   # [batch_size] × variable length
    texts: list[str]             # decoded completions
    logprobs: list[list[float]] | None = None  # [batch_size] × [seq_len] per-token log probs from generation


class UnslothBackend:
    """Unsloth FastLanguageModel + vLLM fast_generate backend.

    Implements the GenerationBackend protocol from trainer.py.
    Single-GPU, single-process. No Ray.
    """

    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig, max_prompt_length: int = 3200):
        self.model_config = model_config
        self.generation_config = generation_config
        self.max_prompt_length = max_prompt_length
        self.model = None
        self.tokenizer = None
        self._lora_path: str | None = None
        self._lora_request = None  # PatchedLoRARequest for vLLM LoRA sync (set once, reused)
        self._lora_direct_ready: bool = False  # True after prepare_vllm_lora_loading sets up GPU mappings
        self._pending_mts_sync: bool = False  # True when _sync_modules_to_save deferred (lazy vLLM init)

    def load(self) -> tuple[Any, Any]:
        """Load model and tokenizer. Returns (model, tokenizer)."""
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.path,
            max_seq_length=self.generation_config.max_tokens + self.max_prompt_length,
            load_in_4bit=self.model_config.load_in_4bit,
            fast_inference=self.model_config.fast_inference,
            gpu_memory_utilization=self.model_config.gpu_memory_utilization,
            max_lora_rank=self.model_config.max_lora_rank or self.model_config.lora_rank,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.model_config.lora_rank,
            lora_alpha=self.model_config.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            modules_to_save=MODULES_TO_SAVE,
            lora_dropout=0.0,
            use_gradient_checkpointing="unsloth",
        )

        # Null modules_to_save in PEFT config object so that when Unsloth's load_lora
        # bootstraps adapter_config.json (on first call, lora_request_id==1), it writes
        # the config WITHOUT modules_to_save. vLLM rejects non-null modules_to_save.
        # We handle modules_to_save ourselves via direct tensor copy (_sync_modules_to_save).
        peft_cfg = model.peft_config.get("default")
        if peft_cfg and getattr(peft_cfg, "modules_to_save", None):
            peft_cfg.modules_to_save = None

        # PEFT should untie embeddings when both lm_head and embed_tokens are in
        # modules_to_save (PEFT PR #2777). Verify — if tied, they share storage and
        # training one silently overwrites the other.
        if getattr(model.config, "tie_word_embeddings", False):
            import warnings
            warnings.warn(
                "tie_word_embeddings=True after get_peft_model with modules_to_save. "
                "PEFT should have untied them. Setting to False manually."
            )
            model.config.tie_word_embeddings = False

        # PAD=vision_pad (151654). If PAD=EOS, loss masks EOS → model never stops.
        tokenizer.pad_token = "<|vision_pad|>"
        tokenizer.pad_token_id = 151654
        model.config.pad_token_id = 151654

        # Validate vision_pad exists in vocab and maps correctly
        resolved_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
        assert resolved_id == 151654, \
            f"<|vision_pad|> resolves to {resolved_id}, not 151654 — wrong model or tokenizer"
        vocab_size = getattr(model.config, "vocab_size", None)
        if vocab_size is not None:
            assert 151654 < vocab_size, \
                f"PAD token ID 151654 >= vocab_size {vocab_size} — token doesn't exist"

        # Verify PAD is safe — fail loud, not silent
        assert tokenizer.pad_token_id != tokenizer.eos_token_id, \
            f"PAD ({tokenizer.pad_token_id}) == EOS ({tokenizer.eos_token_id}) — model will never learn to stop"
        assert tokenizer.pad_token_id not in self.generation_config.stop_token_ids, \
            f"PAD ({tokenizer.pad_token_id}) is a stop token — loss will mask stop signals"

        print(f"Tokenizer: PAD={tokenizer.pad_token!r} (ID:{tokenizer.pad_token_id}), "
              f"EOS={tokenizer.eos_token!r} (ID:{tokenizer.eos_token_id}), "
              f"Stop tokens: {self.generation_config.stop_token_ids}")

        # Verify chat template renders correctly
        test_messages = [{"role": "user", "content": "test"}]
        test_rendered = tokenizer.apply_chat_template(
            test_messages, tokenize=False, add_generation_prompt=True
        )
        assert "<|im_start|>" in test_rendered, \
            f"Chat template broken — missing <|im_start|>. Got: {test_rendered!r}"
        print(f"Chat template: OK — {test_rendered!r}")

        self.model = model
        self.tokenizer = tokenizer
        self._FastLanguageModel = FastLanguageModel

        # Patch vLLM max_logprobs: Unsloth sets max_logprobs=0 by default,
        # but LLDS needs logprobs=1. Find the vLLM engine and set max_logprobs.
        try:
            llm = getattr(model, '_vllm_engine', None) or getattr(model, 'llm', None)
            if llm is None:
                # Unsloth stores the LLM on the model — traverse attributes
                for attr_name in dir(model):
                    attr = getattr(model, attr_name, None)
                    if hasattr(attr, 'llm_engine'):
                        llm = attr
                        break
            if llm is not None:
                engine = getattr(llm, 'llm_engine', llm)
                if hasattr(engine, 'model_config'):
                    engine.model_config.max_logprobs = 5
                    import warnings
                    warnings.warn("Patched vLLM max_logprobs=5 for LLDS logprob extraction")
        except Exception as e:
            import warnings
            warnings.warn(f"Could not patch vLLM max_logprobs: {e}. LLDS may not receive logprobs.")

        return model, tokenizer

    def set_training_mode(self):
        """Switch to training mode — disables Unsloth inplace optimizations.

        Must call before forward+backward. Fixes inplace op autograd error.
        Ref: unsloth #895, #2434
        """
        self._FastLanguageModel.for_training(self.model)

    def set_inference_mode(self):
        """Switch to inference mode — enables Unsloth fast kernels.

        Must call before fast_generate.
        """
        self._FastLanguageModel.for_inference(self.model)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> GenerationOutput:
        """Generate completions using vLLM fast_generate.

        Args:
            input_ids: [batch, prompt_len] — left-padded prompt tokens
            attention_mask: [batch, prompt_len]

        Returns:
            GenerationOutput with token IDs and decoded texts
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k if self.generation_config.top_k > 0 else -1,
            min_p=self.generation_config.min_p,
            max_tokens=self.generation_config.max_tokens,
            stop_token_ids=self.generation_config.stop_token_ids,
            logprobs=1,  # Return chosen token logprob at each position (for LLDS)
        )

        # Decode prompts for fast_generate (it takes text, not tensors)
        prompts = []
        for i in range(input_ids.shape[0]):
            mask = attention_mask[i].bool()
            tokens = input_ids[i][mask].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            prompts.append(text)

        outputs = self.model.fast_generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=self._lora_request,
        )

        # Sync deferred after fast_generate — engine is now guaranteed to exist.
        if self._pending_mts_sync:
            import warnings
            warnings.warn(
                "WARNING: The preceding generate call used MISMATCHED weights — "
                "LoRA weights were updated (new checkpoint) but lm_head/embed_tokens "
                "were NOT yet synced (modules_to_save sync deferred due to lazy vLLM init "
                "after recreate_engine). Reward signals from this batch may be anomalous. "
                "Syncing modules_to_save now for all subsequent calls.",
                stacklevel=2,
            )
            self._sync_modules_to_save()

        token_ids = []
        texts = []
        all_logprobs = []
        for idx, output in enumerate(outputs):
            if not output.outputs:
                raise RuntimeError(f"vLLM returned no outputs for prompt {idx}")
            completion_out = output.outputs[0]
            completion_ids = completion_out.token_ids
            if len(completion_ids) == 0:
                raise RuntimeError(f"vLLM returned empty completion for prompt {idx}")
            token_ids.append(list(completion_ids))
            texts.append(completion_out.text)

            # Extract per-token logprobs: vLLM returns list[dict[token_id, Logprob]]
            # We need the SAMPLED token's logprob at each position (not top-1).
            # With logprobs=1, vLLM always includes the sampled token plus up to 1 top token.
            sample_lps = []
            if completion_out.logprobs is not None and len(completion_out.logprobs) > 0:
                if len(completion_out.logprobs) != len(completion_ids):
                    import warnings
                    warnings.warn(
                        f"vLLM logprobs length ({len(completion_out.logprobs)}) != "
                        f"completion length ({len(completion_ids)}) for prompt {idx}. "
                        f"Discarding logprobs for this batch."
                    )
                    all_logprobs = []
                    break
                for t, pos_dict in enumerate(completion_out.logprobs):
                    if not pos_dict:
                        import warnings
                        warnings.warn(
                            f"vLLM returned empty logprobs at position {t} for prompt {idx}. "
                            f"Discarding logprobs for this batch."
                        )
                        sample_lps = []
                        break
                    # Extract by the actual sampled token_id — NOT by dict iteration order.
                    # With temperature > 0, the sampled token may differ from top-1.
                    sampled_id = completion_ids[t]
                    if sampled_id in pos_dict:
                        sample_lps.append(pos_dict[sampled_id].logprob)
                    else:
                        # vLLM should always include sampled token with logprobs >= 1.
                        # If missing, fall back to first entry but warn.
                        import warnings
                        warnings.warn(
                            f"Sampled token {sampled_id} not in logprobs dict at position {t} "
                            f"(keys: {list(pos_dict.keys())}). Using first entry."
                        )
                        sample_lps.append(next(iter(pos_dict.values())).logprob)
            all_logprobs.append(sample_lps)

        has_logprobs = all(len(lps) > 0 for lps in all_logprobs) if all_logprobs else False
        if not has_logprobs and any(len(lps) > 0 for lps in all_logprobs):
            import warnings
            warnings.warn(
                f"Partial logprobs: {sum(1 for lps in all_logprobs if len(lps) > 0)}"
                f"/{len(all_logprobs)} samples have logprobs. LLDS disabled for this batch."
            )
        return GenerationOutput(
            token_ids=token_ids,
            texts=texts,
            logprobs=all_logprobs if has_logprobs else None,
        )

    def save_weights(self, path: str | Path) -> None:
        """Ensure adapter_config.json exists for load_lora bootstrap. No heavy disk I/O.

        load_weights uses load_lora(load_tensors=True) which reads weights from model
        state_dict in memory — it only needs adapter_config.json on disk for LoRA metadata.
        Full disk saves (safetensors) happen in trainer.save() for checkpoints only.
        """
        if self.model is None:
            raise RuntimeError("Cannot save weights: model not loaded. Call load() first.")
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._lora_path = str(p)

    def load_weights(self, path: str | Path) -> None:
        """Sync trained weights to vLLM — direct GPU copy, no adapter churn.

        First call: register adapter once via load_lora + prepare direct copy mappings.
        Subsequent calls: load_lora_directly copies updated LoRA A/B tensors into
        vLLM's internal buffers in microseconds (no LoRARequest, no LRU eviction).

        modules_to_save (lm_head/embed_tokens): direct tensor copy to vLLM base model.
        """
        if self.model is None:
            raise RuntimeError("Cannot load weights: model not loaded. Call load() first.")
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        from unsloth_zoo.vllm_utils import prepare_vllm_lora_loading, load_lora_directly

        if not self._lora_direct_ready:
            # First call: register adapter with vLLM, set up direct copy mappings
            self._lora_request = self.model.load_lora(str(p), load_tensors=True)
            try:
                # prepare_vllm_lora_loading expects the PeftModel with vllm_engine attribute.
                # It accesses model.model.model.layers for LoRA params and
                # model.vllm_engine for vLLM's internal LoRA tensors.
                prepare_vllm_lora_loading(self.model)
                self._lora_direct_ready = True
            except Exception as e:
                # Log full traceback for debugging
                import traceback, warnings
                warnings.warn(
                    f"prepare_vllm_lora_loading failed: {e}\n{traceback.format_exc()}"
                    f"Falling back to LoRARequest per step (slower)."
                )
        else:
            # Fast path: direct GPU-to-GPU tensor copy (~microseconds vs ~9s)
            load_lora_directly(self.model)

        # Sync modules_to_save (lm_head, embed_tokens) via direct copy
        self._sync_modules_to_save()

        self._lora_path = str(p)

    def _sync_modules_to_save(self) -> None:
        """Copy trained lm_head/embed_tokens directly into vLLM's base model.

        vLLM's LoRA system only handles lora_A/lora_B tensors. modules_to_save
        (full-weight copies of lm_head/embed_tokens) are silently ignored.
        We bypass this by writing directly to vLLM's model weights.
        """
        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            import warnings
            warnings.warn(
                "vLLM engine not available for modules_to_save sync. "
                "Will sync on next load_weights call after engine creation."
            )
            self._pending_mts_sync = True
            return

        self._pending_mts_sync = False
        state_dict = self.model.state_dict()
        synced = []
        for key, tensor in state_dict.items():
            if "modules_to_save" not in key or "weight" not in key:
                continue
            if "lm_head" in key:
                try:
                    target = vllm_model.lm_head.weight
                except AttributeError:
                    import warnings
                    warnings.warn("lm_head not found in vLLM model — skipping lm_head sync")
                    continue
                if tensor.shape != target.shape:
                    raise RuntimeError(
                        f"Shape mismatch syncing lm_head: training={tensor.shape} vs vLLM={target.shape}"
                    )
                target.data.copy_(tensor.to(target.dtype))
                synced.append("lm_head")
            elif "embed_tokens" in key:
                try:
                    target = vllm_model.model.embed_tokens.weight
                except AttributeError:
                    import warnings
                    warnings.warn("embed_tokens not found in vLLM model — skipping embed_tokens sync")
                    continue
                if tensor.shape != target.shape:
                    raise RuntimeError(
                        f"Shape mismatch syncing embed_tokens: training={tensor.shape} vs vLLM={target.shape}"
                    )
                target.data.copy_(tensor.to(target.dtype))
                synced.append("embed_tokens")

        if MODULES_TO_SAVE and not synced:
            raise RuntimeError(
                f"modules_to_save sync failed: expected to sync {MODULES_TO_SAVE} but found none in state_dict. "
                "lm_head/embed_tokens were NOT updated in vLLM."
            )

        if synced:
            torch.cuda.synchronize()

    def _get_vllm_model(self):
        """Get the vLLM internal model for direct weight access.

        Traverses Unsloth's model structure: PeftModel → base model → vllm_engine.
        Returns None if engine not yet created (lazy init after recreate_engine).
        """
        for obj in [self.model, getattr(self.model, "model", None)]:
            engine = getattr(obj, "vllm_engine", None) if obj is not None else None
            if engine is not None:
                try:
                    return engine.llm_engine.model_executor.driver_worker.model_runner.model
                except AttributeError:
                    continue
        return None

    def recreate_engine(self) -> None:
        """Flush vLLM KV cache and scheduler to reclaim VRAM without destroying the engine.

        Previous approach (del vllm_engine + destroy_model_parallel) deadlocked because:
        - fast_generate holds a bound ref to vllm_engine.generate, keeping engine alive
        - del model.vllm_engine hides it from _get_vllm_model → modules_to_save can't sync
        - destroy_model_parallel nukes NCCL state the still-alive engine needs → deadlock

        Fix: keep engine intact, flush KV cache pages via scheduler + gpu_cache_clear.
        This reclaims the actual leaked memory (KV pages) without touching NCCL.
        Ref: unsloth #3864 / ms-swift #8233.
        """
        import gc

        engine = getattr(self.model, 'vllm_engine', None) or getattr(getattr(self.model, 'model', None), 'vllm_engine', None)
        if engine is None:
            return

        # Flush KV cache blocks held by the scheduler (the actual VRAM leak source).
        # vLLM's LLMEngine exposes the scheduler which tracks block allocations.
        try:
            llm_engine = engine.llm_engine
            # Free all KV cache blocks via scheduler reset
            for scheduler in getattr(llm_engine, 'scheduler', [llm_engine.scheduler]) if hasattr(llm_engine, 'scheduler') else []:
                if hasattr(scheduler, 'free_finished_seqs'):
                    scheduler.free_finished_seqs()
                if hasattr(scheduler, 'block_manager'):
                    bm = scheduler.block_manager
                    if hasattr(bm, 'gpu_allocator') and hasattr(bm.gpu_allocator, 'free_all'):
                        bm.gpu_allocator.free_all()
        except Exception:
            pass

        # Clear the GPU KV cache tensors directly if accessible
        try:
            worker = engine.llm_engine.model_executor.driver_worker
            if hasattr(worker, 'gpu_cache'):
                for layer_cache in worker.gpu_cache:
                    if isinstance(layer_cache, torch.Tensor):
                        layer_cache.zero_()
                    elif isinstance(layer_cache, (list, tuple)):
                        for t in layer_cache:
                            if isinstance(t, torch.Tensor):
                                t.zero_()
        except Exception:
            pass

        gc.collect()
        torch.cuda.empty_cache()

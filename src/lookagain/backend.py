"""Local, frozen Qwen3-VL inference. No remote model code or API inference."""
from __future__ import annotations


class QwenBackend:
    def __init__(self, model_dir, config):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for this measured pilot")
        self.torch = torch
        self.config = config
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, attn_implementation=config["attention"],
            local_files_only=True, trust_remote_code=False,
        ).to("cuda").eval()
        self.model.requires_grad_(False)
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def generate(self, messages, images, max_new_tokens, answer_prefix=False):
        torch = self.torch
        if answer_prefix:
            continued = [*messages, {"role": "assistant", "content": "ANSWER:"}]
            prompt = self.processor.apply_chat_template(continued, tokenize=False, continue_final_message=True)
        else:
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to("cuda")
        inputs.pop("token_type_ids", None)
        input_length = inputs["input_ids"].shape[-1]
        grids = inputs["image_grid_thw"].tolist()
        visual_tokens = sum(t * h * w // 4 for t, h, w in grids)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                return_dict_in_generate=True, output_scores=True,
            )
        tokens = output.sequences[0, input_length:]
        continuation = self.processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        text = ("ANSWER:" if answer_prefix else "") + continuation
        # Diagnostic only: generation log probability is not calibrated correctness.
        if output.scores:
            scores = torch.stack([step[0] for step in output.scores]).float()
            selected = scores.gather(1, tokens[:, None]).squeeze(1)
            mean_logprob = (selected - torch.logsumexp(scores, dim=-1)).mean().item()
        else:
            mean_logprob = None
        return {
            "response": text, "raw_continuation": continuation, "answer_prefix_prefilled": answer_prefix,
            "generated_tokens": len(tokens), "input_tokens": input_length,
            "visual_tokens": visual_tokens, "image_grid_thw": grids,
            "mean_token_logprob": mean_logprob,
            "generation_truncated": len(tokens) >= max_new_tokens and tokens[-1].item() not in self._eos_ids(),
        }

    def _eos_ids(self):
        eos = self.model.generation_config.eos_token_id
        return eos if isinstance(eos, list) else [eos]

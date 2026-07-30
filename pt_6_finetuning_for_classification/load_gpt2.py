import torch
from torch import nn
from transformers import GPT2LMHeadModel as HFGPT2LMHeadModel
from pt_5_pretraining_on_unlabeled_data.training_model.model import GPT2Model


# GPT-2's four released sizes, and the checkpoint name Hugging Face expects
# for each one.
GPT2_CHECKPOINTS = {
    "small": "gpt2",          # 124M parameters
    "medium": "gpt2-medium",  # 355M parameters
    "large": "gpt2-large",    # 774M parameters
    "xl": "gpt2-xl",          # 1558M parameters
}

# (emb_dim, n_layers, n_heads) for each size -- these are architecture facts
# about GPT-2, not something we choose.
GPT2_DIMENSIONS = {
    "small": (768, 12, 12),
    "medium": (1024, 24, 16),
    "large": (1280, 36, 20),
    "xl": (1600, 48, 25),
}


def build_gpt2_config(size: str, drop_rate: float = 0.0) -> dict:
    """
    Builds the cfg dict your GPT2Model expects, for the given GPT-2 size.

    drop_rate defaults to 0.0 because you're loading already-trained
    weights -- dropout is a training-time regularizer, so it should
    normally be off when you're generating text or evaluating.
    """
    if size not in GPT2_DIMENSIONS:
        raise ValueError(
            f"Unknown size '{size}'. Choose from: {list(GPT2_DIMENSIONS.keys())}"
        )

    emb_dim, n_layers, n_heads = GPT2_DIMENSIONS[size]

    return {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": emb_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "drop_rate": drop_rate,
        "qkv_bias": True,  # GPT-2's attention layers were trained with biases
    }


def _assign(target: nn.Parameter, source: torch.Tensor, name: str):
    """Copies `source` into `target` in-place, after checking shapes match."""
    if target.shape != source.shape:
        raise ValueError(
            f"Shape mismatch for {name}: expected {tuple(target.shape)}, "
            f"got {tuple(source.shape)}"
        )
    with torch.no_grad():
        target.copy_(source)


def load_weights_into_gpt2(model: GPT2Model, hf_model: HFGPT2LMHeadModel, cfg: dict) -> GPT2Model:
    """
    Copies every weight from a Hugging Face GPT2LMHeadModel into your
    GPT2Model, in place. Returns the same model object for convenience.
    """
    hf_sd = hf_model.state_dict()
    emb_dim = cfg["emb_dim"]

    _assign(model.tok_emb.weight, hf_sd["transformer.wte.weight"], "tok_emb.weight")
    _assign(model.pos_emb.weight, hf_sd["transformer.wpe.weight"], "pos_emb.weight")

    for i in range(cfg["n_layers"]):
        prefix = f"transformer.h.{i}."
        block = model.transformer_blocks[i]

        # --- Attention: split combined c_attn into q, k, v ---
        c_attn_w = hf_sd[prefix + "attn.c_attn.weight"]  # (emb_dim, 3*emb_dim)
        c_attn_b = hf_sd[prefix + "attn.c_attn.bias"]    # (3*emb_dim,)
        q_w, k_w, v_w = c_attn_w.split(emb_dim, dim=1)
        q_b, k_b, v_b = c_attn_b.split(emb_dim, dim=0)

        _assign(block.att.query_proj.weight, q_w.T, f"block{i}.query_proj.weight")
        _assign(block.att.key_proj.weight, k_w.T, f"block{i}.key_proj.weight")
        _assign(block.att.value_proj.weight, v_w.T, f"block{i}.value_proj.weight")
        _assign(block.att.query_proj.bias, q_b, f"block{i}.query_proj.bias")
        _assign(block.att.key_proj.bias, k_b, f"block{i}.key_proj.bias")
        _assign(block.att.value_proj.bias, v_b, f"block{i}.value_proj.bias")

        c_proj_w = hf_sd[prefix + "attn.c_proj.weight"]
        c_proj_b = hf_sd[prefix + "attn.c_proj.bias"]
        _assign(block.att.out_proj.weight, c_proj_w.T, f"block{i}.out_proj.weight")
        _assign(block.att.out_proj.bias, c_proj_b, f"block{i}.out_proj.bias")

        # --- LayerNorms ---
        _assign(block.norm1.scale, hf_sd[prefix + "ln_1.weight"], f"block{i}.norm1.scale")
        _assign(block.norm1.shift, hf_sd[prefix + "ln_1.bias"], f"block{i}.norm1.shift")
        _assign(block.norm2.scale, hf_sd[prefix + "ln_2.weight"], f"block{i}.norm2.scale")
        _assign(block.norm2.shift, hf_sd[prefix + "ln_2.bias"], f"block{i}.norm2.shift")

        # --- FeedForward (two linear layers, GELU in between) ---
        fc_w = hf_sd[prefix + "mlp.c_fc.weight"]
        fc_b = hf_sd[prefix + "mlp.c_fc.bias"]
        _assign(block.ff.layers[0].weight, fc_w.T, f"block{i}.ff.0.weight")
        _assign(block.ff.layers[0].bias, fc_b, f"block{i}.ff.0.bias")

        proj_w = hf_sd[prefix + "mlp.c_proj.weight"]
        proj_b = hf_sd[prefix + "mlp.c_proj.bias"]
        _assign(block.ff.layers[2].weight, proj_w.T, f"block{i}.ff.2.weight")
        _assign(block.ff.layers[2].bias, proj_b, f"block{i}.ff.2.bias")

    _assign(model.final_norm.scale, hf_sd["transformer.ln_f.weight"], "final_norm.scale")
    _assign(model.final_norm.shift, hf_sd["transformer.ln_f.bias"], "final_norm.shift")

    # GPT-2 ties its output projection to the token embedding matrix -- it's
    # the same weight used twice, not two separately-trained matrices.
    _assign(model.linear_out.weight, hf_sd["transformer.wte.weight"], "linear_out.weight")
    # The Hugging Face lm_head has no bias term at all, so zero yours out
    # rather than leaving it at its random initialization.
    with torch.no_grad():
        model.linear_out.bias.zero_()

    return model


def load_gpt2(size: str = "small", drop_rate: float = 0.0) -> tuple[GPT2Model, dict[str, int | bool]]:
    """
    Downloads OpenAI's pretrained GPT-2 weights for the given size and loads
    them into your GPT2Model architecture.

    Args:
        size: one of "small", "medium", "large", "xl"
        drop_rate: dropout probability for your model's cfg (0.0 for
            generation/evaluation; only set > 0 if you plan to continue
            training with dropout active)

    Returns:
        (model, cfg): your GPT2Model with pretrained weights loaded, and
        the cfg dict used to build it (so you can reuse it elsewhere, e.g.
        for the tokenizer's context length).
    """
    if size not in GPT2_CHECKPOINTS:
        raise ValueError(
            f"Unknown size '{size}'. Choose from: {list(GPT2_CHECKPOINTS.keys())}"
        )

    cfg = build_gpt2_config(size, drop_rate)

    hf_model = HFGPT2LMHeadModel.from_pretrained(GPT2_CHECKPOINTS[size])
    hf_model.eval()

    model = GPT2Model(cfg)
    load_weights_into_gpt2(model, hf_model, cfg)
    model.eval()

    return model, cfg
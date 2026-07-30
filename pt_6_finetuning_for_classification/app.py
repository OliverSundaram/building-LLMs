import torch
import tiktoken
import chainlit as cl

from model import GPT2Model

MODEL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}

MAX_LENGTH = 120
PAD_TOKEN_ID = 50256
WEIGHTS_PATH = "spam_not_spam_classifying_LLM.pth"

# ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = tiktoken.get_encoding("gpt2")

model = GPT2Model(MODEL_CONFIG)
state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()  # disables dropout so predictions are deterministic


def classify(text: str) -> tuple[str, float]:
    """Tokenize, pad/truncate, run the model, and return (label, confidence)."""
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    # Truncate long inputs, pad short ones — the model always expects MAX_LENGTH tokens
    ids = ids[:MAX_LENGTH]
    ids = ids + [PAD_TOKEN_ID] * (MAX_LENGTH - len(ids))

    input_tensor = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # add batch dim -> [1, MAX_LENGTH]

    with torch.no_grad():
        logits = model(input_tensor)          # shape: [1, MAX_LENGTH, 2]
        last_token_logits = logits[:, -1, :]  # the last position has attended to the whole sequence
        probs = torch.softmax(last_token_logits, dim=-1)
        pred = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, pred].item())

    label = "Spam" if pred == 1 else "Not spam"
    return label, confidence


@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "**Spam Classifier**\n\n"
            "This is a GPT-2 architecture I built from scratch in PyTorch, then fine-tuned "
            "to classify text as spam or not spam.\n\n"
            "Paste in an email or message and I'll tell you what it thinks."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    text = message.content.strip()

    if not text:
        await cl.Message(content="Send me some text to classify.").send()
        return

    label, confidence = classify(text)
    emoji = "🚫" if label == "Spam" else "✅"

    await cl.Message(content=f"{emoji} **{label}** — confidence {confidence:.1%}").send()
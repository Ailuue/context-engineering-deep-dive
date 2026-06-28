"""
Example 07 — more context is not better (context rot).
======================================================

It's tempting to treat a big context window as a reason to stuff everything in:
all the docs, the whole history, every tool result, just in case. Don't. Beyond a
point, *more* context makes answers *worse* — the relevant signal gets diluted by
noise, the model latches onto a plausible-but-irrelevant passage, and you pay for
every wasted token on every turn. Practitioners call this "context rot": quality
degrades as junk accumulates, even well under the token limit.

This example measures the cheap, undeniable half of the cost: tokens. It answers
the same question two ways — a **lean** context with just what's needed, and a
**bloated** one padded with irrelevant "retrieved" documents — and shows the token
(and therefore cost and latency) blowup for zero added value. On `PROVIDER=mock`
the answer stays correct either way (the mock isn't distractible); the quality
half of the cost is real on actual models, which is the point of keeping context
lean.

Run:  python examples/07_context_rot.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context import describe, generate, tokens

SYSTEM = "You are a helpful assistant. Answer using only this conversation."

# The one fact that answers the question.
RELEVANT = "Hi, my name is Dana and I'm on the Pro plan."

# Irrelevant "retrieved documents" — the kind of just-in-case padding that rots a
# context window: plausible, on-topic-ish, and completely useless for the question.
NOISE = [
    "Acme Cloud was founded in 2019 and is headquartered in a mid-sized city. " * 4,
    "Our changelog for version 12 includes dozens of minor fixes and tweaks. " * 4,
    "The mobile app supports dark mode, widgets, and offline drafts. " * 4,
    "Enterprise customers can request a custom data-residency region. " * 4,
    "Our status page reports uptime across six global regions every minute. " * 4,
]


def ask(messages: list[dict]) -> tuple[str, int]:
    answer = generate(SYSTEM, messages)
    return answer, tokens.estimate(SYSTEM) + tokens.estimate_messages(messages)


def main() -> None:
    print(f"Provider: {describe()}\n")
    question = "What's my name?"

    lean = [{"role": "user", "content": RELEVANT}, {"role": "user", "content": question}]
    bloated = (
        [{"role": "user", "content": RELEVANT}]
        + [{"role": "user", "content": f"(FYI doc) {n}"} for n in NOISE]
        + [{"role": "user", "content": question}]
    )

    lean_answer, lean_tokens = ask(lean)
    bloat_answer, bloat_tokens = ask(bloated)

    print(f"LEAN context    — {lean_tokens:>4} tokens -> {lean_answer}")
    print(f"BLOATED context — {bloat_tokens:>4} tokens -> {bloat_answer}")
    print(f"\nThe bloated window costs {bloat_tokens / lean_tokens:.1f}x the tokens "
          f"for the same answer —\nand you'd pay that multiplier on EVERY turn.")

    print(
        "\nTakeaway: a big window is a budget, not a goal. Padding it with "
        "just-in-case\ncontext burns tokens now and, on real models, buries the "
        "signal and lowers\nquality (context rot). Retrieve and keep what THIS turn "
        "needs — relevance beats\nvolume. The cheapest, fastest, most accurate token "
        "is the one you didn't send."
    )


if __name__ == "__main__":
    main()

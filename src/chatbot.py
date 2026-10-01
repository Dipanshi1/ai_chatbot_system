import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml import ChatResult, answer


def format_debug(result: ChatResult) -> str:
    """Render pipeline diagnostics for the --debug flag."""
    reason = result.abstain_reason.value if result.abstain_reason else "-"
    margin = f"{result.margin:.4f}" if result.margin is not None else "-"
    overlap = f"{result.overlap:.4f}" if result.overlap is not None else "-"
    similarity = f"{result.similarity:.4f}" if result.similarity is not None else "-"
    lines = [
        f"[debug] outcome          : {result.outcome.value}",
        f"[debug] reason           : {reason}",
        f"[debug] predicted intent : {result.predicted_intent or '-'}",
        f"[debug] committed intent : {result.intent or '-'}",
        f"[debug] margin           : {margin}",
        f"[debug] overlap (dice)   : {overlap}",
        f"[debug] similarity (cos) : {similarity}",
        f"[debug] matched question : {result.matched_question or '-'}",
        f"[debug] shared tokens    : {list(result.shared_content_tokens)}",
    ]
    if result.lexical is not None:
        lines.append(
            f"[debug] content tokens   : {list(result.lexical.content_tokens)} "
            f"(in-vocab {list(result.lexical.in_vocab_content_tokens)}, "
            f"oov ratio {result.lexical.content_oov_ratio:.2f})"
        )
    lines.append(
        f"[debug] versions         : {result.model_version} | "
        f"{result.kb_version} | {result.policy_version}"
    )
    return "\n".join(lines)


def run_cli(debug: bool = False) -> None:
    """Read one message from stdin and print the chatbot's answer."""
    message = input("You: ")
    result = answer(message)

    if debug:
        print(format_debug(result))
    print(f"Bot: {result.answer}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="University support chatbot (CLI)")
    parser.add_argument(
        "--debug", action="store_true", help="print pipeline diagnostics"
    )
    args = parser.parse_args(argv)
    run_cli(debug=args.debug)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from react_agent import MultiTurnReactAgent


def main():
    parser = argparse.ArgumentParser(description="Run a single ReAct inference query")
    parser.add_argument("--model", required=True, help="Path to the local model used by vLLM")
    parser.add_argument("--question", required=True, help="Single user query to run")
    parser.add_argument(
        "--port",
        type=int,
        default=6001,
        help="Port of the running vLLM server (default: 6001)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature for the model"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p sampling parameter for the model"
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=1.1,
        help="Presence penalty used during generation",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Optional path to save the raw JSON result",
    )

    args = parser.parse_args()

    llm_cfg = {
        "model": args.model,
        "generate_cfg": {
            "max_input_tokens": 320000,
            "max_retries": 10,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": args.presence_penalty,
        },
        "model_type": "qwen_dashscope",
    }

    agent = MultiTurnReactAgent(llm=llm_cfg)

    task = {
        "item": {"question": args.question, "answer": ""},
        "planning_port": args.port,
    }

    result = agent._run(task, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print(f"Saved result to {save_path}")


if __name__ == "__main__":
    main()

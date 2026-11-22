import json
from pathlib import Path
import pandas as pd
from pipeline import run_extraction_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "synthetic_dataset.json"


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def score_sample(pred: dict, gold: dict, transcript: str) -> dict:
    gold_actions = gold.get("action_items", [])
    pred_actions = pred.get("action_items", [])

    matched = 0
    owner_correct = 0
    deadline_correct = 0
    evidence_valid = 0

    for p in pred_actions:
        p_ev = normalize(p.get("evidence", ""))
        if p_ev and p_ev in normalize(transcript):
            evidence_valid += 1

        for g in gold_actions:
            if normalize(g["task"])[:18] in normalize(p.get("task", "")) or normalize(p.get("task", ""))[:18] in normalize(g["task"]):
                matched += 1
                if normalize(p.get("owner", "")) == normalize(g["owner"]):
                    owner_correct += 1
                if normalize(p.get("deadline", "")) == normalize(g["deadline"]):
                    deadline_correct += 1
                break

    precision = matched / len(pred_actions) if pred_actions else (1.0 if not gold_actions else 0.0)
    recall = matched / len(gold_actions) if gold_actions else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    owner_acc = owner_correct / len(gold_actions) if gold_actions else 1.0
    deadline_acc = deadline_correct / len(gold_actions) if gold_actions else 1.0
    ev_support = evidence_valid / len(pred_actions) if pred_actions else 1.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "owner_acc": round(owner_acc, 3),
        "deadline_acc": round(deadline_acc, 3),
        "evidence_support": round(ev_support, 3),
    }


def run_benchmark(version: str = "V3", enable_verification: bool = True, model: str = "gpt-4o-mini"):
    dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = []
    for item in dataset:
        res = run_extraction_pipeline(
            transcript=item["transcript"],
            version=version,
            enable_verification=enable_verification,
            model=model,
        )
        final_out = res.get("final", {})
        metrics = score_sample(final_out, item["gold"], item["transcript"])
        metrics["id"] = item["id"]
        metrics["category"] = item["category"]
        rows.append(metrics)

    df = pd.DataFrame(rows)
    print(f"=== Benchmark Results ({version} | Verified={enable_verification} | Model={model}) ===")
    print(df.to_string(index=False))
    print("\nAverages:")
    print(df[["precision", "recall", "f1", "owner_acc", "deadline_acc", "evidence_support"]].mean().round(3))


if __name__ == "__main__":
    run_benchmark(version="V3", enable_verification=True)

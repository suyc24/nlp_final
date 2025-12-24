# analyze_rag_gain_by_type.py

import json
from collections import defaultdict
from typing import Dict, Any


def load_results(json_path: str):
    """
    从 JSON 文件加载评测结果
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_rag_gain_by_type_from_json(json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    按 dataset_type 统计 RAG 的提升效果

    统计口径：
    - baseline_acc
    - final_acc
    - absolute_gain
    - rag_recovery_rate (baseline 错 → RAG 对)
    - rag_harmed (baseline 对 → RAG 错)
    """

    results = load_results(json_path)

    # 每个题型一个 bucket
    stats = defaultdict(lambda: {
        "total": 0,

        # baseline
        "baseline_correct": 0,

        # final
        "final_correct": 0,

        # rag usage
        "rag_activated": 0,
        "rag_correct": 0,

        # transition analysis
        "rag_recovered": 0,   # baseline wrong → final correct
        "rag_harmed": 0       # baseline correct → final wrong
    })

    for r in results:
        problem_type = r.get("dataset_type", "Unknown")
        s = stats[problem_type]

        s["total"] += 1

        baseline_ok = bool(r.get("baseline_correct", False))
        final_ok = bool(r.get("final_correct", False))

        if baseline_ok:
            s["baseline_correct"] += 1
        if final_ok:
            s["final_correct"] += 1

        # 是否启用了 RAG
        if r.get("final_method", "").startswith("RAG"):
            s["rag_activated"] += 1

            if r.get("rag_correct", False):
                s["rag_correct"] += 1

            # RAG 修正成功
            if (not baseline_ok) and final_ok:
                s["rag_recovered"] += 1

            # RAG 造成反效果
            if baseline_ok and (not final_ok):
                s["rag_harmed"] += 1

    # 生成最终报告
    report = {}

    for t, s in stats.items():
        total = s["total"]
        rag_total = s["rag_activated"]

        report[t] = {
            "total": total,

            "baseline_acc": (
                s["baseline_correct"] / total if total else 0.0
            ),

            "final_acc": (
                s["final_correct"] / total if total else 0.0
            ),

            "absolute_gain": (
                (s["final_correct"] - s["baseline_correct"]) / total
                if total else 0.0
            ),

            "rag_activated": rag_total,

            "rag_recovery_rate": (
                s["rag_recovered"] / rag_total
                if rag_total else 0.0
            ),

            "rag_recovered": s["rag_recovered"],
            "rag_harmed": s["rag_harmed"],
        }

    return report


def print_rag_gain_report(report: Dict[str, Dict[str, Any]]):
    """
    以人类可读形式打印统计结果
    """
    print("\n" + "=" * 90)
    print("📊 RAG Improvement Analysis by Problem Type")
    print("=" * 90)

    header = (
        f"{'Type':15s} | {'Total':>5s} | "
        f"{'Base Acc':>9s} | {'Final Acc':>9s} | "
        f"{'Gain':>8s} | {'RAG Recov':>10s} | {'Harmed':>7s}"
    )
    print(header)
    print("-" * len(header))

    for t, r in sorted(report.items()):
        print(
            f"{t:15s} | "
            f"{r['total']:5d} | "
            f"{r['baseline_acc']*100:8.2f}% | "
            f"{r['final_acc']*100:8.2f}% | "
            f"{r['absolute_gain']*100:+7.2f}% | "
            f"{r['rag_recovery_rate']*100:9.2f}% | "
            f"{r['rag_harmed']:7d}"
        )

    print("=" * 90)


if __name__ == "__main__":
    # 🔧 修改为你的结果文件路径
    RESULT_JSON_PATH = "math_adaptive_eval_result.json"

    report = analyze_rag_gain_by_type_from_json(RESULT_JSON_PATH)
    print_rag_gain_report(report)

import csv
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from calc_Score_pareto import compute_score_pareto

cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mask_config_path = os.path.join(project_root, "Algorithms", "mask_config.json")


def plot_pareto_objectives(all_results):
    cmap = plt.get_cmap("tab10")
    colors = [cmap(index % cmap.N) for index in range(len(all_results))]
    strategy_handles = []
    for result, color in zip(all_results, colors):
        marker = "*" if result["is_pareto_optimal"] else "o"
        label = f"epsilon={result['epsilon_il']:g}" + (" (max entropy)" if result["is_max_entropy_pareto"] else "")
        strategy_handles.append(Line2D([0], [0], marker=marker, linestyle="none", color=color,
                                       markeredgecolor="gold" if result["is_max_entropy_pareto"] else "black" if result["is_pareto_optimal"] else color,
                                       markeredgewidth=2.5 if result["is_max_entropy_pareto"] else 1.0,
                                       markersize=13 if result["is_max_entropy_pareto"] else 11 if result["is_pareto_optimal"] else 7,
                                       label=label))

    fig1 = plt.figure(figsize=(8, 6))
    ax3d = fig1.add_subplot(111, projection="3d")
    for result, color in zip(all_results, colors):
        is_pareto = result["is_pareto_optimal"]
        ax3d.scatter(result["avg_score"], result["min_rule_score"], result["policy_entropy"],
                     color=color, marker="*" if is_pareto else "o", s=180 if is_pareto else 70,
                     edgecolors="black" if is_pareto else "none", linewidths=1.2)
        if result["is_max_entropy_pareto"]:
            ax3d.scatter(result["avg_score"], result["min_rule_score"], result["policy_entropy"],
                         facecolors="none", edgecolors="gold", marker="o", s=320, linewidths=2.5)
    ax3d.set_xlabel("Average score")
    ax3d.set_ylabel("Worst score")
    ax3d.set_zlabel("Policy entropy")
    ax3d.set_title("Score Pareto front and maximum-entropy selection")
    legend1 = ax3d.legend(handles=strategy_handles, title="Strategy", loc="upper left")
    legend1.set_draggable(True)
    fig1.subplots_adjust(left=0.08, right=0.94, bottom=0.08, top=0.90)

    fig2, (score_ax, entropy_ax) = plt.subplots(1, 2, figsize=(9, 5.2), gridspec_kw={"width_ratios": [2, 1]})
    for result, color in zip(all_results, colors):
        is_pareto = result["is_pareto_optimal"]
        marker = "*" if is_pareto else "o"
        size = 180 if is_pareto else 70
        edgecolor = "black" if is_pareto else "none"
        score_ax.scatter(result["avg_score"], result["min_rule_score"], color=color, marker=marker,
                         s=size, edgecolors=edgecolor, linewidths=1.2)
        entropy_ax.scatter(0, result["policy_entropy"], color=color, marker=marker,
                           s=size, edgecolors=edgecolor, linewidths=1.2)
        if result["is_max_entropy_pareto"]:
            score_ax.scatter(result["avg_score"], result["min_rule_score"], facecolors="none",
                             edgecolors="gold", marker="o", s=320, linewidths=2.5)
            entropy_ax.scatter(0, result["policy_entropy"], facecolors="none",
                               edgecolors="gold", marker="o", s=320, linewidths=2.5)

    score_ax.set_xlabel("Average score")
    score_ax.set_ylabel("Worst score")
    score_ax.set_title("Score Pareto front")
    score_ax.grid(True, alpha=0.3)
    entropy_ax.axvline(0, color="gray", linewidth=1, alpha=0.6)
    entropy_ax.set_xlim(-0.5, 0.5)
    entropy_ax.set_xticks([])
    entropy_ax.set_ylabel("Policy entropy")
    entropy_ax.set_title("Policy entropy")
    entropy_ax.grid(True, axis="y", alpha=0.3)
    fig2.suptitle("Score Pareto front and maximum-entropy selection", y=0.96)
    legend2 = fig2.legend(handles=strategy_handles, title="Strategy", loc="upper center",
                          bbox_to_anchor=(0.5, 0.84), ncol=min(len(strategy_handles), 3))
    legend2.set_draggable(True)
    fig2.subplots_adjust(left=0.10, right=0.95, bottom=0.14, top=0.68, wspace=0.38)
    plt.show()


def main():
    with open(mask_config_path, "r", encoding="utf-8") as f:
        mask_config = json.load(f)
    ver = int(mask_config.get("ver", 0))
    hor = int(mask_config.get("hor", 0))
    mask_tag = f"v{ver}h{hor}"
    result_json = os.path.join(cur_dir, f"marwil_epsilon_sweep_results_{mask_tag}.json")
    output_json = os.path.join(cur_dir, f"marwil_pareto_recomputed_{mask_tag}.json")
    output_csv = os.path.join(cur_dir, f"marwil_pareto_recomputed_{mask_tag}.csv")

    if not os.path.isfile(result_json):
        raise FileNotFoundError(f"Result file not found for mask {mask_tag}: {result_json}")

    with open(result_json, "r", encoding="utf-8") as f:
        source = json.load(f)

    all_results = source["all_results"]
    rule_ids = source.get("config", {}).get("test_rule_ids", [0, 1, 2, 3])
    pareto_set = compute_score_pareto(all_results, rule_ids)
    objective_keys = ["avg_score", "min_rule_score"]

    output = {
        "created_at": datetime.now().isoformat(),
        "source_json": result_json,
        "mask_tag": mask_tag,
        "pareto_method": "score_pareto_then_max_entropy",
        "pareto_objectives": objective_keys,
        "all_results": all_results,
        "pareto_optimal": pareto_set,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    rule_score_keys = [f"score_vs_rule_{rule_id}" for rule_id in rule_ids]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epsilon_il", *rule_score_keys, "min_rule_score", "avg_score", "policy_entropy", "is_pareto_optimal", "is_max_entropy_pareto"])
        for result in all_results:
            writer.writerow([
                result["epsilon_il"],
                *(result[key] for key in rule_score_keys),
                result["min_rule_score"],
                result["avg_score"],
                result["policy_entropy"],
                result["is_pareto_optimal"],
                result["is_max_entropy_pareto"],
            ])

    plot_pareto_objectives(all_results)

    print(f"Loaded existing results: {result_json}")
    print(f"Mask: {mask_tag}")
    print("\n========== Full Results (score Pareto, then maximum entropy) ==========")
    for result in all_results:
        mark = "[SELECT]" if result["is_max_entropy_pareto"] else "[PARETO]" if result["is_pareto_optimal"] else "        "
        scores = ", ".join(f"r{rule_id}={result[f'score_vs_rule_{rule_id}']:.3f}" for rule_id in rule_ids)
        print(f"  {mark} epsilon_il={result['epsilon_il']:5.2f}: {scores}, min={result['min_rule_score']:.3f}, avg={result['avg_score']:.3f}, entropy={result['policy_entropy']:.4f}")

    print("\n========== Pareto Optimal Set (average score, worst score) ==========")
    for result in pareto_set:
        scores = ", ".join(f"r{rule_id}={result[f'score_vs_rule_{rule_id}']:.3f}" for rule_id in rule_ids)
        print(f"  epsilon_il={result['epsilon_il']:5.2f}: {scores}, min={result['min_rule_score']:.3f}, avg={result['avg_score']:.3f}, entropy={result['policy_entropy']:.4f}")
    print(f"\nSaved JSON: {output_json}")
    print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
    main()

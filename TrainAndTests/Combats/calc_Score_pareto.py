def compute_score_pareto(results, rule_ids, score_tolerance=1e-9):
    if not results:
        return []
    rule_score_keys = [f"score_vs_rule_{rule_id}" for rule_id in rule_ids]
    for result in results:
        scores = [float(result[key]) for key in rule_score_keys]
        result["avg_score"] = sum(scores) / len(scores)
        result["min_rule_score"] = min(scores)

    objective_keys = ["avg_score", "min_rule_score"]
    pareto = []
    for result in results:
        dominated = False
        for other in results:
            if other is result:
                continue
            not_worse = all(other[key] >= result[key] - score_tolerance for key in objective_keys)
            strictly_better = any(other[key] > result[key] + score_tolerance for key in objective_keys)
            if not_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(result)

    pareto_ids = {id(result) for result in pareto}
    max_pareto_entropy = max(result["policy_entropy"] for result in pareto)
    for result in results:
        result.pop("has_best_shortboard", None)
        result.pop("pareto_eligible", None)
        result["is_pareto_optimal"] = id(result) in pareto_ids
        result["is_max_entropy_pareto"] = (
            result["is_pareto_optimal"]
            and max_pareto_entropy - result["policy_entropy"] <= score_tolerance
        )
    pareto.sort(key=lambda result: (result["policy_entropy"], result["avg_score"], result["min_rule_score"]), reverse=True)
    return pareto

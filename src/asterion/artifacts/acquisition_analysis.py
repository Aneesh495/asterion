"""Check acquisition costs and paired outcomes against retained work and author coordinates."""

import numpy as np

from asterion.artifacts.io import file_hash, read_json, safe_member
from asterion.contracts.errors import IntegrityError


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def tied_ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        ranks[order[begin:end]] = (begin + end - 1) / 2
        begin = end
    return ranks


def verify_uncertainty_summary(summary, scores, errors, bounds):
    values = [np.asarray(value) for value in (scores, errors, bounds)]
    require(
        all(
            value.ndim == 1
            and value.shape == values[0].shape
            and len(value) > 0
            and np.isfinite(value).all()
            and (value >= 0).all()
            for value in values
        ),
        "uncertainty observations require finite nonnegative matched geometry axes",
    )
    score_rank, error_rank = [tied_ranks(value) for value in values[:2]]
    score_rank -= score_rank.mean()
    error_rank -= error_rank.mean()
    denominator = np.linalg.norm(score_rank) * np.linalg.norm(error_rank)
    correlation = float(np.dot(score_rank, error_rank) / denominator) if denominator else None
    require(
        (
            summary["spearman"] is None
            if correlation is None
            else summary["spearman"] is not None
            and np.isclose(summary["spearman"], correlation, atol=1e-12, rtol=1e-12)
        ),
        "committee error-ranking correlation differs from independently ranked observations",
    )
    scale = float(values[2].max())
    mean_bound = float(np.sum(values[2] / scale) / len(values[2]) * scale) if scale else 0.0
    require(
        type(summary["geometries"]) is int
        and summary["geometries"] == len(values[0])
        and np.isclose(summary["coverage"], np.mean(values[1] <= values[2]), atol=1e-12, rtol=0)
        and np.isclose(summary["mean_bound"], mean_bound, atol=1e-12, rtol=1e-12),
        "committee coverage, bound scale or geometry denominator differs from raw observations",
    )


def verify_interval(value, outer=None):
    start, end = value["started_ns"], value["ended_ns"]
    require(
        type(start) is int
        and type(end) is int
        and 0 < start < end
        and np.isfinite(value["seconds"])
        and np.isclose(value["seconds"], (end - start) * 1e-9, rtol=1e-12, atol=0),
        "acquisition duration disagrees with retained clock readings",
    )
    if outer is not None:
        require(
            outer["started_ns"] <= start < end <= outer["ended_ns"],
            "acquisition phase clock escapes its recorded round",
        )


def verify_cost(directory, row):
    record = read_json(directory / f"cost-{row['round']}.json")
    require(
        record == row["cost"] and record["format"] == "asterion-acquisition-cost-v1",
        "acquisition cost summary differs from retained observations",
    )
    outer = record["clock"]
    verify_interval(outer)
    require(
        len(record["members"]) == 3 and [r["member"] for r in record["members"]] == [1, 2, 3],
        "acquisition cost omitted or repeated a committee member",
    )
    last = outer["started_ns"]
    if row["round"]:
        verify_interval(record["decision_and_reveal"], outer)
        last = record["decision_and_reveal"]["ended_ns"]
    else:
        require(
            record["decision_and_reveal"] is None,
            "shared initial round invents acquisition decisions",
        )
    updates = examples = 0
    epoch_seconds = 0.0
    for member, entry in enumerate(record["members"], 1):
        path = safe_member(directory, row["members"][member - 1]) / "history.json"
        history = read_json(path)
        require(
            entry["history_sha256"] == file_hash(path)
            and entry["epochs"] == len(history) > 0
            and entry["updates"] == history[-1]["updates"]
            and entry["examples_processed"] == history[-1]["examples_processed"],
            "acquisition cost changes completed optimizer or label work",
        )
        elapsed = sum(r["elapsed_seconds"] for r in history)
        require(
            all(np.isfinite(r["elapsed_seconds"]) and r["elapsed_seconds"] > 0 for r in history)
            and np.isclose(
                entry["epoch_train_and_validation_seconds"], elapsed, rtol=1e-12, atol=1e-12
            ),
            "acquisition epoch cost differs from retained training",
        )
        verify_interval(entry["call"], outer)
        require(
            entry["call"]["started_ns"] >= last and entry["call"]["seconds"] + 1e-6 >= elapsed,
            "acquisition training clocks overlap or omit recorded epoch work",
        )
        last = entry["call"]["ended_ns"]
        updates += entry["updates"]
        examples += entry["examples_processed"]
        epoch_seconds += elapsed
    for phase in ("calibration_and_evaluation", "candidate_inference"):
        verify_interval(record[phase], outer)
        require(record[phase]["started_ns"] >= last, "acquisition phases overlap or change order")
        last = record[phase]["ended_ns"]
    require(
        record["completed_optimizer_updates"] == updates
        and record["processed_training_examples"] == examples
        and np.isclose(record["epoch_seconds"], epoch_seconds, atol=1e-12, rtol=1e-12)
        and record["includes_checkpoint_and_observation_io"] is True
        and record["excludes_shared_preprocessing_and_initial_reveal"] is True
        and record["excludes_cost_summary_and_final_pairing_publication"] is True
        and record["oracle_cost_claim"]
        == "released-label access; no electronic-structure calculation measured",
        "acquisition total cost inflates work or claims unmeasured oracle calculation",
    )


def author_descriptors(coordinates, numbers):
    pairs: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for a in range(len(numbers)):
        for b in range(a + 1, len(numbers)):
            pair = tuple(sorted((int(numbers[a]), int(numbers[b]))))
            pairs.setdefault(pair, []).append((a, b))
    columns = []
    for pair in sorted(pairs):
        distances = np.stack(
            [
                np.sqrt(np.sum((coordinates[:, a] - coordinates[:, b]) ** 2, axis=1))
                for a, b in pairs[pair]
            ],
            axis=1,
        )
        columns.append(np.sort(distances, axis=1))
    descriptors = np.concatenate(columns, axis=1)
    return (descriptors - descriptors.mean(0)) / np.maximum(descriptors.std(0), 1e-6)


def verify_diversity(saved, coordinates, numbers, pool_rows, identities):
    rows = [int(identity.rsplit(":", 1)[1]) for identity in identities]
    values = author_descriptors(coordinates, numbers)
    lookup = {row: index for index, row in enumerate(pool_rows)}
    selected = values[[lookup[row] for row in rows]]
    require(
        saved["selected_geometries"] == len(selected) >= 2
        and saved["descriptor_dimensions"] == values.shape[1]
        and saved["normalization_scope"] == "all-permitted-candidate-coordinates",
        "acquisition diversity changed geometry or coordinate scope",
    )
    nearest = []
    total = 0.0
    for index, point in enumerate(selected):
        distances = np.sqrt(np.sum((selected - point) ** 2, axis=1))
        total += float(distances.sum())
        distances[index] = np.inf
        nearest.append(float(distances.min()))
    require(
        np.isclose(
            saved["mean_nearest_descriptor_distance"], np.mean(nearest), atol=1e-12, rtol=1e-9
        )
        and np.isclose(
            saved["mean_pairwise_descriptor_distance"],
            total / (len(rows) * (len(rows) - 1)),
            atol=1e-12,
            rtol=1e-9,
        ),
        "acquisition selected diversity disagrees with author coordinates",
    )


def verify_paired_policies(base, outcomes, datasets):
    saved = read_json(base / "paired-policy-outcomes.json")
    expected = {
        (row["molecule"], row["trial"], row["budget"], policy)
        for row in outcomes
        for policy in ("diversity", "uncertainty-diversity")
    }
    require(
        len(saved) == len(expected)
        and {(r["molecule"], r["trial"], r["budget"], r["policy"]) for r in saved} == expected,
        "paired policy outcome population is missing or repeated",
    )
    for row in saved:
        key = (row["molecule"], row["trial"], row["budget"])
        selected = [r for r in outcomes if (r["molecule"], r["trial"], r["budget"]) == key]
        require(
            len(selected) == 3
            and {entry["policy"] for entry in selected}
            == {"uniform", "diversity", "uncertainty-diversity"},
            "paired acquisition control arms are missing or repeated",
        )
        data = {}
        for entry in selected:
            path = safe_member(base, entry["path"]) / f"uncertainty-evaluation-{entry['round']}.npz"
            with np.load(path, allow_pickle=False) as raw:
                data[entry["policy"]] = {
                    "identities": raw["identities"].copy(),
                    "errors": raw["errors"].copy(),
                }
        first, second = data["uniform"], data[row["policy"]]
        require(
            np.array_equal(first["identities"], second["identities"])
            and row["control"] == "uniform"
            and row["geometries"] == len(first["identities"])
            and row["temporal_block_span"] == 1000
            and row["independent_geometry_claim"] is False,
            "paired acquisition comparison changes held-out units or their dependence",
        )
        require(
            np.isclose(
                row["uniform_geometry_force_mae"], first["errors"].mean(), atol=1e-12, rtol=1e-10
            )
            and np.isclose(
                row["policy_geometry_force_mae"], second["errors"].mean(), atol=1e-12, rtol=1e-10
            ),
            "paired acquisition summary changes measured geometry errors",
        )
        indices = [int(identity.rsplit(":", 1)[1]) for identity in first["identities"]]
        with np.load(datasets[f"rmd17_{row['molecule']}.npz"], allow_pickle=False) as author:
            blocks = author["old_indices"][indices] // 1000
        unique = np.unique(blocks)
        interval = row["paired_interval"]
        if len(unique) < 2:
            require(
                interval == {"available": False, "reason": "fewer-than-two-temporal-blocks"},
                "single temporal block invents a policy confidence interval",
            )
            continue
        require(
            interval["available"] is True
            and interval["blocks"] == len(unique)
            and interval["seed"] == 491
            and interval["repetitions"] == 2000,
            "paired acquisition bootstrap workload changed",
        )
        difference = first["errors"] - second["errors"]
        sums = np.array([difference[blocks == block].sum() for block in unique])
        counts = np.array([(blocks == block).sum() for block in unique])
        draws = np.random.default_rng(491).integers(0, len(unique), size=(2000, len(unique)))
        samples = sums[draws].sum(1) / counts[draws].sum(1)
        expected_values = [
            difference.mean(),
            np.quantile(samples, 0.025),
            np.quantile(samples, 0.975),
        ]
        require(
            np.allclose(
                [interval[key] for key in ("mean_improvement", "lower_95", "upper_95")],
                expected_values,
                atol=1e-12,
                rtol=1e-9,
            ),
            "paired policy interval disagrees with retained temporal blocks",
        )

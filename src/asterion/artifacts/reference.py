import math

from asterion.artifacts.io import read_json_lines
from asterion.contracts.errors import IntegrityError


def reference_history(path, epochs, budget, batch_size):
    """Count executed optimizer batches and validation records from MACE's raw journal."""
    if budget % batch_size:
        raise IntegrityError("reference drop-last policy changes the matched label workload")
    rows = read_json_lines(path)
    updates = {epoch: 0 for epoch in range(epochs)}
    validation = {epoch: 0 for epoch in range(epochs)}
    initial = 0
    for row in rows:
        if row["mode"] not in {"opt", "eval"}:
            raise IntegrityError("unknown reference training journal mode")
        if (
            not math.isfinite(row["loss"])
            or row["loss"] < 0
            or not math.isfinite(row["time"])
            or row["time"] <= 0
        ):
            raise IntegrityError("nonfinite or invalid reference training observation")
        epoch = row["epoch"]
        if epoch is None:
            if row["mode"] != "eval":
                raise IntegrityError("reference optimizer work has no epoch identity")
            initial += 1
            continue
        if type(epoch) is not int or epoch not in updates:
            raise IntegrityError("reference training epoch is outside the frozen workload")
        if row["mode"] == "opt":
            updates[epoch] += 1
        else:
            if row.get("head") != "Default" or not math.isfinite(row["mae_f"]) or row["mae_f"] < 0:
                raise IntegrityError("reference validation identity or force error changed")
            validation[epoch] += 1
    if (
        initial != 1
        or any(count != budget // batch_size for count in updates.values())
        or any(count != 1 for count in validation.values())
    ):
        raise IntegrityError("reference raw optimizer or validation workload is incomplete")
    return {
        "epochs": epochs,
        "updates": sum(updates.values()),
        "examples_processed": sum(updates.values()) * batch_size,
        "initial_validation": initial,
    }

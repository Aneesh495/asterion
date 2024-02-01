"""Reject ambiguous dataset row identities before any integer conversion or label access."""

import numpy as np

from asterion.contracts.errors import ContractError


def checked_rows(values, size=None):
    try:
        supplied = list(values)
    except TypeError as error:
        raise ContractError(
            "dataset row selection requires an iterable of integer identities"
        ) from error
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in supplied
    ):
        raise ContractError(
            "dataset row identities must be integers, not masks, strings or fractions"
        )
    rows = [int(value) for value in supplied]
    if any(row < 0 or (size is not None and row >= size) for row in rows):
        raise ContractError("dataset row identity is outside the coordinate pool")
    return rows

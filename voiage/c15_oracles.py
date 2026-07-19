"""Independent high-precision scientific oracle helpers for C15."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, InvalidOperation, localcontext

import numpy as np
import numpy.typing as npt


def _decimal_matrix(values: object) -> tuple[tuple[Decimal, ...], ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("net benefit must be a matrix")
    rows: list[tuple[Decimal, ...]] = []
    for raw_row in values:
        if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes)):
            raise TypeError("net benefit rows must be arrays")
        try:
            row = tuple(Decimal(str(value)) for value in raw_row)
        except InvalidOperation as exc:
            raise ValueError("net benefit values must be decimal numbers") from exc
        if any(not value.is_finite() for value in row):
            raise ValueError("net benefit values must be finite")
        rows.append(row)
    widths = {len(row) for row in rows}
    if not rows or widths == {0} or len(widths) != 1 or next(iter(widths)) < 2:
        raise ValueError("net benefit must be a non-empty rectangular matrix")
    return tuple(rows)


def decimal_evpi(values: object) -> Decimal:
    """Calculate EVPI with only Decimal arithmetic and literal matrix semantics."""
    rows = _decimal_matrix(values)
    draws = Decimal(len(rows))
    with localcontext() as context:
        context.prec = 50
        perfect = sum(max(row) for row in rows) / draws
        current = max(
            sum(row[index] for row in rows) / draws for index in range(len(rows[0]))
        )
        return perfect - current


def numpy_evpi(values: npt.ArrayLike) -> float:
    """Calculate the same generic EVPI reduction through NumPy."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("net benefit must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("net benefit values must be finite")
    return float(np.mean(np.max(matrix, axis=1)) - np.max(np.mean(matrix, axis=0)))


__all__ = ["decimal_evpi", "numpy_evpi"]

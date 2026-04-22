"""
tools/date_range.py

Flexible date range specification for densities, tuning, and strategy testing.

Format: "1990-2007, 2010-2019, 2021"
  - Ranges: "YYYY-YYYY" (inclusive on both ends)
  - Single years: "YYYY"
  - Comma-separated, whitespace ignored

Examples:
  "2010-2022"              → 2010 through 2022
  "1990-2007, 2010-2019"  → exclude 2008-2009
  "2010-2019, 2021-2022"  → exclude 2020
  "2021"                  → single year only
"""

import numpy as np
import re


def parse_date_mask(spec: str, dates: np.ndarray) -> np.ndarray:
    """
    Parse a date range spec string into a boolean mask over dates.

    Parameters
    ----------
    spec   : str   e.g. "1990-2007, 2010-2019, 2021"
    dates  : np.ndarray of datetime-like objects (from DataStore/DataLoader)

    Returns
    -------
    mask   : np.ndarray[bool], shape (len(dates),)
    """
    years = _dates_to_years(dates)
    mask = np.zeros(len(dates), dtype=bool)

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            tokens = part.split("-")
            if len(tokens) != 2:
                raise ValueError(f"Invalid range segment: '{part}'")
            start, end = int(tokens[0].strip()), int(tokens[1].strip())
            if start > end:
                raise ValueError(f"Range start > end: '{part}'")
            mask |= (years >= start) & (years <= end)
        else:
            year = int(part)
            mask |= (years == year)

    return mask


def describe_mask(spec: str, mask: np.ndarray, dates: np.ndarray) -> str:
    """
    Return a human-readable summary of the mask.

    Example output:
      Spec:       "1990-2007, 2010-2019"
      Total days: 4523 / 7841 (57.7%)
      Ranges:     1990-01-02 → 2007-12-31, 2010-01-04 → 2019-12-31
    """
    total = len(dates)
    selected = int(mask.sum())
    pct = 100.0 * selected / total if total > 0 else 0.0

    # Find contiguous runs of True in the mask
    runs = []
    in_run = False
    for i, v in enumerate(mask):
        if v and not in_run:
            run_start = i
            in_run = True
        elif not v and in_run:
            runs.append((run_start, i - 1))
            in_run = False
    if in_run:
        runs.append((run_start, len(mask) - 1))

    range_strs = [
        f"{str(dates[s])[:10]} → {str(dates[e])[:10]}"
        for s, e in runs
    ]

    lines = [
        f"  Spec:       \"{spec}\"",
        f"  Total days: {selected} / {total} ({pct:.1f}%)",
        f"  Ranges:     " + (", ".join(range_strs) if range_strs else "(none)"),
    ]
    return "\n".join(lines)


def validate_spec(spec: str) -> bool:
    """Return True if spec is syntactically valid."""
    try:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                tokens = part.split("-")
                assert len(tokens) == 2
                start, end = int(tokens[0]), int(tokens[1])
                assert start <= end
            else:
                int(part)
        return True
    except Exception:
        return False


def spec_from_years(start_year, end_year) -> str:
    """Convert old-style (start_year, end_year) to spec string. Backward compat."""
    if start_year is None or end_year is None:
        return None
    return f"{start_year}-{end_year}"


def _dates_to_years(dates: np.ndarray) -> np.ndarray:
    """Extract integer years from a dates array."""
    try:
        # numpy datetime64
        return dates.astype("datetime64[Y]").astype(int) + 1970
    except Exception:
        # Python date objects
        return np.array([d.year for d in dates], dtype=int)

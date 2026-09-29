"""Shared human-readable dashboard formatting."""

import math


def top_percent_label(percentile: float | None) -> str:
    """Convert a higher-is-better percentile to an intuitive top-share label."""
    if percentile is None or math.isnan(float(percentile)):
        return "—"
    top_share = max(1, min(100, math.ceil(100 - float(percentile))))
    return f"Top {top_share}%"

"""Timestamped progress logging for long-running commands."""
from datetime import datetime


def log(*args, **kwargs):
    """print() with a local date-time prefix, e.g. '2026-11-12 07:17:32 [phi3-mini] 50/1800 done'."""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *args, flush=True, **kwargs)

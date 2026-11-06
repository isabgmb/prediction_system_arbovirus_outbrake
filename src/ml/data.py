"""
ml/data.py
----------
City-wide weekly series for modelling, from the raw files or the database.

The paper models the city as a whole: for each epidemiological week, the
mean over the 36 neighborhoods of dengue cases (MATLAB: nanmean(DengueOcc, 2))
and of the egg density index (nanmean(ODI_Mat, 2)).

    python -m src.ml.data        # checks that both sources give the same series
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from src.core import epiweek

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"


@dataclass(frozen=True)
class CitySeries:
    dengue: np.ndarray                 # mean cases per neighborhood, per week
    edi:    np.ndarray                 # mean egg density index, per week
    weeks:  list[tuple[int, int]]      # (epi_year, epi_week)

    @property
    def week_starts(self) -> list[date]:
        return [epiweek.week_start(y, w) for y, w in self.weeks]

    def series(self, input_type: str) -> np.ndarray:
        return self.dengue if input_type == "dengue" else self.edi


def load_from_raw(data_raw: Path = DATA_RAW) -> CitySeries:
    """Exactly what Run1DLSTM_01.m computes from the same files."""
    df = pd.read_excel(data_raw / "Dados_Modelagem.xlsx", sheet_name="Casos de Dengue", index_col=0)
    weeks = [epiweek.parse_label(c) for c in df.columns]
    dengue = np.nanmean(df.to_numpy(dtype=float), axis=0)
    egg = sio.loadmat(str(data_raw / "Rearranged_Data.mat"), squeeze_me=True)["EggIndice_Agregated"]
    edi = np.nanmean(egg[:, :, 1], axis=0)      # row order does not affect the mean
    if len(edi) != len(weeks):
        raise ValueError(f"EDI has {len(edi)} weeks, dengue has {len(weeks)}")
    return CitySeries(dengue=dengue, edi=edi, weeks=weeks)


def contiguous_weeks(first: tuple[int, int], last: tuple[int, int]) -> list[tuple[int, int]]:
    """Every epidemiological week from first to last, inclusive."""
    weeks, w = [], first
    while epiweek.week_start(*w) <= epiweek.week_start(*last):
        weeks.append(w)
        w = epiweek.add_weeks(*w, 1)
    return weeks


def week_label(week: tuple[int, int]) -> str:
    return f"{week[0]}-{week[1]:02d}"


def describe_weeks(weeks: list[tuple[int, int]], limit: int = 6) -> str:
    """'2019-14, 2019-15 and 3 more' for error messages."""
    labels = [week_label(w) for w in weeks]
    if len(labels) <= limit:
        return ", ".join(labels)
    return f"{', '.join(labels[:limit])} and {len(labels) - limit} more"


def missing_weeks(values: np.ndarray, weeks: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    """
    Gaps in a series: weeks without a value between its first and last
    values, and the last week with a value (None if there is none). Weeks
    after the last value are not gaps: that data may simply not have arrived.
    """
    has = ~np.isnan(np.asarray(values, dtype=float))
    if not has.any():
        return [], None
    first, last = int(np.argmax(has)), len(has) - 1 - int(np.argmax(has[::-1]))
    return [weeks[i] for i in range(first, last + 1) if not has[i]], weeks[last]


async def load_from_db(session) -> CitySeries:
    """
    Same series from the database, with one entry per epidemiological week
    from the first to the last week with dengue data. A week without any
    record is kept as NaN in its place instead of being dropped, which would
    shift every later week and make the LSTM read weeks that are not
    consecutive. Weeks without EDI are NaN as well (no filling: filling with
    the series mean would use future weeks).
    """
    from src.api.repositories.dengue_repository import DengueRepository
    from src.api.repositories.ovitrap_repository import OvitrapRepository

    dengue_rows = await DengueRepository(session).get_city_wide_mean()
    edi_rows    = await OvitrapRepository(session).get_city_wide_edi_mean()
    if not dengue_rows:
        raise ValueError("No dengue data in the database. Run the ETL first.")
    dengue_by_week = {(r["epi_year"], r["epi_week"]): r["mean_cases"] for r in dengue_rows}
    edi_by_week = {(r["epi_year"], r["epi_week"]): r["mean_edi"] for r in edi_rows}
    first = (dengue_rows[0]["epi_year"], dengue_rows[0]["epi_week"])
    last = (dengue_rows[-1]["epi_year"], dengue_rows[-1]["epi_week"])
    weeks = contiguous_weeks(first, last)
    return CitySeries(
        dengue=np.array([dengue_by_week.get(w, np.nan) for w in weeks], dtype=float),
        edi=np.array([edi_by_week.get(w, np.nan) for w in weeks], dtype=float),
        weeks=weeks,
    )


def compare(a: CitySeries, b: CitySeries) -> list[str]:
    """Differences between two series (empty list = identical)."""
    problems = []
    if a.weeks != b.weeks:
        problems.append(f"weeks differ ({len(a.weeks)} vs {len(b.weeks)})")
    else:
        for name in ("dengue", "edi"):
            x, y = getattr(a, name), getattr(b, name)
            if not np.allclose(x, y, equal_nan=True, rtol=1e-9, atol=1e-9):
                worst = int(np.nanargmax(np.abs(x - y)))
                problems.append(f"{name} differs, largest gap at {a.weeks[worst]}: {x[worst]} vs {y[worst]}")
    return problems


async def _check() -> None:
    from src.core.database import AsyncSessionLocal, engine
    raw = load_from_raw()
    try:
        async with AsyncSessionLocal() as session:
            db = await load_from_db(session)
    finally:
        await engine.dispose()
    problems = compare(raw, db)
    print(f"{len(raw.weeks)} weeks, {raw.weeks[0]} to {raw.weeks[-1]}")
    print("Raw files and database give identical series." if not problems else "\n".join(problems))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(_check())

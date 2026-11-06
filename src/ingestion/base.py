"""
ingestion/base.py
-----------------
Records, the source interface, and the loader shared by every source.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import DataSource, DengueCase, EpiWeek, Forecast, Neighborhood, OvitrapReading
from src.api.repositories.week_repository import WeekRepository
from src.core.names import name_key

log = logging.getLogger("sipsa.ingestion")


@dataclass(frozen=True)
class WeeklyCases:
    neighborhood: str          # official code or name (spelling variants are matched)
    epi_year:     int
    epi_week:     int
    cases:        int | None
    disease:      str = "dengue"


@dataclass(frozen=True)
class WeeklyOvitrap:
    neighborhood: str
    epi_year:     int
    epi_week:     int
    opi:          float | None
    edi:          float | None


class WeeklySource(Protocol):
    """Anything that can provide weekly records. Implement fetch() for a new feed."""
    name:        str            # stored in data_source.name (unique per source)
    file_or_url: str | None
    license:     str | None
    citation:    str | None

    def fetch(self) -> Iterable[WeeklyCases | WeeklyOvitrap]: ...


@dataclass
class DataHealth:
    """What in the stored data keeps forecasts from running (see forecast_service.history_problems)."""
    first_week:          tuple[int, int]
    last_week:           tuple[int, int]
    case_gaps:           list[tuple[int, int]]                 # weeks without any case record
    partial_case_weeks:  list[tuple[tuple[int, int], int]]     # (week, neighborhoods reporting)
    n_neighborhoods:     int
    edi_gaps:            list[tuple[int, int]]                 # weeks without EDI, between EDI's first and last
    last_edi_week:       tuple[int, int] | None

    def warnings(self) -> list[str]:
        from src.ml.data import describe_weeks, week_label
        out = []
        if self.case_gaps:
            out.append(f"No case records in {describe_weeks(self.case_gaps)}. No forecast can be made from "
                       "these weeks or later ones until they are loaded.")
        if self.edi_gaps:
            out.append(f"No EDI in {describe_weeks(self.edi_gaps)}. EDI-model forecasts from these weeks or "
                       "later ones fail until they are loaded.")
        if self.partial_case_weeks:
            listed = ", ".join(f"{week_label(w)} ({n} of {self.n_neighborhoods})" for w, n in self.partial_case_weeks[:6])
            more = f" and {len(self.partial_case_weeks) - 6} more" if len(self.partial_case_weeks) > 6 else ""
            out.append(f"Only some neighborhoods reported cases in {listed}{more}. Dengue-model forecasts from "
                       "these weeks or later ones fail until they are complete (the latest weeks may still be "
                       "arriving); a neighborhood with no cases should be reported as 0, not left out.")
        return out


async def data_health(session: AsyncSession) -> DataHealth | None:
    """Gaps and partial weeks in the stored city-wide series (None if there are no cases yet)."""
    from src.ml import data as ml_data
    try:
        series = await ml_data.load_from_db(session)
    except ValueError:
        return None
    n_total = (await session.execute(select(func.count()).select_from(Neighborhood))).scalar_one()
    rows = (await session.execute(
        select(EpiWeek.epi_year, EpiWeek.epi_week, func.count(DengueCase.case_count))
        .join(DengueCase, DengueCase.epi_week_id == EpiWeek.id)
        .where(DengueCase.disease == "dengue")
        .group_by(EpiWeek.id, EpiWeek.epi_year, EpiWeek.epi_week))).all()
    reporting = {(y, w): n for y, w, n in rows}
    case_gaps, _ = ml_data.missing_weeks(series.dengue, series.weeks)
    edi_gaps, last_edi = ml_data.missing_weeks(series.edi, series.weeks)
    partial = [(w, reporting[w]) for w in series.weeks if 0 < reporting.get(w, 0) < n_total]
    return DataHealth(series.weeks[0], series.weeks[-1], case_gaps, partial, n_total, edi_gaps, last_edi)


@dataclass
class IngestReport:
    inserted:            int = 0
    updated:             int = 0
    unchanged:           int = 0
    unknown_neighborhoods: set[str] = field(default_factory=set)
    weeks:               set[tuple[int, int]] = field(default_factory=set)
    forecasts_cleared:   int = 0
    health:              DataHealth | None = None


async def _source_id(session: AsyncSession, source: WeeklySource) -> int:
    row = (await session.execute(select(DataSource).where(DataSource.name == source.name))).scalar_one_or_none()
    if row is None:
        row = DataSource(name=source.name)
        session.add(row)
    row.file_or_url, row.license, row.citation = source.file_or_url, source.license, source.citation
    row.imported_at = datetime.now(timezone.utc)
    await session.flush()
    return row.id


async def ingest(session: AsyncSession, source: WeeklySource) -> IngestReport:
    """
    Write a source's records in one transaction (the caller commits).
    Existing rows for the same neighborhood, week (and disease) are updated.
    Stored forecasts made as of any week from the earliest changed week on
    are deleted, since they were computed without the new data. Afterwards
    the whole stored series is checked (report.health): gaps and partial
    weeks are not refused, since a later load may be the one that fills
    them, but report.health.warnings() names the weeks they affect.
    """
    report = IngestReport()
    src_id = await _source_id(session, source)
    by_key: dict[str, int] = {}
    for n in (await session.execute(select(Neighborhood))).scalars():
        by_key[name_key(n.name)] = n.id
        if n.code:
            by_key[n.code.lstrip("0") or "0"] = n.id
    weeks = WeekRepository(session)
    changed_weeks: list[EpiWeek] = []

    for rec in source.fetch():
        ref = str(rec.neighborhood).strip()
        nid = by_key.get(ref.lstrip("0") or "0") if ref.isdigit() else by_key.get(name_key(ref))
        if nid is None:
            report.unknown_neighborhoods.add(ref)
            continue
        week = (await weeks.ensure([(rec.epi_year, rec.epi_week)]))[(rec.epi_year, rec.epi_week)]
        report.weeks.add((rec.epi_year, rec.epi_week))

        if isinstance(rec, WeeklyCases):
            row = (await session.execute(select(DengueCase).where(
                DengueCase.neighborhood_id == nid, DengueCase.epi_week_id == week.id,
                DengueCase.disease == rec.disease))).scalar_one_or_none()
            values = {"case_count": rec.cases}
            new = DengueCase(neighborhood_id=nid, epi_week_id=week.id, disease=rec.disease, source_id=src_id, **values)
        else:
            row = (await session.execute(select(OvitrapReading).where(
                OvitrapReading.neighborhood_id == nid, OvitrapReading.epi_week_id == week.id))).scalar_one_or_none()
            values = {"opi": rec.opi, "edi": rec.edi}
            new = OvitrapReading(neighborhood_id=nid, epi_week_id=week.id, source_id=src_id, **values)

        if row is None:
            session.add(new)
            report.inserted += 1
        elif any(getattr(row, k) != v for k, v in values.items()):
            for k, v in values.items():
                setattr(row, k, v)
            row.source_id = src_id
            report.updated += 1
        else:
            report.unchanged += 1
            continue
        changed_weeks.append(week)

    await session.flush()
    if changed_weeks:
        earliest = min(w.start_date for w in changed_weeks)
        stale = select(EpiWeek.id).where(EpiWeek.start_date >= earliest)
        report.forecasts_cleared = (await session.execute(
            delete(Forecast).where(Forecast.as_of_week_id.in_(stale)))).rowcount
    if report.unknown_neighborhoods:
        log.warning("Skipped records for unknown neighborhoods: %s", sorted(report.unknown_neighborhoods))
    report.health = await data_health(session)          # the caller shows report.health.warnings()
    return report

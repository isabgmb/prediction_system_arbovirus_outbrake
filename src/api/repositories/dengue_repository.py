"""
api/repositories/dengue_repository.py
Dengue case queries — never mixed with ovitrap data.
"""
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.models.db_models import DengueCase


class DengueRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_series(
        self,
        neighborhood_id: int,
        iso_year_from:   int | None = None,
        iso_year_to:     int | None = None,
    ) -> list[DengueCase]:
        stmt = (
            select(DengueCase)
            .where(DengueCase.neighborhood_id == neighborhood_id)
            .order_by(DengueCase.iso_year, DengueCase.iso_week)
        )
        if iso_year_from:
            stmt = stmt.where(DengueCase.iso_year >= iso_year_from)
        if iso_year_to:
            stmt = stmt.where(DengueCase.iso_year <= iso_year_to)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_city_wide_mean(self) -> list[dict]:
        """
        City-wide weekly mean — replicates MATLAB nanmean(DengueOcc, 2).
        Used by AggregationService to build the LSTM input array.
        """
        stmt = (
            select(
                DengueCase.iso_year,
                DengueCase.iso_week,
                DengueCase.week_start,
                func.avg(DengueCase.case_count).label("mean_cases"),
            )
            .where(DengueCase.case_count.isnot(None))
            .group_by(DengueCase.iso_year, DengueCase.iso_week, DengueCase.week_start)
            .order_by(DengueCase.iso_year, DengueCase.iso_week)
        )
        result = await self.db.execute(stmt)
        return [
            {
                "iso_year":   r.iso_year,
                "iso_week":   r.iso_week,
                "week_start": r.week_start,
                "mean_cases": float(r.mean_cases),
            }
            for r in result.all()
        ]

    async def count_available_weeks(self, neighborhood_id: int) -> int:
        stmt = (
            select(func.count())
            .select_from(DengueCase)
            .where(DengueCase.neighborhood_id == neighborhood_id)
            .where(DengueCase.case_count.isnot(None))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one()

    async def get_latest_week(self) -> dict | None:
        stmt = (
            select(DengueCase.iso_year, DengueCase.iso_week, DengueCase.week_start)
            .order_by(DengueCase.iso_year.desc(), DengueCase.iso_week.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        row = result.one_or_none()
        return {"iso_year": row.iso_year, "iso_week": row.iso_week,
                "week_start": row.week_start} if row else None

"""
api/repositories/ovitrap_repository.py
OPI and EDI queries — never mixed with dengue case data.
"""
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.models.db_models import OvitrapReading


class OvitrapRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_series(
        self,
        neighborhood_id: int,
        iso_year_from:   int | None = None,
        iso_year_to:     int | None = None,
    ) -> list[OvitrapReading]:
        stmt = (
            select(OvitrapReading)
            .where(OvitrapReading.neighborhood_id == neighborhood_id)
            .order_by(OvitrapReading.iso_year, OvitrapReading.iso_week)
        )
        if iso_year_from:
            stmt = stmt.where(OvitrapReading.iso_year >= iso_year_from)
        if iso_year_to:
            stmt = stmt.where(OvitrapReading.iso_year <= iso_year_to)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_city_wide_edi_mean(self) -> list[dict]:
        """
        City-wide weekly mean EDI — replicates MATLAB nanmean(ODI_Mat, 2).
        """
        stmt = (
            select(
                OvitrapReading.iso_year,
                OvitrapReading.iso_week,
                OvitrapReading.week_start,
                func.avg(OvitrapReading.edi).label("mean_edi"),
            )
            .where(OvitrapReading.edi.isnot(None))
            .group_by(OvitrapReading.iso_year, OvitrapReading.iso_week, OvitrapReading.week_start)
            .order_by(OvitrapReading.iso_year, OvitrapReading.iso_week)
        )
        result = await self.db.execute(stmt)
        return [
            {
                "iso_year":   r.iso_year,
                "iso_week":   r.iso_week,
                "week_start": r.week_start,
                "mean_edi":   float(r.mean_edi),
            }
            for r in result.all()
        ]

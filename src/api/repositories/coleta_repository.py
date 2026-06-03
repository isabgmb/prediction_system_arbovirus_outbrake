"""
api/repositories/coleta_repository.py
---------------------------------------
All database access for coleta_semanal — the central fact table.

RN06 is enforced here: incidencia_dengue and densidade_ovos are always
queried and returned separately, never combined.
"""

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import ColetaSemanal, Bairro


class ColetaRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Dengue incidence (RN06 — exposed separately from EDI) ────────────────

    async def get_incidencia_por_bairro(
        self,
        bairro_id: int,
        ano_inicio: int | None = None,
        ano_fim:    int | None = None,
    ) -> list[ColetaSemanal]:
        """Weekly dengue incidence rows for one bairro."""
        stmt = (
            select(ColetaSemanal)
            .where(ColetaSemanal.bairro_fk == bairro_id)
            .order_by(ColetaSemanal.ano, ColetaSemanal.semana_do_ano)
        )
        if ano_inicio:
            stmt = stmt.where(ColetaSemanal.ano >= ano_inicio)
        if ano_fim:
            stmt = stmt.where(ColetaSemanal.ano <= ano_fim)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_media_municipal_dengue(self) -> list[dict]:
        """
        City-wide weekly mean dengue incidence across all bairros.
        Replicates MATLAB:  Dengue = nanmean(DengueOcc, 2)
        Used by the aggregation service to build the LSTM input array.
        """
        stmt = (
            select(
                ColetaSemanal.ano,
                ColetaSemanal.semana_do_ano,
                ColetaSemanal.data_coleta,
                func.avg(ColetaSemanal.incidencia_dengue).label("media_dengue"),
            )
            .where(ColetaSemanal.incidencia_dengue.isnot(None))
            .group_by(
                ColetaSemanal.ano,
                ColetaSemanal.semana_do_ano,
                ColetaSemanal.data_coleta,
            )
            .order_by(ColetaSemanal.ano, ColetaSemanal.semana_do_ano)
        )
        result = await self.db.execute(stmt)
        return [
            {
                "ano":           row.ano,
                "semana_do_ano": row.semana_do_ano,
                "data_coleta":   row.data_coleta,
                "media_dengue":  float(row.media_dengue),
            }
            for row in result.all()
        ]

    # ── Egg density / EDI (RN06 — exposed separately from dengue) ────────────

    async def get_densidade_por_bairro(
        self,
        bairro_id: int,
        ano_inicio: int | None = None,
        ano_fim:    int | None = None,
    ) -> list[ColetaSemanal]:
        """Weekly egg density rows for one bairro."""
        stmt = (
            select(ColetaSemanal)
            .where(ColetaSemanal.bairro_fk == bairro_id)
            .order_by(ColetaSemanal.ano, ColetaSemanal.semana_do_ano)
        )
        if ano_inicio:
            stmt = stmt.where(ColetaSemanal.ano >= ano_inicio)
        if ano_fim:
            stmt = stmt.where(ColetaSemanal.ano <= ano_fim)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_media_municipal_edi(self) -> list[dict]:
        """
        City-wide weekly mean EDI across all bairros.
        Replicates MATLAB:  EDI = nanmean(ODI_Mat, 2)
        """
        stmt = (
            select(
                ColetaSemanal.ano,
                ColetaSemanal.semana_do_ano,
                ColetaSemanal.data_coleta,
                func.avg(ColetaSemanal.densidade_ovos).label("media_edi"),
            )
            .where(ColetaSemanal.densidade_ovos.isnot(None))
            .group_by(
                ColetaSemanal.ano,
                ColetaSemanal.semana_do_ano,
                ColetaSemanal.data_coleta,
            )
            .order_by(ColetaSemanal.ano, ColetaSemanal.semana_do_ano)
        )
        result = await self.db.execute(stmt)
        return [
            {
                "ano":           row.ano,
                "semana_do_ano": row.semana_do_ano,
                "data_coleta":   row.data_coleta,
                "media_edi":     float(row.media_edi),
            }
            for row in result.all()
        ]

    async def count_semanas_disponiveis(self, bairro_id: int) -> int:
        """Week count for a bairro — used by ValidationService (RN: min 6 weeks)."""
        stmt = (
            select(func.count())
            .select_from(ColetaSemanal)
            .where(ColetaSemanal.bairro_fk == bairro_id)
            .where(ColetaSemanal.incidencia_dengue.isnot(None))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one()

    async def get_ultima_semana(self) -> dict | None:
        """Return the most recent (ano, semana_do_ano) available."""
        stmt = (
            select(
                ColetaSemanal.ano,
                ColetaSemanal.semana_do_ano,
                ColetaSemanal.data_coleta,
            )
            .order_by(ColetaSemanal.ano.desc(), ColetaSemanal.semana_do_ano.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        row = result.one_or_none()
        if row is None:
            return None
        return {"ano": row.ano, "semana_do_ano": row.semana_do_ano, "data_coleta": row.data_coleta}

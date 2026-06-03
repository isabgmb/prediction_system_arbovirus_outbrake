"""
api/repositories/bairro_repository.py
---------------------------------------
Bairro, Municipio, Estado lookups + prediction cache access.
"""

from datetime import date
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import Bairro, Municipio, Estado, PrevisaoModelo


class BairroRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(self) -> list[Bairro]:
        result = await self.db.execute(
            select(Bairro).where(Bairro.status_ativo == True).order_by(Bairro.nome)
        )
        return result.scalars().all()

    async def get_by_id(self, bairro_id: int) -> Bairro | None:
        result = await self.db.execute(
            select(Bairro).where(Bairro.id == bairro_id)
        )
        return result.scalar_one_or_none()

    async def get_with_municipio(self, bairro_id: int) -> tuple[Bairro, Municipio] | None:
        """Returns bairro + its municipio in one query (needed for MapBubble)."""
        stmt = (
            select(Bairro, Municipio)
            .join(Municipio, Bairro.municipio_fk == Municipio.id)
            .where(Bairro.id == bairro_id)
        )
        result = await self.db.execute(stmt)
        row = result.one_or_none()
        return (row[0], row[1]) if row else None


class PrevisaoRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_cache(
        self,
        bairro_id:     int,
        versao_modelo: str,
    ) -> list[PrevisaoModelo]:
        result = await self.db.execute(
            select(PrevisaoModelo)
            .where(PrevisaoModelo.bairro_fk == bairro_id)
            .where(PrevisaoModelo.versao_modelo == versao_modelo)
            .order_by(PrevisaoModelo.ano, PrevisaoModelo.semana_do_ano)
        )
        return result.scalars().all()

    async def salvar_previsoes(self, previsoes: list[PrevisaoModelo]) -> None:
        if not previsoes:
            return
        bairro_id     = previsoes[0].bairro_fk
        versao_modelo = previsoes[0].versao_modelo

        await self.db.execute(
            delete(PrevisaoModelo)
            .where(PrevisaoModelo.bairro_fk == bairro_id)
            .where(PrevisaoModelo.versao_modelo == versao_modelo)
        )
        self.db.add_all(previsoes)
        await self.db.flush()

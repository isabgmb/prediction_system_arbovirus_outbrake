"""
Integridade das séries lidas do banco: semanas faltantes mantêm a posição,
e as mensagens de erro dizem quais semanas faltam.
"""
import asyncio

import numpy as np
import pytest

from src.api.models.db_models import DengueCase, District, Neighborhood, OvitrapReading
from src.api.repositories.week_repository import WeekRepository
from src.core.database import AsyncSessionLocal, Base, engine
from src.ml import data
from src.ml.preprocessing import MissingDataError, log2_transform

SEMANAS = [(2019, w) for w in range(1, 6)]


async def _banco(casos: dict, edi: dict):
    """casos/edi: {(ano, semana): [valor por bairro]}; semanas fora do dict não têm registro."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as s:
        d = District(id=1, name="Sul")
        s.add_all([d, Neighborhood(id=1, code="001", name="A", district=d),
                   Neighborhood(id=2, code="002", name="B", district=d)])
        semanas = await WeekRepository(s).ensure(SEMANAS)   # todas existem em epi_week
        for w, valores in casos.items():
            for nid, v in enumerate(valores, start=1):
                s.add(DengueCase(neighborhood_id=nid, epi_week_id=semanas[w].id, case_count=v))
        for w, valores in edi.items():
            for nid, v in enumerate(valores, start=1):
                s.add(OvitrapReading(neighborhood_id=nid, epi_week_id=semanas[w].id, edi=v))
        await s.commit()


async def _serie():
    async with AsyncSessionLocal() as s:
        return await data.load_from_db(s)


def test_semana_sem_casos_fica_na_posicao_certa():
    casos = {w: [2, 4] for w in SEMANAS if w != (2019, 3)}
    edi = {w: [10.0, 20.0] for w in SEMANAS if w not in [(2019, 2), (2019, 5)]}
    asyncio.run(_banco(casos, edi))
    serie = asyncio.run(_serie())

    # Antes, a semana 3 sumia e a 4 ocupava a posição dela.
    assert serie.weeks == SEMANAS
    assert np.isnan(serie.dengue[2]) and serie.dengue[3] == 3.0
    assert np.isnan(serie.edi[1]) and np.isnan(serie.edi[4]) and serie.edi[3] == 15.0

    lacunas, ultima = data.missing_weeks(serie.edi, serie.weeks)
    assert lacunas == [(2019, 2)] and ultima == (2019, 4)


def test_mensagem_de_erro_diz_qual_semana_falta():
    serie = asyncio.run(_serie())
    with pytest.raises(MissingDataError, match=r"edi has no value for 1 week\(s\) \(2019-02\)"):
        log2_transform(serie.edi[:4], "edi", serie.weeks[:4])


# ── Previsão: o histórico precisa estar completo ─────────────────────────────

from pathlib import Path

from fastapi import HTTPException

from src.api.services.forecast_service import ForecastService
from src.ml.inference import RunInfo


class _Registro:
    """Registro de modelos falso: a verificação do histórico acontece antes de carregar modelos."""
    def __init__(self, input_type, lag):
        self.run = RunInfo(1, "teste", input_type, lag, 1, Path("."))

    def get(self, input_type, lag):
        return self.run


async def _prever(input_type, lag):
    async with AsyncSessionLocal() as s:
        return await ForecastService(s, models=_Registro(input_type, lag)).for_area("city", None, input_type, lag, None, None)


def test_previsao_com_semana_sem_edi_no_historico_retorna_422_com_a_semana():
    casos = {w: [2, 4] for w in SEMANAS}
    edi = {w: [10.0, 20.0] for w in SEMANAS if w != (2019, 2)}
    asyncio.run(_banco(casos, edi))
    with pytest.raises(HTTPException) as erro:
        asyncio.run(_prever("edi", 1))
    assert erro.value.status_code == 422
    assert "Cannot forecast from 2019-05: no EDI value in 2019-02" in erro.value.detail


def test_previsao_de_dengue_com_semana_parcial_no_historico_retorna_422():
    casos = {w: ([2] if w == (2019, 2) else [2, 4]) for w in SEMANAS}   # semana 2: só um dos dois bairros
    edi = {w: [10.0, 20.0] for w in SEMANAS}
    asyncio.run(_banco(casos, edi))
    with pytest.raises(HTTPException) as erro:
        asyncio.run(_prever("dengue", 1))
    assert erro.value.status_code == 422
    assert "only some neighborhoods reported cases in 2019-02" in erro.value.detail


# ── Ingestão: avisa sobre lacunas e semanas parciais ─────────────────────────

from src.ingestion import CsvCasesSource, ingest


async def _ingerir(caminho):
    async with AsyncSessionLocal() as s, s.begin():
        return await ingest(s, CsvCasesSource(caminho, "teste"))


def test_ingestao_aponta_lacuna_e_semana_parcial(tmp_path):
    asyncio.run(_banco({w: [2, 4] for w in SEMANAS}, {w: [10.0, 20.0] for w in SEMANAS}))
    csv = tmp_path / "casos.csv"
    # Semana 6 não vem; a 7 vem completa; a 8 só com o bairro 001.
    csv.write_text("neighborhood,epi_year,epi_week,cases\n001,2019,7,3\n002,2019,7,5\n001,2019,8,1\n")
    h = asyncio.run(_ingerir(csv)).health

    assert h.first_week == (2019, 1) and h.last_week == (2019, 8)
    assert h.case_gaps == [(2019, 6)]
    assert h.partial_case_weeks == [((2019, 8), 1)]
    assert h.edi_gaps == [] and h.last_edi_week == (2019, 5)     # EDI que ainda não chegou não é lacuna
    avisos = " ".join(h.warnings())
    assert "No case records in 2019-06" in avisos
    assert "2019-08 (1 of 2)" in avisos


def test_ingestao_sem_problemas_nao_gera_avisos(tmp_path):
    asyncio.run(_banco({w: [2, 4] for w in SEMANAS}, {w: [10.0, 20.0] for w in SEMANAS}))
    csv = tmp_path / "casos.csv"
    csv.write_text("neighborhood,epi_year,epi_week,cases\n001,2019,6,0\n002,2019,6,2\n")
    assert asyncio.run(_ingerir(csv)).health.warnings() == []

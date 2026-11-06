"""Testes unitários da verificação do histórico antes de prever (forecast_service.history_problems)."""
import numpy as np

from src.api.services.forecast_service import history_problems

SEMANAS = [(2019, w) for w in range(1, 7)]
TODOS = [36] * 6


def test_historico_completo_nao_tem_problemas():
    assert history_problems(SEMANAS, np.ones(6), TODOS, 36, "edi") == [""] * 6


def test_semana_sem_valor_bloqueia_so_as_semanas_seguintes():
    x = np.array([1, 1, np.nan, 1, 1, 1], dtype=float)
    out = history_problems(SEMANAS, x, TODOS, 36, "edi")
    assert out[:3] == ["", "", ""]                       # até a própria semana 3, o histórico anterior está ok
    assert out[3] == out[5] == "no EDI value in 2019-03"


def test_semana_parcial_bloqueia_modelos_de_dengue():
    reporting = [36, 36, 30, 36, 36, 36]
    out = history_problems(SEMANAS, np.ones(6), reporting, 36, "dengue")
    assert out[3] == "only some neighborhoods reported cases in 2019-03"


def test_semana_parcial_nao_bloqueia_modelos_de_edi():
    # O artigo usa a média de EDI mesmo com bairros faltando; os casos parciais não entram no modelo de EDI.
    reporting = [36, 36, 30, 36, 36, 36]
    assert history_problems(SEMANAS, np.ones(6), reporting, 36, "edi") == [""] * 6


def test_semana_sem_registro_conta_como_sem_valor_e_nao_como_parcial():
    x = np.array([1, np.nan, 1, 1, 1, 1], dtype=float)
    reporting = [36, 0, 36, 36, 36, 36]
    out = history_problems(SEMANAS, x, reporting, 36, "dengue")
    assert out[2] == "no DENGUE value in 2019-02"

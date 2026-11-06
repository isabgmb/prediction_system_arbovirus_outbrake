"""Testes unitários da sequência de semanas e das lacunas (src/ml/data.py)."""
import numpy as np

from src.core import epiweek
from src.ml import data


def test_semanas_contiguas_atravessam_o_ano():
    assert data.contiguous_weeks((2019, 51), (2020, 2)) == [(2019, 51), (2019, 52), (2020, 1), (2020, 2)]


def test_semanas_contiguas_incluem_a_semana_53():
    ano = next(a for a in range(2010, 2030) if epiweek.weeks_in_year(a) == 53)
    semanas = data.contiguous_weeks((ano, 52), (ano + 1, 1))
    assert semanas == [(ano, 52), (ano, 53), (ano + 1, 1)]


def test_lacunas_internas_e_ultimo_valor():
    semanas = [(2019, w) for w in range(1, 8)]
    valores = np.array([np.nan, 1.0, np.nan, 2.0, 3.0, np.nan, np.nan])
    lacunas, ultima = data.missing_weeks(valores, semanas)
    # A semana 1 vem antes do primeiro valor e as 6 e 7 depois do último: não são lacunas.
    assert lacunas == [(2019, 3)]
    assert ultima == (2019, 5)


def test_serie_sem_valores():
    assert data.missing_weeks(np.array([np.nan, np.nan]), [(2019, 1), (2019, 2)]) == ([], None)


def test_descricao_das_semanas_e_resumida():
    semanas = [(2019, w) for w in range(1, 10)]
    assert data.describe_weeks(semanas[:2]) == "2019-01, 2019-02"
    assert data.describe_weeks(semanas, limit=3) == "2019-01, 2019-02, 2019-03 and 6 more"

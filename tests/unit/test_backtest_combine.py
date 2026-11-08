"""Testes unitários da combinação LSTM + persistência (src/ml/backtest_combine.py)."""
import numpy as np
import pytest

from src.ml import backtest as bt
from src.ml import backtest_combine as bc


def _res():
    return {"name": "edi_lag6", "horizon": 4, "targets": np.arange(5),
            "observed": np.array([5.0, 6.0, 8.0, 7.0, 4.0]),
            "forecast": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            "persistence": np.array([4.0, 5.0, 6.0, 8.0, 7.0])}


def test_pesos_extremos_sao_lstm_e_persistencia():
    res = _res()
    assert np.allclose(bc.combine(res, 1.0)["forecast"], res["forecast"])
    assert np.allclose(bc.combine(res, 0.0)["forecast"], res["persistence"])


def test_peso_meio_e_a_media_e_nao_altera_o_original():
    res = _res()
    c = bc.combine(res, 0.5)
    assert np.allclose(c["forecast"], [3.0, 3.5, 4.0, 5.0, 4.5])
    assert np.allclose(res["forecast"], 2.0)                   # o resultado original continua igual
    assert c["persistence"] is res["persistence"]


def test_peso_fora_do_intervalo_e_rejeitado():
    with pytest.raises(ValueError):
        bc.combine(_res(), 1.5)


def test_combinacao_com_peso_zero_tem_skill_zero():
    res = _res()
    c = bc.combine(res, 0.0)
    assert bt.skill_score(c["observed"], c["forecast"], c["persistence"]) == pytest.approx(0.0)


def _temporada(ic_inferior, pico):
    return {"skill_ci95": [ic_inferior, ic_inferior + 0.5], "peak": {"relative": pico}}


def test_criterio_exige_as_duas_temporadas():
    ok = _temporada(0.1, -0.2)
    assert bc.meets_criterion({"horizon": 4, "2018": ok, "2019": ok})
    assert not bc.meets_criterion({"horizon": 4, "2019": ok})                       # falta 2018
    assert not bc.meets_criterion({"horizon": 4, "2018": ok, "2019": _temporada(-0.01, 0.0)})
    assert not bc.meets_criterion({"horizon": 4, "2018": ok, "2019": _temporada(0.2, -0.26)})
    assert bc.meets_criterion({"horizon": 4, "2018": ok, "2019": _temporada(0.2, 0.25)})   # limite incluído


class _Serie:
    """Série falsa com 208 semanas (2016-01 a 2019-52), sem ler arquivos."""
    def __init__(self):
        self.weeks = [(2016 + i // 52, i % 52 + 1) for i in range(208)]
        self.dengue = 2.0 + np.sin(np.arange(208) / 8.0)
    def series(self, input_type):
        return self.dengue


def test_le_as_previsoes_salvas_sem_retreinar(tmp_path, monkeypatch):
    serie = _Serie()
    origens = bt.default_origins(serie.weeks, (2018, 1), 26, 1)
    meta = {"origins": [bt._label(serie.weeks[o]) for o in origens]}
    assert bc.origins_from_run(meta, serie.weeks) == origens

    n_reps, lag = 3, 1
    for o in origens:     # previsões falsas: log2 = 0, ou seja, 1 caso em toda semana
        np.save(bc.prediction_path(tmp_path, "dengue_lag1", o), np.zeros((n_reps, 208 - lag)))

    def nao_treinar(*args, **kwargs):
        raise AssertionError("o backtest tentou treinar: o nome dos arquivos não confere com o de run_config")
    monkeypatch.setattr(bt, "predict_origin", nao_treinar)

    assert bc.available_configs(tmp_path, origens) == [("dengue", 1)]
    res = bt.run_config(serie, "dengue", lag, origens, n_reps, 1, tmp_path)
    assert np.allclose(res["forecast"], 1.0)


def test_configuracao_incompleta_e_recusada(tmp_path):
    origens = [104, 130, 156, 182]
    np.save(bc.prediction_path(tmp_path, "edi_lag6", 104), np.zeros((2, 10)))
    with pytest.raises(SystemExit):
        bc.available_configs(tmp_path, origens)

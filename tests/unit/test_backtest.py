"""Testes unitários das métricas e da atribuição de modelos do backtest (src/ml/backtest.py)."""
import numpy as np
import pytest

from src.ml import backtest as bt


def test_horizonte_e_a_entrada_mais_recente():
    assert [bt.horizon(lag) for lag in (1, 3, 4, 5, 6)] == [1, 1, 2, 3, 4]


def test_modelo_so_e_usado_depois_do_corte():
    # Modelos treinados com dados até as semanas 103 e 129; horizonte de 4 semanas.
    cutoffs = np.array([103, 129])
    targets = np.array([106, 107, 132, 133, 150])
    # 106 seria emitida na semana 102, antes do primeiro corte: nenhum modelo.
    assert bt.assign_models(targets, cutoffs, 4).tolist() == [-1, 0, 0, 1, 1]


def test_atribuicao_nunca_vaza_dados_futuros():
    rng = np.random.default_rng(0)
    for _ in range(50):
        cutoffs = np.sort(rng.choice(np.arange(50, 200), 5, replace=False))
        h = int(rng.integers(1, 5))
        targets = np.arange(cutoffs[0] + h, 210)
        which = bt.assign_models(targets, cutoffs, h)
        assert (which >= 0).all()
        assert (cutoffs[which] <= targets - h).all()          # só dados vistos até a emissão
        nxt = np.minimum(which + 1, len(cutoffs) - 1)          # e o modelo mais novo permitido
        assert ((which == len(cutoffs) - 1) | (cutoffs[nxt] > targets - h)).all()


def test_skill_score():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    ref = np.array([0.0, 1.0, 2.0, 3.0])
    assert bt.skill_score(y, y, ref) == pytest.approx(1.0)
    assert bt.skill_score(y, ref, ref) == pytest.approx(0.0)
    assert bt.skill_score(y, y - 2, ref) == pytest.approx(-3.0)   # MSE 4 contra 1


def test_bootstrap_reprodutivel_e_contem_a_estimativa():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=80)) + 20
    ref = np.r_[y[0], y[:-1]]
    pred = y + rng.normal(scale=0.5, size=80)
    a = bt.block_bootstrap_ci(y, pred, ref, seed=3)
    assert a == bt.block_bootstrap_ci(y, pred, ref, seed=3)
    assert a[0] <= bt.skill_score(y, pred, ref) <= a[1]


def test_bootstrap_de_previsao_igual_a_referencia_e_zero():
    y = np.arange(30, dtype=float)
    ref = y + 1
    assert bt.block_bootstrap_ci(y, ref, ref) == (0.0, 0.0)


def test_erro_no_pico():
    pk = bt.peak_error(np.array([1.0, 5.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0, 1.0]))
    assert pk.observed == 5.0 and pk.predicted == 3.0
    assert pk.relative == pytest.approx(-0.4)
    assert pk.timing_weeks == 1


def test_origens():
    weeks = [(2016 + i // 52, i % 52 + 1) for i in range(208)]
    assert bt.default_origins(weeks, (2018, 1), 26, 4) == [104, 130, 156, 182]


def test_sementes_do_backtest_unicas_e_fora_das_do_treino():
    seeds = {bt.origin_seed(1, ci, o, rep) for ci in range(10) for o in (104, 130, 156, 182) for rep in range(30)}
    assert len(seeds) == 10 * 4 * 30
    assert min(seeds) > 109_029          # maior semente de train.config_seed com base 1

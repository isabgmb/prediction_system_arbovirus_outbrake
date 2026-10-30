# Arquitetura

## Fluxo dos dados

```
data/raw/ (arquivos do artigo)        CSV de novas semanas
        │                                     │
        ▼                                     ▼
 python -m src.etl.migrate           python -m src.ingestion
        │                                     │
        └──────────────► banco de dados ◄─────┘
                         (SQLite ou PostgreSQL)
                                │
          ┌─────────────────────┼──────────────────────┐
          ▼                     ▼                      ▼
 python -m src.ml.train    API (FastAPI)  ◄──  modelos salvos
 treina, salva em          histórico,          em data/models/
 data/models/ e            previsões,          (encontrados pela
 registra em model_run     dashboard           tabela model_run)
```

1. **ETL** (`src/etl/migrate.py`) lê os arquivos do artigo e o arquivo de referência dos
   bairros e preenche o banco. Pode ser rodado de novo sem duplicar nada.
2. **Ingestão** (`src/ingestion/`) acrescenta ou corrige semanas a partir de outras fontes
   (hoje, CSV). Veja [INGESTION.md](INGESTION.md).
3. **Treino** (`src/ml/train.py`) roda fora da API, salva cada repetição do modelo e registra
   a configuração na tabela `model_run`.
4. **API** (`src/main.py`) serve histórico e previsões. Na inicialização encontra os modelos
   registrados; as previsões rodam os modelos salvos sobre os dados observados até a semana
   pedida e ficam guardadas na tabela `forecast`.

## Estrutura de pastas

```
├── data/
│   ├── raw/                  arquivos do artigo (fora do git)
│   ├── reference/            neighborhoods.csv: bairros entre as fontes (versionado)
│   ├── models/               modelos treinados (fora do git)
│   └── processed/            reservado
├── docs/                     esta documentação
├── migrations/               migrations do Alembic (histórico do esquema)
├── notebooks/                notebook de avaliação
├── outputs/                  figuras e relatórios gerados (fora do git)
├── scripts/                  geração do arquivo de referência (dependências próprias)
└── src/
    ├── main.py               aplicação FastAPI
    ├── cli.py                gestão de contas pela linha de comando
    ├── core/
    │   ├── config.py         configurações lidas do .env
    │   ├── database.py       conexão assíncrona (SQLite e PostgreSQL)
    │   ├── epiweek.py        semanas epidemiológicas brasileiras
    │   ├── names.py          comparação de nomes de bairros entre fontes
    │   └── security.py       senhas (bcrypt) e tokens (JWT)
    ├── api/
    │   ├── controllers/      rotas (areas, auth, dashboard, resultados e as obsoletas)
    │   ├── services/         regras de negócio (áreas, previsões, dashboard, validação)
    │   ├── repositories/     consultas ao banco
    │   └── models/
    │       ├── db_models.py  tabelas (ORM)
    │       └── schemas.py    formatos das respostas (Pydantic)
    ├── ml/
    │   ├── model.py          a rede LSTM, como no MATLAB
    │   ├── preprocessing.py  janelas de lag, divisão treino/teste, log2
    │   ├── data.py           série da cidade a partir dos arquivos ou do banco
    │   ├── train.py          treino e registro dos modelos
    │   ├── inference.py      previsões a partir dos modelos salvos
    │   ├── evaluate.py       comparação com o artigo e figuras
    │   └── utils.py          métricas (RMSE, r) e leitura dos arquivos
    ├── etl/
    │   └── migrate.py        carga dos arquivos do artigo
    └── ingestion/            carga de novas semanas (interface de fontes + CSV)
```

## Banco de dados

Gerenciado pelo Alembic (`migrations/`).

| Tabela | Conteúdo |
|---|---|
| `data_source` | De onde veio cada conjunto de dados: arquivo ou URL, licença, citação |
| `district` | Distritos sanitários (Norte I, Norte II, Sul, Leste, Oeste) |
| `neighborhood` | Os 36 bairros de Natal: código oficial, distrito, coordenadas e sua fonte |
| `socioeconomic` | Censo 2010 por bairro: população, renda total e per capita |
| `epi_week` | Calendário de semanas epidemiológicas |
| `dengue_case` | Casos notificados por bairro, semana e doença |
| `ovitrap_reading` | OPI e EDI por bairro e semana |
| `users` | Contas: nome, senha (hash), papel, ativa ou não |
| `model_run` | Uma configuração treinada: período de treino, métricas, onde estão os modelos |
| `forecast` | Previsões guardadas por modelo, área, semana `as_of` e semana prevista |

`dengue_case` e `ovitrap_reading` são consultadas separadamente: vêm de sistemas de medição
diferentes e nunca são combinadas numa mesma linha.

## O modelo

Uma rede LSTM 1D com 100 unidades, reproduzida do código MATLAB do artigo:

- **Entrada:** casos de dengue ou EDI das semanas anteriores (janela definida pelo lag),
  em escala log2. **Saída:** casos da semana seguinte, como média por bairro da cidade.
- **Treino como no MATLAB:** uma única sequência contínua, 250 épocas, taxa de aprendizado
  0,005 caindo para 0,001 na época 125, gradiente limitado a 1, regularização L2 de 1e-4.
- **30 repetições** com sementes diferentes; a previsão é a média delas.
- **Divisão treino/teste do artigo:** 2016-01 a 2019-10 para treino, 2019-11 a 2019-52 para teste.

Detalhes, resultados e diferenças em relação ao artigo: [REPRODUCTION.md](REPRODUCTION.md).

## Correspondência MATLAB → Python

| MATLAB | Python |
|---|---|
| `Run1DLSTM_01.m` | `src/ml/train.py` |
| `TrainLSTM` (dentro de `Run1DLSTM_01.m`) | `src/ml/model.py` |
| `predictAndUpdateState` | `src/ml/model.predict_sequence()` e `src/ml/inference.py` |
| `ModelsPerform_1DLSTM_01.m` | `src/ml/evaluate.py` |
| `ComputeRMSE` | `src/ml/utils.compute_rmse()` |
| `ComputeR` | `src/ml/utils.compute_r()` |

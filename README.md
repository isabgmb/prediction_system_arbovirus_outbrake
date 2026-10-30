# SIPSA — Backend de previsão de dengue

Backend do **SIPSA** (Sistema Inteligente de Previsão de Surtos de Arboviroses) para Natal, RN.

Prevê casos semanais de dengue com uma rede LSTM 1D, reproduzida em Python a partir do código
MATLAB do artigo:

> *Data-Driven Computational Intelligence Applied to Dengue Outbreak Forecasting: a case study
> at the scale of the city of Natal, RN-Brazil*
> ([repositório do artigo](https://github.com/isabgmb/dengueknowledgePaper_pi))

O que o sistema oferece, por uma API REST (FastAPI):

- **Histórico** de casos e de ovitrampas (OPI e EDI) por cidade, distrito sanitário e bairro.
- **Previsões** de 1 a 4 semanas à frente, a partir de modelos já treinados, com nível de
  risco relativo ao histórico de cada área.
- **Replay**: qualquer semana passada pode ser consultada como o sistema a veria naquele
  momento, com a previsão ao lado do que de fato aconteceu.
- **Ingestão** de novas semanas de dados a partir de CSV.

Os dados de pesquisa cobrem 2016 a 2019 (semanas epidemiológicas). Ainda não há fonte de dados
ao vivo.

## Início rápido

Requer Python 3.10+ e os arquivos de dados do artigo em `data/raw/`
(lista em [docs/SETUP.md](docs/SETUP.md#arquivos-de-dados)).

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                # e preencha SECRET_KEY (veja docs/SETUP.md)

alembic upgrade head                # cria o banco (SQLite por padrão)
python -m src.etl.migrate           # carrega os dados do artigo
python -m src.cli create-user admin --role admin
python -m src.ml.train --dev        # treina e registra os modelos (~8 min; completo: ~80 min)

uvicorn src.main:app --port 8000
```

Documentação interativa em http://localhost:8000/docs (use **Authorize** para fazer login).

## Principais rotas

Todas exigem token (`Authorization: Bearer ...`), exceto login, `/health` e `/resultados`.

| Rota | O que retorna |
|---|---|
| `POST /auth/token` | Login; devolve o token |
| `GET /bairros`, `GET /distritos` | Os 36 bairros e os 5 distritos sanitários |
| `GET /cidade/casos` · `/distritos/{id}/casos` · `/bairros/{id}/casos` | Casos semanais da área |
| `GET /cidade/ovitrampas` · `/distritos/{id}/ovitrampas` · `/bairros/{id}/ovitrampas` | OPI e EDI semanais da área |
| `GET /cidade/previsao` · `/distritos/{id}/previsao` · `/bairros/{id}/previsao` | Previsão e nível de risco |
| `GET /dashboard` | Painel principal: semana atual, previsão, risco por bairro, alertas |

Todas as rotas de área e o `/dashboard` aceitam `as_of=AAAA-SS` para o replay.
Detalhes em [docs/API.md](docs/API.md).

## Documentação

| Documento | Conteúdo |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Instalação completa, arquivos de dados, SQLite e PostgreSQL, contas, problemas comuns |
| [docs/API.md](docs/API.md) | Todas as rotas, parâmetros, previsões por área, risco, replay e erros |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Estrutura de pastas, fluxo dos dados, banco de dados, correspondência MATLAB → Python |
| [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | Treino do modelo e comparação com o artigo e com baselines ingênuos |
| [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | Origem, licença e citação de cada conjunto de dados |
| [docs/INGESTION.md](docs/INGESTION.md) | Como adicionar novas semanas de dados |

## Situação do modelo

A reprodução do artigo está em [docs/REPRODUCTION.md](docs/REPRODUCTION.md). Em resumo: os
modelos com entrada EDI reproduzem as correlações do artigo (r 0,79–0,88), mas subestimam o
pico de 2019; os modelos só com dengue não reproduzem. Um baseline ingênuo (repetir o valor
observado) ainda é competitivo, e toda melhoria futura deve ser comparada com ele.

## Licença

Dados e modelo do artigo: Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0). Uso comercial não é permitido. Veja [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

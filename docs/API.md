# API

Com a API rodando, a documentação interativa (com todos os campos de cada resposta) fica em
http://localhost:8000/docs.

## Rotas

| Método | Rota | Token | Descrição |
|---|---|---|---|
| `POST` | `/auth/token` | Não | Login (campos de formulário `username`, `password`); devolve o token |
| `GET` | `/auth/me` | Sim | Usuário logado e seu papel |
| `GET` | `/health` | Não | Verificação de que a API está no ar |
| `GET` | `/resultados` | Não | Métricas e previsões pré-calculadas do artigo (arquivo `.mat`) |
| `GET` | `/bairros` | Sim | Os 36 bairros: código, distrito, coordenadas, população |
| `GET` | `/distritos` | Sim | Os 5 distritos sanitários e seus bairros |
| `GET` | `/cidade/casos` | Sim | Casos semanais da cidade |
| `GET` | `/distritos/{id}/casos` | Sim | Casos semanais de um distrito |
| `GET` | `/bairros/{id}/casos` | Sim | Casos semanais de um bairro |
| `GET` | `/cidade/ovitrampas` | Sim | OPI e EDI semanais médios da cidade |
| `GET` | `/distritos/{id}/ovitrampas` | Sim | O mesmo para um distrito |
| `GET` | `/bairros/{id}/ovitrampas` | Sim | O mesmo para um bairro |
| `GET` | `/cidade/previsao` | Sim | Previsão da cidade (saída do modelo) |
| `GET` | `/distritos/{id}/previsao` | Sim | Previsão de um distrito (estimativa por divisão) |
| `GET` | `/bairros/{id}/previsao` | Sim | Previsão de um bairro (estimativa por divisão) |
| `GET` | `/dashboard` | Sim | Painel principal numa só chamada |

**Rotas antigas** (ainda funcionam, marcadas como obsoletas e serão removidas):
`/casos/{id}` → `/bairros/{id}/casos`, `/ovitrampas/{id}` → `/bairros/{id}/ovitrampas`,
`/predicao` → `/cidade/previsao`.

## Autenticação

Faça login em `POST /auth/token` e envie o token recebido em todas as rotas protegidas:

```
Authorization: Bearer <token>
```

O token vale `ACCESS_TOKEN_EXPIRE_MINUTES` minutos (padrão 60). Desativar uma conta invalida
os tokens dela imediatamente. Contas são criadas pela linha de comando
(veja [SETUP.md](SETUP.md#contas-de-usuário)).

## Semanas epidemiológicas

Todos os dados semanais usam **semanas epidemiológicas** brasileiras (domingo a sábado, como
no SINAN). Cada ponto tem `epi_year`, `epi_week` e `week_start` (o domingo). A semana 1 é a
primeira com pelo menos 4 dias em janeiro, então pode começar no fim de dezembro.

Nos parâmetros, uma semana é escrita como `AAAA-SS`, por exemplo `2019-30`.

## Histórico (`…/casos`, `…/ovitrampas`)

Uma área é um conjunto de bairros: a cidade (todos), um distrito ou um bairro.

Parâmetros opcionais: `year_from`, `year_to` (anos epidemiológicos) e `as_of`.

Cada semana de `…/casos` traz:

| Campo | Significado |
|---|---|
| `cases` | Total de casos notificados na área |
| `neighborhoods_reporting` | Quantos bairros têm dado naquela semana |
| `mean_per_neighborhood` | Média por bairro: a unidade do artigo e do modelo |
| `incidence_per_100k` | Casos por 100 mil habitantes (população do Censo 2010) |

Cada semana de `…/ovitrampas` traz `opi` (proporção média de armadilhas positivas, de 0 a 1)
e `edi` (índice médio de densidade de ovos).

## Previsões (`…/previsao`)

Nada é treinado durante uma requisição: as previsões usam os modelos salvos por
`python -m src.ml.train`. A primeira chamada de uma configuração carrega os modelos (alguns
segundos); depois as previsões ficam guardadas e são reutilizadas até os dados mudarem.

### Parâmetros

| Parâmetro | Valores | Padrão |
|---|---|---|
| `input_type` | `dengue` ou `edi` | `edi` |
| `lag` | `1`, `3`, `4`, `5`, `6` (janela de entrada; outros valores dão `422`) | `6` |
| `n_weeks` | semanas à frente, até o máximo do lag | o máximo do lag |
| `as_of` | semana `AAAA-SS` | a última semana completa |

Quantas semanas à frente cada lag alcança, usando só dados já observados:

| Lag | Semanas usadas como entrada | Máximo de semanas à frente |
|---|---|---|
| 1 | i−1 | 1 |
| 3 | i−3 a i−1 | 1 |
| 4 | i−4 a i−2 | 2 |
| 5 | i−5 a i−3 | 3 |
| 6 | i−6 a i−4 | 4 |

### Como cada área é prevista

- **Cidade** (`method: "model"`): a saída do LSTM. O modelo do artigo prevê para a cidade
  inteira (média por bairro), e o valor é convertido para o total da cidade.
- **Distrito e bairro** (`method: "split"`): a previsão da cidade multiplicada pela fração dos
  casos da cidade que aquela área costuma ter, calculada com os dados até a semana `as_of`.
  Supõe que cada área mantém sua proporção habitual, então trate como **estimativa**. As
  previsões dos 36 bairros (e dos 5 distritos) somam exatamente a previsão da cidade.

### Campos de cada ponto

| Campo | Significado |
|---|---|
| `horizon` | Semanas depois da semana `as_of` |
| `predicted_cases` | Casos previstos na área naquela semana |
| `lower_bound`, `upper_bound` | Média ± 1 desvio-padrão entre as repetições do modelo. Mostra o quanto elas discordam; **não** é um intervalo de confiança calibrado |
| `mean_per_neighborhood` | Só na cidade: o valor na unidade do modelo |
| `actual_cases` | Casos que de fato ocorreram, quando a semana já é conhecida |

### Nível de risco

`risk_level` compara a semana de pico da previsão com o **próprio histórico** da área até a
semana `as_of`:

- acima do percentil 50 do histórico: `moderate`;
- acima do percentil 90: `high`;
- menos de 1 caso previsto: sempre `low`.

Os percentis e o mínimo são configuráveis (`RISK_PERCENTILES`, `RISK_MIN_CASES`), e cada
previsão devolve os limites usados em `risk_thresholds`. São valores provisórios até a equipe
de saúde definir os oficiais.

## Replay (`as_of`)

Todas as rotas de área e o `/dashboard` aceitam `as_of=AAAA-SS` e respondem como o sistema
responderia naquela semana:

- o histórico termina na semana `as_of`;
- previsões, frações por área e histórico de risco usam só dados observados até ela;
- cada ponto da previsão traz `actual_cases` quando o resultado já é conhecido, para comparar
  previsão e realidade;
- `in_sample: true` indica que alguma semana prevista estava no período de treino do modelo:
  ele já tinha visto aquele resultado, então a comparação não é um teste justo.

Sem `as_of`, a previsão parte da **última semana completa**: todos os bairros com dados e a
entrada do modelo disponível. Pedir explicitamente uma semana incompleta devolve `422`
explicando o que falta.

## Dashboard

`GET /dashboard` (aceita `input_type`, `lag` e `as_of`) devolve numa chamada:

- a semana `as_of` e o total de casos da cidade nela;
- a previsão da cidade;
- `map`: um ponto por bairro, com coordenadas, casos da semana, previsão da próxima semana,
  previsão de pico e nível de risco;
- `alerts`: um cartão por bairro em risco moderado ou alto, com uma explicação em texto.

## `/resultados`

Não recebe parâmetros nem exige token. Lê `Models_1DLSTM_2022_02_04_EvalResults.mat` e devolve
as métricas dos 10 modelos do artigo e as séries previsto × observado dos dois melhores.
Responde `503` se o arquivo não estiver em `data/raw/`.

## Erros

| Código | Significado |
|---|---|
| `401` | Token ausente, inválido ou expirado |
| `404` | Bairro ou distrito não encontrado |
| `422` | Parâmetro inválido, horizonte além do que o lag permite, falta de dados ou semana `as_of` incompleta |
| `503` | Arquivo `.mat` ausente, ou nenhum modelo treinado para a configuração pedida |

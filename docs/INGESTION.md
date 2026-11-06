# Adding new data

The research data ends in epidemiological week 2019-52. New weeks enter through
`src/ingestion`, which every source shares.

## From a CSV (available now)

A spreadsheet export with one row per neighborhood and week:

| Column | Meaning |
|---|---|
| `neighborhood` | official municipal code (e.g. `007`) or name (accents and spelling variants are matched) |
| `epi_year`, `epi_week` | epidemiological year and week (Sunday–Saturday, as in SINAN) |
| `cases` | notified cases; `0` when there were none, blank if unknown |
| `disease` | optional, default `dengue` |

```bash
python -m src.ingestion cases weekly.csv --source "SMS Natal weekly report" --license "..." --citation "..."
```

What happens:
- New rows are inserted, changed rows updated, identical rows skipped; a second run changes nothing.
- Unknown neighborhoods are skipped and listed.
- The source is recorded in `data_source` and linked from every row it wrote.
- Stored forecasts made as of the earliest changed week or later are deleted (they were
  computed without the new data). Replays of earlier weeks stay valid.

A week only becomes the default forecast week once it is **complete**: all neighborhoods
reported cases and the model's input exists (ovitrap EDI for EDI models). Until then forecasts
use the latest complete week, and asking for the partial week returns `422` with what is missing.

The model reads every week from the start of the series, so the **earlier weeks** matter too.
A forecast returns `422` naming the weeks when, before its as-of week, there is:
- a week with no record at all, or with no EDI for EDI models;
- for dengue models, a week where only some neighborhoods reported cases (EDI weeks with some
  neighborhoods missing are averaged over the others, as in the paper).

After every load the CLI checks the whole stored series and prints a `WARNING` for each of
these, with the weeks, e.g.:

```
Stored data: 2016-01 to 2020-12; EDI up to 2020-08
WARNING: No case records in 2020-03. No forecast can be made from these weeks or later ones until they are loaded.
```

The load is not refused, since a later file may be the one that fills the gap. A neighborhood
with no cases in a week should be sent as `0`: a missing row makes the week partial.

## A live source

Implement `WeeklySource` (`src/ingestion/base.py`): a `name`, `file_or_url`, `license`,
`citation`, and `fetch()` returning `WeeklyCases` and/or `WeeklyOvitrap` records. Then:

```python
async with AsyncSessionLocal() as session, session.begin():
    report = await ingest(session, MySource())
```

Candidates to evaluate: weekly SINAN extracts from the municipal health secretariat, the
municipal ovitrap program, and InfoDengue (city-level series; check coverage for Natal and
whether it has neighborhood detail before relying on it).

The models were trained on 2016–2019. Forecasts for later weeks use the same models; after
several months of new data, retrain (`python -m src.ml.train`) and compare with the previous run
(`python -m src.ml.evaluate --run …`) before switching.

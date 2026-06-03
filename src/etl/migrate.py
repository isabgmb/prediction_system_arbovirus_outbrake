"""
etl/migrate.py
--------------
One-time migration: reads original .xlsx and .mat files into the clean
relational schema (neighborhood → dengue_case / ovitrap_reading).

Run: python -m src.etl.migrate
"""
from __future__ import annotations
import asyncio
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from src.core.database import AsyncSessionLocal, engine, Base
from src.api.models.db_models import Neighborhood, Socioeconomic, DengueCase, OvitrapReading

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"

WEEK_ZERO = date(2016, 1, 4)   # first Monday of the dataset


def _week_date(index: int) -> date:
    return WEEK_ZERO + timedelta(weeks=index)


def _iso(d: date) -> tuple[int, int]:
    iso = d.isocalendar()
    return iso[0], iso[1]


def _load_socio() -> pd.DataFrame:
    df = pd.read_excel(str(DATA_RAW / "DADOS GERAIS_01.xlsx"), sheet_name=1, header=0)
    df.columns = ["name", "income_index", "population"] + list(df.columns[3:])
    df["name"]         = df["name"].astype(str).str.strip()
    df["population"]   = pd.to_numeric(df["population"],   errors="coerce").fillna(0).astype(int)
    df["income_index"] = pd.to_numeric(df["income_index"], errors="coerce")
    return df[["name", "population", "income_index"]]


def _load_dengue() -> tuple[np.ndarray, list[str]]:
    df    = pd.read_excel(str(DATA_RAW / "Dados_Modelagem.xlsx"), sheet_name=3, index_col=0, header=0)
    names = [str(n).strip() for n in df.index.tolist()]
    return df.values.astype(float).T, names   # (208, 36)


def _load_edi() -> np.ndarray:
    mat = sio.loadmat(str(DATA_RAW / "Rearranged_Data.mat"), squeeze_me=True)
    return mat["EggIndice_Agregated"][:, :, 1].T   # (208, 36) EDI


def _load_opi() -> np.ndarray:
    mat = sio.loadmat(str(DATA_RAW / "Rearranged_Data.mat"), squeeze_me=True)
    return mat["EggIndice_Agregated"][:, :, 0].T   # (208, 36) OPI


async def run() -> None:
    print("ETL — starting …")
    missing = [f for f in [
        DATA_RAW / "Dados_Modelagem.xlsx",
        DATA_RAW / "DADOS GERAIS_01.xlsx",
        DATA_RAW / "Rearranged_Data.mat",
    ] if not f.exists()]
    if missing:
        for f in missing: print(f"  ✗ Missing: {f}")
        raise FileNotFoundError("Copy missing files to data/raw/ and retry.")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Tables created")

    dengue_mat, names = _load_dengue()
    edi_mat           = _load_edi()
    opi_mat           = _load_opi()
    socio_df          = _load_socio()
    n_weeks           = dengue_mat.shape[0]

    async with AsyncSessionLocal() as session:
        # ── Neighborhoods ─────────────────────────────────────────────────────
        for name in names:
            session.add(Neighborhood(name=name))
        await session.flush()

        from sqlalchemy import select
        result = await session.execute(select(Neighborhood))
        name_to_id = {n.name: n.id for n in result.scalars().all()}
        print(f"✓ {len(name_to_id)} neighborhoods")

        # ── Socioeconomic ─────────────────────────────────────────────────────
        for _, row in socio_df.iterrows():
            nid = name_to_id.get(row["name"])
            if nid is None: continue
            session.add(Socioeconomic(
                neighborhood_id = nid,
                reference_year  = 2016,
                population      = int(row["population"]),
                income_index    = float(row["income_index"]) if pd.notna(row["income_index"]) else None,
            ))
        await session.flush()
        print("✓ Socioeconomic data")

        # ── Dengue cases ──────────────────────────────────────────────────────
        dengue_rows = []
        for wi in range(n_weeks):
            d = _week_date(wi)
            y, w = _iso(d)
            for ni, name in enumerate(names):
                nid = name_to_id.get(name)
                if nid is None: continue
                val = dengue_mat[wi, ni]
                dengue_rows.append(DengueCase(
                    neighborhood_id = nid,
                    week_start      = d,
                    iso_year        = y,
                    iso_week        = w,
                    case_count      = None if np.isnan(val) else float(val),
                ))
        session.add_all(dengue_rows)
        await session.flush()
        print(f"✓ {len(dengue_rows)} dengue_case rows")

        # ── Ovitrap readings ──────────────────────────────────────────────────
        ovitrap_rows = []
        for wi in range(n_weeks):
            d = _week_date(wi)
            y, w = _iso(d)
            for ni, name in enumerate(names):
                nid = name_to_id.get(name)
                if nid is None: continue
                opi = opi_mat[wi, ni]
                edi = edi_mat[wi, ni]
                ovitrap_rows.append(OvitrapReading(
                    neighborhood_id = nid,
                    week_start      = d,
                    iso_year        = y,
                    iso_week        = w,
                    opi             = None if np.isnan(opi) else float(opi),
                    edi             = None if np.isnan(edi) else float(edi),
                ))
        session.add_all(ovitrap_rows)
        await session.flush()
        print(f"✓ {len(ovitrap_rows)} ovitrap_reading rows")

        await session.commit()

    print("\n✓ Migration complete. DB is now the source of truth.")


if __name__ == "__main__":
    asyncio.run(run())

"""GET /ovitrampas/{neighborhood_id}"""
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.api.models.schemas import OvitrapSeriesOut, OvitrapReadingOut
from src.api.repositories.neighborhood_repository import NeighborhoodRepository
from src.api.repositories.dengue_repository import DengueRepository
from src.api.repositories.ovitrap_repository import OvitrapRepository
from src.api.services.validation_service import ValidationService

router = APIRouter(prefix="/ovitrampas", tags=["ovitrampas"])

@router.get("/{neighborhood_id}", response_model=OvitrapSeriesOut)
async def get_ovitrampas(
    neighborhood_id: int,
    year_from: Optional[int] = Query(default=None),
    year_to:   Optional[int] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    nr = NeighborhoodRepository(db)
    dr = DengueRepository(db)
    or_ = OvitrapRepository(db)
    await ValidationService(nr, dr).assert_exists(neighborhood_id)
    n    = await nr.get_by_id(neighborhood_id)
    rows = await or_.get_series(neighborhood_id, year_from, year_to)
    return OvitrapSeriesOut(
        neighborhood_id   = neighborhood_id,
        neighborhood_name = n.name,
        series=[
            OvitrapReadingOut(
                neighborhood_id = r.neighborhood_id,
                week_start      = r.week_start,
                iso_year        = r.iso_year,
                iso_week        = r.iso_week,
                opi             = float(r.opi) if r.opi is not None else None,
                edi             = float(r.edi) if r.edi is not None else None,
            ) for r in rows
        ],
    )

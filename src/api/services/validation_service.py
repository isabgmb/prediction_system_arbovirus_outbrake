"""
api/services/validation_service.py
Validates requests before any DB query or ML call runs.
"""
from fastapi import HTTPException, status
from src.api.repositories.neighborhood_repository import NeighborhoodRepository
from src.api.repositories.dengue_repository import DengueRepository
from src.core.config import settings


class ValidationService:
    def __init__(self, neighborhood_repo: NeighborhoodRepository, dengue_repo: DengueRepository):
        self.neighborhood_repo = neighborhood_repo
        self.dengue_repo       = dengue_repo

    async def assert_exists(self, neighborhood_id: int) -> None:
        n = await self.neighborhood_repo.get_by_id(neighborhood_id)
        if n is None:
            raise HTTPException(status_code=404, detail=f"Neighborhood {neighborhood_id} not found.")

    async def assert_has_geolocation(self, neighborhood_id: int) -> None:
        n = await self.neighborhood_repo.get_by_id(neighborhood_id)
        if n and (n.latitude is None or n.longitude is None):
            raise HTTPException(
                status_code=422,
                detail=f"Neighborhood {neighborhood_id} has no geolocation. "
                       "Add latitude/longitude before requesting a prediction."
            )

    async def assert_sufficient_history(self, neighborhood_id: int) -> None:
        n_weeks = await self.dengue_repo.count_available_weeks(neighborhood_id)
        if n_weeks < settings.min_history_weeks:
            raise HTTPException(
                status_code=422,
                detail=f"Neighborhood {neighborhood_id} has only {n_weeks} week(s) of data. "
                       f"Minimum required: {settings.min_history_weeks}."
            )

"""
main.py
-------
FastAPI application entry point.

Run with:
    uvicorn src.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.ml.inference import load_weights
from src.api.controllers import predicao, casos, ovitrampas, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model weights into memory once.
    Shutdown: nothing to clean up (TF session ends with process).
    """
    print("Loading model weights …")
    load_weights(settings.model_weights_path)
    print("Model weights loaded. API ready.")
    yield


app = FastAPI(
    title="Dengue Forecasting API",
    description=(
        "1D-LSTM model for dengue incidence forecasting — "
        "Natal, RN-Brazil. "
        "Based on: Data-Driven Computational Intelligence Applied to "
        "Dengue Outbreak Forecasting."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow the Next.js frontend origin ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # add production URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(casos.router)
app.include_router(ovitrampas.router)
app.include_router(predicao.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

"""
main.py
-------
FastAPI application entry point.

Run with:
    uvicorn src.main:app --reload --port 8000

The API starts without model weights. The /predicao endpoint will return
503 if weights are not available. All other endpoints work immediately.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.api.controllers import predicao, casos, ovitrampas, auth, resultados


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try to load weights but do not block startup if they are missing.
    # /predicao will return 503 until weights exist.
    try:
        from src.ml.inference import load_weights
        load_weights(settings.model_weights_path)
        print("✓ Model weights loaded.")
    except FileNotFoundError:
        print("⚠  Model weights not found — /predicao will return 503.")
        print("   To generate weights run:  python -m src.ml.train")
    yield


app = FastAPI(
    title="SIPSA — Dengue Forecasting API",
    description=(
        "1D-LSTM model for dengue incidence forecasting — Natal, RN-Brazil.\n\n"
        "Based on: *Data-Driven Computational Intelligence Applied to "
        "Dengue Outbreak Forecasting.*"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(casos.router)
app.include_router(ovitrampas.router)
app.include_router(predicao.router)
app.include_router(resultados.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
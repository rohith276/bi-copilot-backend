from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .api import datasets, analytics, auth
from .db.session import engine, Base
from .core.config import settings
from .core.logger import get_logger
from . import models  # Ensure models are loaded for Base.metadata.create_all
import traceback

logger = get_logger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="""
    An advanced AI-powered analytics engine that provides:
    * **Automated Data Cleaning** via Pandas
    * **Natural Language Querying** using LLMs
    * **Statistical Forecasting** (Exponential Smoothing)
    * **Predictive Trends** (Linear Regression)
    * **Anomaly Detection** (Z-Score Analysis)
    """,
    version="1.0.0",
    contact={
        "name": "Capstone BI Team",
        "url": "http://localhost:3000",
    },
)

# Build allowed origins from settings
_allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://bicopilot.vercel.app",
    "https://bi-copilot-frontend.vercel.app",
]

if settings.FRONTEND_URL:
    # Support comma-separated list of origins
    extra_origins = [o.strip() for o in settings.FRONTEND_URL.split(",") if o.strip()]
    for origin in extra_origins:
        if origin not in _allowed_origins:
            _allowed_origins.append(origin)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception caught: {str(exc)}")
    logger.error(traceback.format_exc())
    response = JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Our team has been notified."},
    )
    # Ensure CORS headers are present even on errors
    origin = request.headers.get("origin")
    if origin in _allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response



# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router)
app.include_router(analytics.router)
app.include_router(auth.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Business Intelligence Copilot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router

app = FastAPI(
    title="Code Dependency Analyzer",
    description="GNN-powered code structure analysis",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import os
    from pathlib import Path
    backend_path = str(Path(__file__).resolve().parent)
    uvicorn.run(app, host="0.0.0.0", port=8000)
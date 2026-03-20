import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from api.v1.endpoints import admin, evaluation


app = FastAPI(
    title="Question Bank Agent API",
    description="API for Administration and Evaluation AI Agents",
    version="1.0.0"
)

app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["evaluation"])

@app.get("/")
def root():
    return {"message": "Welcome to Question Bank Agent API"}

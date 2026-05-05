from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

load_dotenv()

from src.chains import run_full_validation

app = FastAPI(
    title="Startup Idea Validator",
    description="AI-powered startup validation using LangChain",
    version="1.0.0",
)

_env = Environment(loader=FileSystemLoader("templates"), cache_size=0)
templates = Jinja2Templates(env=_env)


class ValidationRequest(BaseModel):
    idea: str
    problem: str
    solution: str
    target_customer: str
    uvp: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "idea": "An AI-powered meal planning app that reduces food waste",
                "problem": "People throw away 30% of groceries due to poor planning",
                "solution": "Smart weekly meal plans based on what's already in your fridge",
                "target_customer": "Busy families aged 28-45 who care about sustainability",
                "uvp": "Cut grocery bills by 40% while reducing food waste by 70%",
            }
        }
    }



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request,name="index.html")


@app.post("/api/validate")
async def validate_startup(payload: ValidationRequest):
    """
    Run full LangChain startup validation pipeline.
    Executes 5 parallel AI analyses + summary synthesis.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "GROQ_API_KEY":
        raise HTTPException(
            status_code=400,
            detail="GROQ API KEY not configured. Please set it in your .env file."
        )

    try:
        result = await run_full_validation(
            idea=payload.idea,
            problem=payload.problem,
            solution=payload.solution,
            target_customer=payload.target_customer,
            uvp=payload.uvp,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("MODEL_NAME", "lllama-3.3-70b-versatile")}


# entry point for running
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
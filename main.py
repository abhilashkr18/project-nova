"""NOVA AI backend: secure, server-side Gemini API integration.

API credentials are read only from environment variables. Never put them in the
browser bundle, HTML, Git history, or a public issue.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict, deque
from typing import Literal
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()  # Optional local .env support; production secrets should be host environment variables.

APP_NAME = "NOVA AI API"


def clean_api_key(value: str | None) -> str:
    candidate = (value or "").strip()
    if not candidate or "add_the_key" in candidate.lower() or "your_api_key" in candidate.lower() or candidate.lower() in {"changeme", "replace_me", "none", "null"}:
        return ""
    return candidate


GEMINI_API_KEY = clean_api_key(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.5-flash").strip()
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image").strip()
GOOGLE_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models/"

raw_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "https://abhilashkr18.github.io,http://localhost:8000,http://127.0.0.1:5500,http://localhost:5500",
)
allowed_origins = [item.strip().rstrip("/") for item in raw_origins.split(",") if item.strip()]

app = FastAPI(
    title=APP_NAME,
    version="0.1.0",
    description="Server-side AI planning and image generation for the NOVA website builder.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["https://abhilashkr18.github.io"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Modest per-process rate limits to reduce accidental cost spikes. For a public,
# paid product, replace this in-memory limiter with Redis/database-backed limits
# and authenticated user quotas before accepting public traffic.
_requests: dict[tuple[str, str], deque[float]] = defaultdict(deque)
_RATE_WINDOW_SECONDS = 3600
_RATE_LIMITS = {"plan": 12, "image": 3}


def enforce_rate_limit(request: Request, bucket: str) -> None:
    address = request.client.host if request.client else "unknown"
    key = (address, bucket)
    now = time.time()
    queue = _requests[key]
    while queue and now - queue[0] >= _RATE_WINDOW_SECONDS:
        queue.popleft()
    if len(queue) >= _RATE_LIMITS[bucket]:
        raise HTTPException(
            status_code=429,
            detail=f"Request limit reached for {bucket}. Please try again later.",
            headers={"Retry-After": "3600"},
        )
    queue.append(now)


class PlanRequest(BaseModel):
    prompt: str = Field(min_length=8, max_length=6000)


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=5, max_length=1200)
    category: str = Field(default="Architecture", max_length=80)


class BuildPlan(BaseModel):
    template: Literal["studio", "portfolio", "store", "community", "architecture", "technology", "restaurant", "realestate", "wellness"] = "studio"
    palette: Literal["midnight", "light", "ocean", "gold"] = "midnight"
    brand: str = Field(default="NOVA Project", max_length=65)
    eyebrow: str = Field(default="DESIGNED WITH PURPOSE", max_length=80)
    headline: str = Field(default="Ideas made visible.", max_length=180)
    subtitle: str = Field(default="A thoughtful digital experience, made around your idea.", max_length=500)
    cta: str = Field(default="Explore the project", max_length=45)
    card1: str = Field(default="Project overview", max_length=65)
    card2: str = Field(default="Our process", max_length=65)
    card3: str = Field(default="Next steps", max_length=65)
    aboutText: str = Field(default="A clear, useful introduction to the work and the thinking behind it.", max_length=500)
    contactText: str = Field(default="Tell us about your idea and the next step you need.", max_length=200)
    customSections: list[dict[str, str]] = Field(default_factory=list, max_length=8)
    visualPrompts: list[str] = Field(default_factory=list, max_length=5)


def require_api_key() -> str:
    key = clean_api_key(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if not key:
        raise HTTPException(
            status_code=503,
            detail="AI backend is deployed, but no Gemini key is configured. Add GEMINI_API_KEY in your server's secret/environment settings.",
        )
    return key


async def gemini_generate(model: str, payload: dict, api_key: str) -> dict:
    url = GOOGLE_API_ROOT + quote(model, safe="") + ":generateContent"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(100.0, connect=15.0)) as client:
            response = await client.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="The AI provider took too long. Please retry.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Could not connect to the AI provider.") from exc

    if response.status_code >= 400:
        # Do not echo the key or upstream response text; it may include provider detail.
        if response.status_code in (401, 403):
            raise HTTPException(status_code=502, detail="The AI provider rejected the configured key. Check the key type and API access in your server settings.")
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="The AI provider rate limit or quota was reached. Check quota and try later.")
        raise HTTPException(status_code=502, detail=f"The AI provider returned an error (HTTP {response.status_code}). Check the selected model and API access.")
    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="The AI provider returned an invalid response.") from exc


def extract_text(result: dict) -> str:
    try:
        parts = result["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="The AI model returned no usable text.") from exc
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    if not text_parts:
        raise HTTPException(status_code=502, detail="The AI model returned an empty response.")
    return "\n".join(text_parts).strip()


@app.get("/")
async def root() -> dict:
    return {"service": APP_NAME, "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": APP_NAME,
        "provider": "Google Gemini API",
        "configured": bool(GEMINI_API_KEY),
        "textModel": TEXT_MODEL,
        "imageModel": IMAGE_MODEL,
        "capabilities": ["website-planning", "image-generation"],
    }


@app.post("/api/generate-plan", response_model=BuildPlan)
async def generate_plan(body: PlanRequest, request: Request) -> BuildPlan:
    enforce_rate_limit(request, "plan")
    key = require_api_key()
    system_brief = f"""You are NOVA, a senior web creative director and UX architect. Turn the user's prompt into a polished, original website content/design plan. Do not output HTML, JavaScript, CSS, executable code, markdown fences, commentary, fake testimonials, fabricated addresses, fake metrics, or unsupported claims.
Return ONLY valid JSON with these keys:
- template: one of studio, portfolio, store, community, architecture, technology, restaurant, realestate, wellness
- palette: one of midnight, light, ocean, gold
- brand (string max 65 chars)
- eyebrow (max 80)
- headline (max 180)
- subtitle (max 500)
- cta (max 45)
- card1, card2, card3 (each max 65)
- aboutText (max 500)
- contactText (max 200)
- customSections: 3 to 5 objects, each {{"title":"...", "text":"..."}}, title max 90 chars and text max 1500 chars. When the prompt asks for phases/workflow include Phase 1 through Phase 5 clearly.
- visualPrompts: 3 to 5 specific photo-realistic image prompts suitable for a portfolio/website. Cover architecture, cities, malls, parks, villages, agriculture or company campuses where relevant. No text/logos in generated imagery.
Use a coherent style and content substantially more specific than a generic template while remaining truthful. Infer the template from the request, and use user's own names/places only if they supplied them. User prompt: {body.prompt}
"""
    result = await gemini_generate(
        TEXT_MODEL,
        {
            "contents": [{"role": "user", "parts": [{"text": system_brief}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.8},
        },
        key,
    )
    raw = extract_text(result)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        data = json.loads(raw)
        plan = BuildPlan.model_validate(data)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="The AI model did not return a valid website plan. Please retry.") from exc
    # Bound all section strings and discard malformed items before returning to the browser.
    cleaned = []
    for item in plan.customSections:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()[:90]
        text = str(item.get("text", "")).strip()[:1500]
        if title and text:
            cleaned.append({"title": title, "text": text})
    prompts = [str(p).strip()[:1200] for p in plan.visualPrompts if str(p).strip()]
    plan.customSections = cleaned[:8]
    plan.visualPrompts = prompts[:5]
    return plan


@app.post("/api/generate-image")
async def generate_image(body: ImageRequest, request: Request) -> dict:
    enforce_rate_limit(request, "image")
    key = require_api_key()
    prompt = (
        "Create a polished, photo-realistic, high-quality website portfolio image. "
        f"Subject/category: {body.category}. User art direction: {body.prompt}. "
        "Use strong composition, natural believable lighting and rich real-world detail. "
        "No text, letters, watermark, signage, UI, or brand logo. Output one image."
    )
    result = await gemini_generate(
        IMAGE_MODEL,
        {"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
        key,
    )
    try:
        parts = result["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="The image model returned no image. Check that the selected model supports image output.") from exc
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data") if isinstance(part, dict) else None
        if isinstance(inline, dict) and inline.get("data"):
            mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            encoded = inline["data"]
            return {"ok": True, "mimeType": mime, "dataUrl": f"data:{mime};base64,{encoded}", "model": IMAGE_MODEL}
    # Avoid returning internal provider payloads; they could include non-user-facing details.
    raise HTTPException(status_code=502, detail="The image model replied without an image. Check model availability and image-generation access.")

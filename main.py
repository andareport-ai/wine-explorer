import os
import json
import asyncio
import hashlib
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="Cave & Vigne - Wine Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── 인메모리 캐시 ─────────────────────────────────────────────────
cache = {}

def get_anthropic_key():
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()

def get_gemini_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()

def cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()

# ── Prompt (간결하게 최적화) ──────────────────────────────────────
def build_prompt(wine_query: str) -> str:
    return f"""세계 최고 소믈리에로서 "{wine_query}" 와인 정보를 한국어 JSON으로 작성하세요.

순수 JSON만 출력 (마크다운 없이):

{{
  "wine_name": "정확한 와인명",
  "wine_subtitle": "생산지 · 품종 · 분류",
  "producer": "생산자 설명 300자 이상 (가문, 철학, 밭, 대표작)",
  "production": "생산방식 300자 이상 (수확, 발효, 숙성)",
  "vineyard": "밭 설명 300자 이상 (위치, 토양, 주변 밭 비교)",
  "vintage": "빈티지 300자 이상 (해당연도 + 주요빈티지 비교)",
  "tasting": "테이스팅 300자 이상 (노즈, 팔레트, 구조감, 여운)",
  "lore": "설화와 미사어구 (문학적으로)",
  "comparison": "비교 와인 2개와 공통점/차이점"
}}"""

# ── Claude 호출 ───────────────────────────────────────────────────
async def call_claude(client: httpx.AsyncClient, wine_query: str) -> dict:
    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": get_anthropic_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-opus-4-5-20251101",
            "max_tokens": 6000,
            "messages": [{"role": "user", "content": build_prompt(wine_query)}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    return json.loads(raw)

# ── Gemini 호출 ───────────────────────────────────────────────────
async def call_gemini(client: httpx.AsyncClient, wine_query: str) -> dict:
    for model in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]:
        try:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={get_gemini_key()}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": build_prompt(wine_query)}]}],
                    "generationConfig": {"responseMimeType": "application/json"},
                },
                timeout=120,
            )
            if resp.status_code == 200:
                raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0]
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0]
                return json.loads(raw)
        except Exception as e:
            print(f"[gemini/{model}] 실패: {e}")
            continue
    raise Exception("모든 Gemini 모델 실패")

# ── 교차검증 합성 ─────────────────────────────────────────────────
async def synthesize_with_claude(client, wine_query, results):
    if len(results) == 1:
        single = list(results.values())[0]
        source = list(results.keys())[0]
        final = {
            "wine_name": single.get("wine_name", wine_query),
            "wine_subtitle": single.get("wine_subtitle", ""),
            "synthesis_note": f"{source.upper()} 단독 결과",
        }
        for k in ["producer", "production", "vineyard", "vintage", "tasting", "lore", "comparison"]:
            final[k] = {"text": single.get(k, ""), "confidence": 70}
        return final

    synthesis_prompt = f"""와인 편집장으로서 "{wine_query}"에 대한 두 AI 응답을 교차검증해 한국어 JSON으로 통합하세요.

=== Claude ===
{json.dumps(results.get("claude", {}), ensure_ascii=False)}

=== Gemini ===
{json.dumps(results.get("gemini", {}), ensure_ascii=False)}

공통사실 중심으로 통합, confidence(0-100) 포함. 순수 JSON만:

{{
  "wine_name": "...",
  "wine_subtitle": "...",
  "producer": {{"text": "...", "confidence": 85}},
  "production": {{"text": "...", "confidence": 90}},
  "vineyard": {{"text": "...", "confidence": 88}},
  "vintage": {{"text": "...", "confidence": 82}},
  "tasting": {{"text": "...", "confidence": 75}},
  "lore": {{"text": "...", "confidence": 70}},
  "comparison": {{"text": "...", "confidence": 80}},
  "synthesis_note": "두 AI 차이점 한 줄"
}}"""

    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": get_anthropic_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-opus-4-5-20251101",
            "max_tokens": 8000,
            "messages": [{"role": "user", "content": synthesis_prompt}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    return json.loads(raw)

class WineRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("templates/index.html")

@app.post("/api/wine")
async def get_wine_info(req: WineRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="와인 이름을 입력해주세요.")

    ck = cache_key(req.query)
    if ck in cache:
        print(f"[cache] HIT: {req.query}")
        return cache[ck]

    async with httpx.AsyncClient() as client:
        tasks = {
            "claude": call_claude(client, req.query),
            "gemini": call_gemini(client, req.query),
        }
        results = {}
        errors = {}
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for key, result in zip(tasks.keys(), gathered):
            if isinstance(result, Exception):
                errors[key] = str(result)
                print(f"[{key}] ERROR: {type(result).__name__}: {result}")
            else:
                results[key] = result

        if not results:
            raise HTTPException(status_code=502, detail="모든 AI 호출이 실패했습니다.")

        final = await synthesize_with_claude(client, req.query, results)
        final["sources_used"] = list(results.keys())
        final["sources_failed"] = errors

        cache[ck] = final
        return final

@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(cache), "apis": {
        "claude": bool(get_anthropic_key()),
        "gemini": bool(get_gemini_key()),
    }}

@app.get("/test-claude")
async def test_claude():
    key = get_anthropic_key()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-opus-4-5-20251101", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            )
            return {"status_code": resp.status_code, "response": resp.text[:200]}
    except Exception as e:
        return {"error": str(e)}

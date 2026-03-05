import os
import json
import asyncio
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

def get_anthropic_key():
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()

def get_gemini_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()

def build_prompt(wine_query: str) -> str:
    return f"""당신은 세계 최고 수준의 소믈리에이자 와인 전문가입니다.
다음 와인에 대해 7가지 항목을 한국어로 상세하게 작성하세요.

와인: "{wine_query}"

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 순수 JSON만:

{{
  "wine_name": "와인의 정확한 이름",
  "wine_subtitle": "생산지 · 품종 · 분류",
  "producer": "생산자 설명 (가문 역사, 철학, 보유 밭, 대표 와인 등 400자 이상)",
  "production": "생산 방식 설명 (수확, 발효, 숙성, 병입 등 400자 이상)",
  "vineyard": "포도밭 설명 (위치, 토양, 미기후, 주변 밭 비교 400자 이상)",
  "vintage": "빈티지 설명 (해당 연도 조건 + 생산자 주요 빈티지 비교 400자 이상)",
  "tasting": "테이스팅 노트 (노즈, 팔레트, 구조감, 여운 400자 이상)",
  "lore": "설화, 일화, 미사어구 (문학적이고 감성적으로)",
  "comparison": "비교 와인 2-3개와 공통점/차이점 상세 비교"
}}"""

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
            "max_tokens": 8000,
            "messages": [{"role": "user", "content": build_prompt(wine_query)}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"]
    print(f"[claude] raw 앞부분: {raw[:200]!r}")
    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]
    return json.loads(cleaned)

async def call_gemini(client: httpx.AsyncClient, wine_query: str) -> dict:
    resp = await client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={get_gemini_key()}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": build_prompt(wine_query)}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(raw.strip().replace("```json", "").replace("```", ""))

async def synthesize_with_claude(client, wine_query, results):
    # 성공한 AI가 하나면 그냥 그 결과를 바로 합성 프롬프트로 넘김
    available = list(results.keys())

    if len(available) == 1:
        # 하나만 성공한 경우 — Claude 단독 결과에 confidence 추가
        single = results[available[0]]
        final = {
            "wine_name": single.get("wine_name", wine_query),
            "wine_subtitle": single.get("wine_subtitle", ""),
            "synthesis_note": f"{available[0].upper()} 단독 결과 (다른 AI 호출 실패)",
        }
        for key in ["producer", "production", "vineyard", "vintage", "tasting", "lore", "comparison"]:
            final[key] = {"text": single.get(key, ""), "confidence": 70}
        return final

    # 두 개 이상 성공한 경우 — 교차검증 합성
    synthesis_prompt = f"""당신은 와인 전문 편집장입니다. 아래는 동일한 와인 "{wine_query}"에 대해
여러 AI가 독립적으로 생성한 정보입니다.

=== Claude 응답 ===
{json.dumps(results.get("claude", {}), ensure_ascii=False, indent=2)}

=== Gemini 응답 ===
{json.dumps(results.get("gemini", {}), ensure_ascii=False, indent=2)}

당신의 임무:
1. 두 응답에서 공통적으로 언급된 사실(신뢰도 높음)을 중심으로 통합
2. 한 곳에서만 언급된 내용은 신중하게 검토 후 포함 여부 결정
3. 서로 상충되는 정보는 가장 신뢰도 높은 내용 선택
4. 각 섹션에 confidence 점수 추가 (0-100)
5. 최종 정보를 풍부하고 정확하게 한국어로 작성

반드시 아래 JSON 형식으로만 응답:

{{
  "wine_name": "정확한 와인명",
  "wine_subtitle": "생산지 · 품종 · 분류",
  "producer":     {{"text": "...", "confidence": 85}},
  "production":   {{"text": "...", "confidence": 90}},
  "vineyard":     {{"text": "...", "confidence": 88}},
  "vintage":      {{"text": "...", "confidence": 82}},
  "tasting":      {{"text": "...", "confidence": 75}},
  "lore":         {{"text": "...", "confidence": 70}},
  "comparison":   {{"text": "...", "confidence": 80}},
  "synthesis_note": "두 AI 간의 주요 차이점 또는 특이사항 한 줄 요약"
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
    raw = resp.json()["content"][0]["text"]
    return json.loads(raw.strip().replace("```json", "").replace("```", ""))

class WineRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("templates/index.html")

@app.post("/api/wine")
async def get_wine_info(req: WineRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="와인 이름을 입력해주세요.")

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

        # Claude만 성공해도 결과 반환
        if not results:
            raise HTTPException(status_code=502, detail="모든 AI 호출이 실패했습니다.")

        final = await synthesize_with_claude(client, req.query, results)
        final["sources_used"] = list(results.keys())
        final["sources_failed"] = errors

        return final

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "apis": {
            "claude": bool(get_anthropic_key()),
            "gemini": bool(get_gemini_key()),
        }
    }

@app.get("/test-claude")
async def test_claude():
    key = get_anthropic_key()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-opus-4-5-20251101",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            return {"status_code": resp.status_code, "response": resp.text[:300]}
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__}

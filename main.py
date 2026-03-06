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

# ── 파일 기반 영구 캐시 ───────────────────────────────────────────
CACHE_DIR = os.environ.get("CACHE_DIR", ".")
CACHE_FILE = os.path.join(CACHE_DIR, "wine_cache.json")
NAME_INDEX_FILE = os.path.join(CACHE_DIR, "wine_name_index.json")

def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"[cache] {os.path.basename(path)}에서 {len(data)}개 로드됨")
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_json(path: str, data: dict):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"[cache] {os.path.basename(path)} 저장 실패: {e}")

cache = _load_json(CACHE_FILE)
name_index = _load_json(NAME_INDEX_FILE)  # 쿼리 → 정식 와인명 매핑

def get_anthropic_key():
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()

def get_gemini_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()

def cache_key(name: str) -> str:
    return hashlib.md5(name.lower().strip().encode()).hexdigest()

# ── 와인명 정규화 (한글/영문 → 정식 영문명) ─────────────────────
async def normalize_wine_name(client: httpx.AsyncClient, query: str) -> str:
    q_lower = query.lower().strip()
    if q_lower in name_index:
        print(f"[normalize] 인덱스 HIT: {query} → {name_index[q_lower]}")
        return name_index[q_lower]

    try:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": get_anthropic_key(),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": f"이 와인의 정식 영문명만 답하세요. 빈티지가 있으면 포함. 와인명만 출력, 설명 없이: {query}"}],
            },
            timeout=15,
        )
        resp.raise_for_status()
        canonical = resp.json()["content"][0]["text"].strip().strip('"').strip("'")
        print(f"[normalize] {query} → {canonical}")
    except Exception as e:
        print(f"[normalize] 실패, 원본 사용: {e}")
        canonical = query

    name_index[q_lower] = canonical
    _save_json(NAME_INDEX_FILE, name_index)
    return canonical

# ── Prompt (간결하게 최적화) ──────────────────────────────────────
def build_prompt(wine_query: str) -> str:
    return f"""세계 최고 소믈리에로서 "{wine_query}" 와인 정보를 한국어 JSON으로 작성하세요.

각 항목은 아래 형식 규칙을 반드시 따르세요:
- producer: 핵심 사실을 줄바꿈(\n)으로 구분된 짧은 단락으로. 예) 설립연도/역사 → 철학 → 주요 밭 → 대표 와인
- production: 단계별로 번호 목록 형식. 예) 1. 수확: ... \n2. 발효: ... \n3. 숙성: ...
- vineyard: 핵심 정보를 항목별로 줄바꿈(\n) 구분. 예) 위치: ...\n면적: ...\n토양: ...\n미기후: ...\n품종 구성: ...
- vineyard_lat: 포도밭 중심부 위도 (숫자만, 예: 47.1234)
- vineyard_lng: 포도밭 중심부 경도 (숫자만, 예: 7.5678)
- vineyard_comparison: 주변 밭 또는 동급 밭과의 비교를 반드시 표 형식으로. 예) 항목 | 해당와인밭 | 인접밭1 | 인접밭2\n----|----|----|----\n등급 | ... | ... | ...\n토양 | ... | ... | ...\n대표와인 | ... | ... | ...\n평균가격 | ... | ... | ...
- vintage: 해당 빈티지 설명 후, 주요 빈티지 비교표를 텍스트 표로. 예) 빈티지 | 특징 | 평가\n----\n2000 | ... | ★★★★★
- tasting: 노즈/팔레트/구조감/여운을 항목별로 명확히 구분. 예) 🌹 노즈: ...\n👅 팔레트: ...\n⚖️ 구조감: ...\n🌊 여운: ...
- lore: 문학적이고 감성적인 문장으로 자유롭게
- comparison: 비교 와인을 표 형식으로. 예) 항목 | {wine_query} | 비교와인1 | 비교와인2\n----\n스타일 | ... | ... | ...
- pricing: 반드시 750ml 1병 기준 가격만 표기 (6병/박스/케이스 가격 절대 사용 금지). 여러 출처를 교차 참조하여 작성:
  Wine-Searcher 평균가: $XXX USD (750ml 1병)\n
  Vivino 평균가: $XXX USD (750ml 1병, 있다면)\n
  기타 참고가: (wine.com, Total Wine 등 주요 리테일 가격이 있다면)\n
  한국 예상 소비자가: 약 XXX만원 (750ml 1병)\n
  가격 근거: (빈티지, 생산량, 수요 등 가격 결정 요인)\n
  구매 팁: (한국 구매처 유형 또는 팁)

순수 JSON만 출력 (마크다운 코드블록 없이):

{{
  "wine_name": "정확한 와인명",
  "wine_subtitle": "생산지 · 품종 · 분류",
  "producer": "단락 구분된 생산자 설명",
  "production": "번호 목록 형식 생산방식",
  "vineyard": "항목별 밭 설명",
  "vineyard_lat": 47.1234,
  "vineyard_lng": 7.5678,
  "vineyard_comparison": "주변/동급 밭 비교 표",
  "vintage": "빈티지 설명 + 비교표",
  "tasting": "항목별 테이스팅 노트",
  "lore": "문학적 설화와 미사어구",
  "comparison": "비교 와인 표",
  "pricing": "750ml 1병 기준 다중 출처 가격 정보"
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
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 8000,
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
    for model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"]:
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
            print(f"[gemini/{model}] 응답코드: {resp.status_code}, 내용: {resp.text[:200]}")
            if resp.status_code == 200:
                raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0]
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0]
                return json.loads(raw)
        except Exception as e:
            print(f"[gemini/{model}] 예외: {type(e).__name__}: {e}")
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
            "vineyard_lat": single.get("vineyard_lat"),
            "vineyard_lng": single.get("vineyard_lng"),
        }
        for k in ["producer", "production", "vineyard", "vintage", "tasting", "lore", "comparison", "pricing", "vineyard_comparison"]:
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
  "vineyard_lat": 47.1234,
  "vineyard_lng": 7.5678,
  "vineyard_comparison": {{"text": "...", "confidence": 85}},
  "vintage": {{"text": "...", "confidence": 82}},
  "tasting": {{"text": "...", "confidence": 75}},
  "lore": {{"text": "...", "confidence": 70}},
  "comparison": {{"text": "...", "confidence": 80}},
  "pricing": {{"text": "...", "confidence": 75}},
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
            "model": "claude-haiku-4-5-20251001",
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
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[synthesis] JSON 파싱 실패, 첫 번째 소스 사용: {e}")
        # 합성 실패 시 첫 번째 결과를 fallback으로 사용
        single = list(results.values())[0]
        source = list(results.keys())[0]
        final = {
            "wine_name": single.get("wine_name", wine_query),
            "wine_subtitle": single.get("wine_subtitle", ""),
            "synthesis_note": f"합성 실패 — {source.upper()} 결과 기반",
        }
        for k in ["producer", "production", "vineyard", "vintage", "tasting", "lore", "comparison", "pricing"]:
            final[k] = {"text": single.get(k, ""), "confidence": 70}
        return final

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
        # 1. 와인명 정규화 (한글/영문 → 정식 영문명)
        canonical = await normalize_wine_name(client, req.query)
        ck = cache_key(canonical)

        if ck in cache and cache[ck].get("vineyard_lat"):
            print(f"[cache] HIT: {req.query} → {canonical}")
            cached = cache[ck].copy()
            cached["canonical_name"] = canonical
            cached["_cached"] = True
            return cached
        elif ck in cache:
            print(f"[cache] STALE (vineyard_lat 없음): {canonical} → 재검색")
            del cache[ck]

        # 2. 캐시 미스 → Claude + Gemini 병렬 호출
        tasks = {
            "claude": call_claude(client, canonical),
            "gemini": call_gemini(client, canonical),
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

        final = await synthesize_with_claude(client, canonical, results)
        final["sources_used"] = list(results.keys())
        final["sources_failed"] = errors
        final["canonical_name"] = canonical

        cache[ck] = final
        _save_json(CACHE_FILE, cache)
        return final

@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(cache), "apis": {
        "claude": bool(get_anthropic_key()),
        "gemini": bool(get_gemini_key()),
    }}

@app.get("/list-gemini-models")
async def list_gemini_models():
    key = get_gemini_key()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
            )
            data = resp.json()
            models = [m["name"] for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
            return {"available_models": models, "total": len(models)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/test-claude")
async def test_claude():
    key = get_anthropic_key()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            )
            return {"status_code": resp.status_code, "response": resp.text[:200]}
    except Exception as e:
        return {"error": str(e)}

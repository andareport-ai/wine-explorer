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
    return f"""?¹ì‹ ?€ ?¸ê³„ ìµœê³  ?˜ì????Œë¯ˆë¦¬ì—?´ì ?€???„ë¬¸ê°€?…ë‹ˆ??
?¤ìŒ ?€?¸ì— ?€??7ê°€ì§€ ??ª©???œêµ­?´ë¡œ ?ì„¸?˜ê²Œ ?‘ì„±?˜ì„¸??

?€?? "{wine_query}"

ë°˜ë“œ???„ë˜ JSON ?•ì‹?¼ë¡œë§??‘ë‹µ?˜ì„¸?? ?¤ë¥¸ ?ìŠ¤???†ì´ ?œìˆ˜ JSONë§?

{{
  "wine_name": "?€?¸ì˜ ?•í™•???´ë¦„",
  "wine_subtitle": "?ì‚°ì§€ Â· ?ˆì¢… Â· ë¶„ë¥˜",
  "producer": "?ì‚°???¤ëª… (ê°€ë¬???‚¬, ì² í•™, ë³´ìœ  ë°? ?€???€????400???´ìƒ)",
  "production": "?ì‚° ë°©ì‹ ?¤ëª… (?˜í™•, ë°œíš¨, ?™ì„±, ë³‘ì… ??400???´ìƒ)",
  "vineyard": "?¬ë„ë°??¤ëª… (?„ì¹˜, ? ì–‘, ë¯¸ê¸°?? ì£¼ë? ë°?ë¹„êµ 400???´ìƒ)",
  "vintage": "ë¹ˆí‹°ì§€ ?¤ëª… (?´ë‹¹ ?°ë„ ì¡°ê±´ + ?ì‚°??ì£¼ìš” ë¹ˆí‹°ì§€ ë¹„êµ 400???´ìƒ)",
  "tasting": "?Œì´?¤íŒ… ?¸íŠ¸ (?¸ì¦ˆ, ?”ë ˆ?? êµ¬ì¡°ê°? ?¬ìš´ 400???´ìƒ)",
  "lore": "?¤í™”, ?¼í™”, ë¯¸ì‚¬?´êµ¬ (ë¬¸í•™?ì´ê³?ê°ì„±?ìœ¼ë¡?",
  "comparison": "ë¹„êµ ?€??2-3ê°œì? ê³µí†µ??ì°¨ì´???ì„¸ ë¹„êµ"
}}"""

async def call_claude(client: httpx.AsyncClient, wine_query: str) -> dict:
    key = get_anthropic_key()
    print(f"[claude] ????20?? {key[:20]!r}, ê¸¸ì´: {len(key)}")
    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-3-haiku-20240307",
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": build_prompt(wine_query)}],
        },
        timeout=60,
    )
    print(f"[claude] ?‘ë‹µì½”ë“œ: {resp.status_code}, ?´ìš©?ë?ë¶? {resp.text[:300]}")
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"]
    return json.loads(raw.strip().replace("```json", "").replace("```", ""))

async def call_gemini(client: httpx.AsyncClient, wine_query: str) -> dict:
    resp = await client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={get_gemini_key()}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": build_prompt(wine_query)}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(raw.strip().replace("```json", "").replace("```", ""))

async def synthesize_with_claude(client, wine_query, results):
    synthesis_prompt = f"""?¹ì‹ ?€ ?€???„ë¬¸ ?¸ì§‘?¥ì…?ˆë‹¤. ?„ë˜???™ì¼???€??"{wine_query}"???€??Claude?€ Gemini ??AIê°€ ?…ë¦½?ìœ¼ë¡??ì„±???•ë³´?…ë‹ˆ??

=== Claude ?‘ë‹µ ===
{json.dumps(results.get("claude", {}), ensure_ascii=False, indent=2)}

=== Gemini ?‘ë‹µ ===
{json.dumps(results.get("gemini", {}), ensure_ascii=False, indent=2)}

?¹ì‹ ???„ë¬´:
1. ???‘ë‹µ?ì„œ ê³µí†µ?ìœ¼ë¡??¸ê¸‰???¬ì‹¤(? ë¢°???’ìŒ)??ì¤‘ì‹¬?¼ë¡œ ?µí•©
2. ??ê³³ì—?œë§Œ ?¸ê¸‰???´ìš©?€ ? ì¤‘?˜ê²Œ ê²€?????¬í•¨ ?¬ë? ê²°ì •
3. ?œë¡œ ?ì¶©?˜ëŠ” ?•ë³´??ê°€??? ë¢°???’ì? ?´ìš© ? íƒ (ë³´ìˆ˜???‘ê·¼)
4. ê°??¹ì…˜??confidence ?ìˆ˜ ì¶”ê? (0-100, ??AIê°€ ?¼ë§ˆ???¼ì¹˜?ˆëŠ”ì§€)
5. ìµœì¢… ?•ë³´ë¥??ë??˜ê³  ?•í™•?˜ê²Œ ?œêµ­?´ë¡œ ?‘ì„±

ë°˜ë“œ???„ë˜ JSON ?•ì‹?¼ë¡œë§??‘ë‹µ:

{{
  "wine_name": "?•í™•???€?¸ëª…",
  "wine_subtitle": "?ì‚°ì§€ Â· ?ˆì¢… Â· ë¶„ë¥˜",
  "producer":     {{"text": "...", "confidence": 85}},
  "production":   {{"text": "...", "confidence": 90}},
  "vineyard":     {{"text": "...", "confidence": 88}},
  "vintage":      {{"text": "...", "confidence": 82}},
  "tasting":      {{"text": "...", "confidence": 75}},
  "lore":         {{"text": "...", "confidence": 70}},
  "comparison":   {{"text": "...", "confidence": 80}},
  "synthesis_note": "??AI ê°„ì˜ ì£¼ìš” ì°¨ì´???ëŠ” ?¹ì´?¬í•­ ??ì¤??”ì•½"
}}"""

    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": get_anthropic_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-3-haiku-20240307",
            "max_tokens": 5000,
            "messages": [{"role": "user", "content": synthesis_prompt}],
        },
        timeout=90,
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
        raise HTTPException(status_code=400, detail="?€???´ë¦„???…ë ¥?´ì£¼?¸ìš”.")

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
                print(f"[{key}] ERROR: {result}")
            else:
                results[key] = result

        if not results:
            raise HTTPException(status_code=502, detail="ëª¨ë“  AI ?¸ì¶œ???¤íŒ¨?ˆìŠµ?ˆë‹¤.")

        final = await synthesize_with_claude(client, req.query, results)
        final["sources_used"] = list(results.keys())
        final["sources_failed"] = errors

        return final

@app.get("/health")
async def health():
    key = get_anthropic_key()
    return {
        "status": "ok",
        "apis": {
            "claude": bool(key),
            "gemini": bool(get_gemini_key()),
        },
        "claude_key_preview": key[:15] + "..." if key else "?†ìŒ",
        "claude_key_length": len(key),
    }

# Cave & Vigne — Wine Intelligence

3개 AI(Claude, GPT-4o, Gemini)가 교차검증한 와인 정보 탐색기.

## 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경변수 설정 (.env 파일 생성)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...

# 3. 실행
uvicorn main:app --reload --port 8000
# → http://localhost:8000
```

## Railway 배포

### 1. GitHub에 push
```bash
git init
git add .
git commit -m "init: wine explorer"
git remote add origin https://github.com/YOUR_ID/wine-explorer.git
git push -u origin main
```

### 2. Railway 프로젝트 생성
1. railway.app → New Project → Deploy from GitHub repo
2. 해당 repo 선택

### 3. 환경변수 설정 (중요!)
Railway 대시보드 → Variables 탭:
```
ANTHROPIC_API_KEY = sk-ant-...
OPENAI_API_KEY    = sk-...
GEMINI_API_KEY    = AI...
```

### 4. 도메인 발급
Railway 대시보드 → Settings → Networking → Generate Domain

## 아키텍처

```
[브라우저] → POST /api/wine
              ↓
         [FastAPI 서버]
              ↓ asyncio.gather (병렬)
    ┌─────────┬─────────┬─────────┐
 Claude    GPT-4o   Gemini
    └─────────┴─────────┴─────────┘
              ↓
     [Claude 교차검증 합성]
     - 공통 사실 → 신뢰도 HIGH
     - 단독 언급 → 신뢰도 LOW
     - 상충 정보 → 보수적 선택
              ↓
     confidence 점수와 함께 반환
```

## 파일 구조
```
wine-explorer/
├── main.py           # FastAPI 서버 + AI 호출 + 교차검증
├── templates/
│   └── index.html    # 프론트엔드 SPA
├── static/           # 정적 파일 (필요시)
├── requirements.txt
├── Procfile
├── railway.toml
└── .gitignore
```

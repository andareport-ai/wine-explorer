# Wine Explorer - Claude Code 프로젝트 가이드

## 프로젝트 구조
- `main.py`: FastAPI 서버 (Claude Haiku + Gemini Flash 교차검증, 와인명 정규화, 지오코딩)
- `templates/index.html`: 프론트엔드 (CSS + JS 인라인, Leaflet 지도, localStorage 캐시)
- `requirements.txt`: Python 패키지
- `railway.toml`: Railway 배포 설정

## 배포
- Railway 자동 배포: `git push origin main`
- Railway 프로젝트: tender-fascination (wine-explorer 서비스)
- Volume: `/data` (wine_cache.json, wine_name_index.json 영구 저장)
- Railway CLI: `MSYS_NO_PATHCONV=1` 필수 (Git Bash 경로 변환 방지)

## 환경변수 (Railway)
- ANTHROPIC_API_KEY, GEMINI_API_KEY, GOOGLE_MAPS_API_KEY, CACHE_DIR=/data

## 아키텍처
- 와인명 정규화 → 캐시 조회 → (미스 시) Claude+Gemini 병렬 → 교차검증 합성 → 지오코딩 → 캐시 저장
- 서버 캐시: wine_cache.json (`_geocoded` 플래그로 유효성 판단)
- 프론트 캐시: localStorage (wine_cache_v2, wine_name_map_v2, normKey 정규화)
- 지오코딩: Google Maps API 우선, Nominatim 폴백
- 지도: Leaflet.js + OpenStreetMap
- 가격: 750ml 1병 기준, Wine-Searcher + Vivino + 리테일 다중 출처

## 주의사항
- 정규식에 중첩 수량자+\s 조합 금지 (catastrophic backtracking)
- max_tokens: 8000 (Claude 개별/합성 모두)
- Gemini 2.0 Flash deprecated (2.5 Flash → 2.5 Pro 폴백)

## 사용자 선호
- 한국어 소통, 간결한 커뮤니케이션
- 배포: git push로 요청

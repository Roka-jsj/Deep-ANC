# CLAUDE.md

**[AGENTS.md](AGENTS.md)를 먼저 읽어라** — 절대 규칙(anc_project 읽기전용, Jetson 시스템 불가침,
AI 커밋 표기 금지 등)과 환경 요약이 있다. 규칙은 AGENTS.md가 단일 출처다.

**"이어서 진행해줘" 요청 시: [HANDOFF.md](HANDOFF.md)의 "다음 단계"를 위에서부터 실행하라.**

Claude Code 전용 참고:
- 파이썬은 반드시 `.venv/bin/python` (시스템 python3에는 torch 없음)
- 커밋 후 HANDOFF.md의 상태 섹션을 갱신할 것 (다음 세션의 진입점이다)
- 대규모 변경 전 `.venv/bin/python -m pytest -q` 로 기준선 확인 (30+개 전부 통과 유지가 규칙)

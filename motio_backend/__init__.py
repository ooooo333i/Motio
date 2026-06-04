"""Motio 백엔드 어댑터 패키지.

motio-frontend 의 API 명세(motio-demo-api-spec.md)를 그대로 구현하고,
내부에서 추론서버(entry_point_opt.py, Tracker, postprocess_pipeline)를 subprocess 로 호출한다.

실행: uvicorn motio_backend.main:app --host 0.0.0.0 --port 8000
설계: docs/MOTIO_INTEGRATION.md
"""

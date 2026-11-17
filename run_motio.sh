#!/usr/bin/env bash
# Motio 한 줄 실행 — 백엔드(+프론트 단일 오리진) 기동 후 공개 URL 출력.
#
#   bash /workspace/capstone_deploy/run_motio.sh            # 백엔드 기동(기존 dist 서빙)
#   bash /workspace/capstone_deploy/run_motio.sh --build    # 프론트 재빌드(실API) 후 기동 (npm 필요)
#
# 환경변수:
#   MOTIO_PORT      기본 8000
#   MOTIO_BLENDER   Blender 실행파일 경로(설정 시 결과 FBX 생성; 미설정 시 npz 까지)
#   CAP_PYTHON      추론/백엔드 파이썬 (기본 cap-deploy)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONT="$(cd "$REPO/.." && pwd)/motio-frontend"
PY="${CAP_PYTHON:-/workspace/envs/cap-deploy/bin/python}"
PORT="${MOTIO_PORT:-8000}"
export PYOPENGL_PLATFORM=        # HaMeR egl 우회

# Blender 자동 감지(FBX export 용). 5.1 우선(템플릿 호환). MOTIO_BLENDER 로 덮어쓰기 가능.
# ※ FBX 실제 생성에는 Blender 에 SMPL-X 애드온(smplx_add_animation)이 설치돼 있어야 함.
#    애드온 없으면 후처리 npz 까지만 생성되고 files.fbx=null(서비스는 정상).
if [[ -z "${MOTIO_BLENDER:-}" ]]; then
  if   [[ -x /workspace/blender5/blender ]]; then export MOTIO_BLENDER=/workspace/blender5/blender
  elif [[ -x /workspace/blender/blender  ]]; then export MOTIO_BLENDER=/workspace/blender/blender
  fi
fi

# SMPL-X 애드온은 /workspace 영속 유저리소스에 설치·활성화돼 있다(/root 는 휘발성).
# Blender 백그라운드 서브프로세스가 이 디렉터리를 읽어야 smplx_add_animation 이 등록됨.
if [[ -z "${BLENDER_USER_RESOURCES:-}" && -d /workspace/blender5_userres ]]; then
  export BLENDER_USER_RESOURCES=/workspace/blender5_userres
fi

# (옵션) 프론트 재빌드: 실제 백엔드(같은 오리진)를 바라보게 빌드
if [[ "${1:-}" == "--build" ]]; then
  if command -v npm >/dev/null 2>&1; then
    echo "[motio] 프론트 빌드 (실 API, 단일 오리진)…"
    ( cd "$FRONT" && [ -d node_modules ] || npm ci 2>/dev/null || npm install )
    ( cd "$FRONT" && VITE_USE_MOCK_API=false VITE_API_BASE_URL="" npm run build )
  else
    echo "[motio] npm 이 없어 프론트 빌드를 건너뜁니다(기존 dist 사용). node/npm 설치 후 --build."
  fi
fi

POD="${RUNPOD_POD_ID:-}"
if [[ -n "$POD" ]]; then
  URL="https://${POD}-${PORT}.proxy.runpod.net"
else
  URL="http://localhost:${PORT}"
fi

echo "──────────────────────────────────────────────"
echo " Motio 서비스 기동"
echo "   공개 URL : $URL"
echo "   로컬     : http://0.0.0.0:${PORT}   (API: $URL/api, 문서: $URL/docs)"
echo "   FBX      : ${MOTIO_BLENDER:+활성(MOTIO_BLENDER=$MOTIO_BLENDER)}${MOTIO_BLENDER:-비활성(npz까지만; MOTIO_BLENDER 설정 시 FBX)}"
echo "──────────────────────────────────────────────"

cd "$REPO"
exec "$PY" -m uvicorn motio_backend.main:app --host 0.0.0.0 --port "$PORT"

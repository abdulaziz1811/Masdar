#!/usr/bin/env bash
# Deploy Masdar to Google Cloud Run in Dammam (me-central2).
#
# The server then runs inside the Kingdom, so the national open-data portal
# (open.data.gov.sa), which refuses non-Saudi connections, is searched too.
#
# Run it in Google Cloud Shell (https://shell.cloud.google.com), from the
# project folder:
#
#   git clone https://github.com/abdulaziz1811/Masdar && cd Masdar
#   ./deploy/cloudrun.sh
#
# It asks for the access code and the optional keys, never echoes them, and
# passes them to Cloud Run through a temporary file that it deletes.
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${REGION:-me-central2}"
SERVICE="${SERVICE:-masdar}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud غير موجود. شغّل هذا السكربت من Google Cloud Shell."
  exit 1
fi
PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [ -z "$PROJECT" ]; then
  echo "اختر مشروع Google Cloud أولاً:  gcloud config set project <PROJECT_ID>"
  exit 1
fi
echo "المشروع: $PROJECT — المنطقة: $REGION (الدمام)"

read -rsp "رمز الدخول الذي سيكتبه مقدّم العرض (مطلوب): " ACCESS_CODE; echo
if [ -z "$ACCESS_CODE" ]; then
  echo "رمز الدخول مطلوب: بدونه يستطيع أي أحد لديه الرابط استخدام الخدمة."
  exit 1
fi
read -rsp "مفتاح Gemini للفهم الذكي (اختياري، Enter للتخطي): " GEMINI_API_KEY; echo
read -rsp "أو مفتاح Anthropic بدلاً منه (اختياري، Enter للتخطي): " ANTHROPIC_API_KEY; echo
read -rsp "مفتاح بوابة مطوّري الهيئة العامة للإحصاء (اختياري): " GASTAT_API_KEY; echo

ENV_FILE="$(mktemp)"
trap 'rm -f "$ENV_FILE"' EXIT
ACCESS_CODE="$ACCESS_CODE" ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" GEMINI_API_KEY="$GEMINI_API_KEY" \
GASTAT_API_KEY="$GASTAT_API_KEY" python3 - "$ENV_FILE" <<'PY'
import json, os, sys
env = {"MASDAR_ACCESS_CODE": os.environ["ACCESS_CODE"], "MASDAR_WARMUP_ON_START": "1"}
for key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GASTAT_API_KEY"):
    if os.environ.get(key):
        env[key] = os.environ[key]
# JSON is valid YAML, and quotes every value safely.
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(env, f)
PY

# One instance: conversations, download links and access grants live in the
# server's memory. No CPU throttling: the start-up warm-up runs between
# requests, which a throttled instance would stall.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 1Gi \
  --max-instances 1 \
  --no-cpu-throttling \
  --timeout 300 \
  --env-vars-file "$ENV_FILE"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
echo
echo "تم. الرابط: $URL"
echo "افتحه قبل العرض بخمس دقائق: تظهر «جاهز للعرض» أعلى الصفحة عند اكتمال التجهيز."

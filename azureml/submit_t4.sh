#!/usr/bin/env bash
# Submit the T4 QLoRA job to Azure ML (subscription pinned).
set -euo pipefail
SUB=52056bc2-3eb2-437d-95e7-36d360871ea7
RG=rg-artemis-ml
WS=artemis-ml
cd "$(dirname "$0")"
az ml job create --file t4_job.yml --subscription "$SUB" -g "$RG" -w "$WS" --query "{name:name,status:status,studio:services.Studio.endpoint}" -o json
# Download when finished:
#   az ml job download -n <job> --all --download-path ../outputs/t4-run --subscription $SUB -g $RG -w $WS

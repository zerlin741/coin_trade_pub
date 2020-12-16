#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
LOG_DIR="${ROOT_DIR}/log"
EXE_DIR="${ROOT_DIR}/coin_trade"
cd "${EXE_DIR}"
#source $HOME/miniconda3/etc/profile.d/conda.sh
#conda activate coin_env
DATETIME="$(date +%Y%m%d-%H%M%S%z)"
STRATEGY_LOG_FILE="okex_strategy.${DATETIME}"
while :
do
  python pyrunner Strategy/app/okex_strategy.py >> "${LOG_DIR}/${STRATEGY_LOG_FILE}" 2>&1
  sleep 10
done

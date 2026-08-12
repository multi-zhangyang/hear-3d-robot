#!/usr/bin/env bash
set -euo pipefail

: "${HEAR_WORKYARD_PID_FILE:?HEAR_WORKYARD_PID_FILE is required}"
: "${HEAR_WORKYARD_TIMEOUT_SECONDS:?HEAR_WORKYARD_TIMEOUT_SECONDS is required}"

case "$HEAR_WORKYARD_PID_FILE" in
  /*) ;;
  *)
    echo "HEAR_WORKYARD_PID_FILE must be absolute" >&2
    exit 2
    ;;
esac
case "$HEAR_WORKYARD_TIMEOUT_SECONDS" in
  ''|*[!0-9]*)
    echo "HEAR_WORKYARD_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
    ;;
esac
if [[ "$HEAR_WORKYARD_TIMEOUT_SECONDS" == "0" ]]; then
  echo "HEAR_WORKYARD_TIMEOUT_SECONDS must be positive" >&2
  exit 2
fi
if [[ "$#" -lt 2 ]]; then
  echo "Expected Python executable and bootstrap path" >&2
  exit 2
fi

printf '%s\n' "$$" > "$HEAR_WORKYARD_PID_FILE"
exec timeout \
  --foreground \
  --signal=TERM \
  --kill-after=30s \
  "${HEAR_WORKYARD_TIMEOUT_SECONDS}s" \
  "$@"

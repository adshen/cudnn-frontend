#!/bin/bash
set -e

expires=$1
shift
today=$(date -u +%F)

if [[ "$today" > "$expires" || "$today" == "$expires" ]]; then
    echo "ERROR: CI waiver expired on $expires; remove or extend it."
    exit 1
fi

if [[ "${CI_PIPELINE_SOURCE:-}" == "schedule" ]]; then
    echo "Nightly scheduled pipeline: running without waiver."
    exec "$@"
fi

set +e
"$@"
status=$?
set -e

if [[ $status -ne 0 ]]; then
    echo "WARNING: Waiving exit code $status until $expires: $*"
fi

exit 0

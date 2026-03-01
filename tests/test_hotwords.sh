#!/bin/bash
# test_hotwords.sh - Test hotwords and initial_prompt features
#
# Usage:
#   ./tests/test_hotwords.sh <audio_file>
#   ./tests/test_hotwords.sh testfiles/recording.flac
#
# The audio should contain domain-specific words that Whisper tends to
# misspell (brand names, acronyms, unusual proper nouns). The script runs
# three transcriptions and compares the output:
#   1. No hints (baseline)
#   2. With hotwords only
#   3. With hotwords + initial_prompt

set -euo pipefail

HOST=${HOST:-"localhost"}
PORT=${PORT:-"9000"}
BASE_URL="http://${HOST}:${PORT}"
HOTWORDS=${HOTWORDS:-"Speakr,CTranslate2,PyAnnote,SDRs"}
INITIAL_PROMPT=${INITIAL_PROMPT:-"Speakr is a transcription app built on CTranslate2 and PyAnnote."}

if [ -z "${1:-}" ] || [ ! -f "${1:-}" ]; then
    echo "Usage: $0 <audio_file>"
    echo ""
    echo "Environment variables:"
    echo "  HOST            Service host (default: localhost)"
    echo "  PORT            Service port (default: 9000)"
    echo "  HOTWORDS        Comma-separated hotwords (default: Speakr,CTranslate2,PyAnnote,SDRs)"
    echo "  INITIAL_PROMPT  Prompt to steer the model (default: Speakr is a transcription app...)"
    exit 1
fi

AUDIO_FILE="$1"

# Check service is up
if ! curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
    echo "Error: service not reachable at ${BASE_URL}"
    exit 1
fi

echo "========================================"
echo "Hotwords Test"
echo "========================================"
echo "File:           ${AUDIO_FILE}"
echo "Hotwords:       ${HOTWORDS}"
echo "Initial prompt: ${INITIAL_PROMPT}"
echo ""

echo "1) Baseline (no hints)"
echo "----------------------------------------"
text1=$(curl -s -X POST "${BASE_URL}/asr?language=en&diarize=false&output=text" \
    -F "audio_file=@${AUDIO_FILE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])")
echo "${text1}"
echo ""

echo "2) With hotwords"
echo "----------------------------------------"
text2=$(curl -s -X POST "${BASE_URL}/asr?language=en&diarize=false&output=text&hotwords=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${HOTWORDS}'))")" \
    -F "audio_file=@${AUDIO_FILE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])")
echo "${text2}"
echo ""

echo "3) With hotwords + initial_prompt"
echo "----------------------------------------"
text3=$(curl -s -X POST "${BASE_URL}/asr?language=en&diarize=false&output=text&hotwords=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${HOTWORDS}'))")&initial_prompt=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${INITIAL_PROMPT}'))")" \
    -F "audio_file=@${AUDIO_FILE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])")
echo "${text3}"
echo ""

echo "========================================"
echo "Diff"
echo "========================================"

PASS=0
FAIL=0

if [ "$text1" = "$text2" ] && [ "$text1" = "$text3" ]; then
    echo "All three outputs are identical. Hotwords had no effect on this audio."
    echo "Try using audio that contains the words in your hotwords list."
else
    if [ "$text1" != "$text2" ]; then
        echo "Hotwords changed the output (comparing 1 vs 2):"
        diff <(echo "$text1") <(echo "$text2") || true
        echo ""
    else
        echo "Hotwords alone had no effect (1 and 2 are identical)."
        echo ""
    fi
    if [ "$text2" != "$text3" ]; then
        echo "Initial prompt changed the output further (comparing 2 vs 3):"
        diff <(echo "$text2") <(echo "$text3") || true
        echo ""
    else
        echo "Initial prompt had no additional effect (2 and 3 are identical)."
        echo ""
    fi
fi

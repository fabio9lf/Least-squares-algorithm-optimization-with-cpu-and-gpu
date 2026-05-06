#!/usr/bin/env bash
# ============================================================
#  benchmark.sh
#  Uso: ./benchmark.sh [M] [N] [max_threads] [ripetizioni]
#  Default: M=5000 N=100 max_threads=16 ripetizioni=30
#
#  Compila il sorgente C, esegue il binario per ogni numero
#  di thread da 1 a max_threads, 'ripetizioni' volte ciascuno,
#  e salva i tempi in results.csv nella stessa cartella.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/least-squares-it1.c"
BIN="${SCRIPT_DIR}/ls_parallel"
CSV="${SCRIPT_DIR}/results.csv"

M="${1:-5000}"
N="${2:-100}"
MAX_T="${3:-16}"
REPS="${4:-30}"

# ---------- Compilazione ----------
echo "==> Compilazione: gcc -O2 -march=native ..."
gcc -O2 -march=native -o "$BIN" "$SRC" -lm -lpthread
echo "    Binario: $BIN"

# ---------- CSV header ----------
echo "threads,run,time_s" > "$CSV"

echo "==> Parametri: M=$M  N=$N  thread da 1 a $MAX_T  ripetizioni=$REPS"
echo ""

# ---------- Benchmark loop ----------
for T in $(seq 1 "$MAX_T"); do
    printf "Thread %2d/%d  [" "$T" "$MAX_T"
    for R in $(seq 1 "$REPS"); do
        # Estrai solo la riga col tempo (evita output di x[])
        TIME=$("$BIN" "$M" "$N" "$T" 2>/dev/null | grep "^Tempo" | awk '{print $(NF-1)}')
        echo "${T},${R},${TIME}" >> "$CSV"
        printf "."
    done
    printf "]\n"
done

echo ""
echo "==> Risultati salvati in: $CSV"
echo "==> Righe totali: $(wc -l < "$CSV") (header incluso)"
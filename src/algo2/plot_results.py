#!/usr/bin/env python3
"""
plot_results.py
---------------
Legge results.csv prodotto da benchmark.sh e genera due grafici:
  1. Tempo medio di esecuzione al crescere dei thread
  2. Speedup (T1/Tn) con confronto Amdahl

Salva: time_vs_threads.png  e  speedup_vs_threads.png
"""

from __future__ import annotations
import csv
import math
import sys
import os
from collections import defaultdict
import statistics

# ── dipendenze ──────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import FancyArrowPatch
except ImportError:
    sys.exit("Errore: installa matplotlib  →  pip install matplotlib")

# ── parametri ───────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "results.csv")
OUT_TIME   = os.path.join(SCRIPT_DIR, "time_vs_threads.png")
OUT_SPEED  = os.path.join(SCRIPT_DIR, "speedup_vs_threads.png")

# ── palette / stile ─────────────────────────────────────────
BG        = "#0d1117"
SURFACE   = "#161b22"
BORDER    = "#30363d"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
ACCENT1   = "#58a6ff"   # blu  – tempo reale / speedup reale
ACCENT2   = "#3fb950"   # verde – Amdahl teorico
ACCENT3   = "#f78166"   # rosso – ideale lineare
FILL_A    = "#58a6ff22"
FILL_B    = "#3fb95018"

# ── lettura CSV ─────────────────────────────────────────────
data: dict[int, list[float]] = defaultdict(list)

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data[int(row["threads"])].append(float(row["time_s"]))

threads_list = sorted(data.keys())
mean_times   = [statistics.mean(data[t]) for t in threads_list]
std_times    = [statistics.stdev(data[t]) if len(data[t]) > 1 else 0
                for t in threads_list]

t1 = mean_times[0]   # tempo con 1 thread (baseline)
speedup_real = [t1 / m for m in mean_times]

# ── stima frazione seriale (Amdahl fit) ─────────────────────
# Amdahl: S(n) = 1 / (f + (1-f)/n)
# Fit su tutti i punti minimizzando errore quadratico medio
def amdahl_speedup(f, n):
    return 1.0 / (f + (1 - f) / n)

best_f, best_err = 0.5, float("inf")
for f_try in [i / 1000 for i in range(1, 1000)]:
    err = sum((amdahl_speedup(f_try, n) - s) ** 2
              for n, s in zip(threads_list, speedup_real))
    if err < best_err:
        best_err, best_f = err, f_try

amdahl_line = [amdahl_speedup(best_f, n) for n in threads_list]
ideal_line  = [float(n) for n in threads_list]

print(f"  Frazione seriale stimata (Amdahl fit): {best_f:.3f}")
print(f"  Speedup massimo teorico: {1/best_f:.1f}×")

# ╔══════════════════════════════════════════════════════════╗
# ║  GRAFICO 1 – Tempo medio vs thread                      ║
# ╚══════════════════════════════════════════════════════════╝
fig1, ax1 = plt.subplots(figsize=(9, 5.5), facecolor=BG)
ax1.set_facecolor(SURFACE)
for spine in ax1.spines.values():
    spine.set_edgecolor(BORDER)

ax1.errorbar(
    threads_list, mean_times,
    yerr=std_times,
    color=ACCENT1, linewidth=2, marker="o", markersize=6,
    capsize=4, capthick=1.5, elinewidth=1.2,
    label="Tempo medio ± dev.std",
    zorder=3,
)
ax1.fill_between(
    threads_list,
    [m - s for m, s in zip(mean_times, std_times)],
    [m + s for m, s in zip(mean_times, std_times)],
    color=ACCENT1, alpha=0.12, zorder=2,
)

# ── punto minimo ────────────────────────────────────────────
min_idx  = mean_times.index(min(mean_times))
min_t    = min(mean_times)
min_th   = threads_list[min_idx]
ax1.annotate(
    f" minimo: {min_t*1000:.2f} ms\n @ {min_th} thread",
    xy=(min_th, min_t),
    xytext=(min_th + max(threads_list) * 0.05, min_t * 1.15),
    color=ACCENT2, fontsize=9,
    arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.2),
)

ax1.set_xlabel("Numero di thread", color=TEXT_SEC, fontsize=11)
ax1.set_ylabel("Tempo medio (s)", color=TEXT_SEC, fontsize=11)
ax1.set_title("Tempo di esecuzione parallela al crescere dei thread",
              color=TEXT_PRI, fontsize=13, pad=12)
ax1.tick_params(colors=TEXT_SEC)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)
ax1.legend(facecolor=SURFACE, edgecolor=BORDER,
           labelcolor=TEXT_PRI, fontsize=9)

fig1.tight_layout()
fig1.savefig(OUT_TIME, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Salvato: {OUT_TIME}")

# ╔══════════════════════════════════════════════════════════╗
# ║  GRAFICO 2 – Speedup vs thread                          ║
# ╚══════════════════════════════════════════════════════════╝
fig2, ax2 = plt.subplots(figsize=(9, 5.5), facecolor=BG)
ax2.set_facecolor(SURFACE)
for spine in ax2.spines.values():
    spine.set_edgecolor(BORDER)

# Linea ideale (lineare)
ax2.plot(threads_list, ideal_line,
         color=ACCENT3, linewidth=1.2,
         linestyle="--", alpha=0.7, label="Ideale (lineare)")

# Amdahl teorico
ax2.fill_between(threads_list, amdahl_line, ideal_line,
                 color=ACCENT2, alpha=0.08)
ax2.plot(threads_list, amdahl_line,
         color=ACCENT2, linewidth=1.8,
         linestyle=":", label=f"Amdahl (f≈{best_f:.2f}, S_max≈{1/best_f:.1f}×)")

# Speedup reale
ax2.plot(threads_list, speedup_real,
         color=ACCENT1, linewidth=2.4, marker="o", markersize=7,
         label="Speedup reale", zorder=4)
ax2.fill_between(threads_list, speedup_real, 1,
                 color=ACCENT1, alpha=0.10, zorder=1)

# Linea di base y=1
ax2.axhline(1, color=BORDER, linewidth=0.8, linestyle="-")

# Annotazione picco speedup
peak_idx = speedup_real.index(max(speedup_real))
peak_s   = max(speedup_real)
peak_th  = threads_list[peak_idx]
ax2.annotate(
    f" picco: {peak_s:.2f}×\n @ {peak_th} thread",
    xy=(peak_th, peak_s),
    xytext=(peak_th + max(threads_list) * 0.05, peak_s * 0.9),
    color=ACCENT1, fontsize=9,
    arrowprops=dict(arrowstyle="->", color=ACCENT1, lw=1.2),
)

ax2.set_xlabel("Numero di thread", color=TEXT_SEC, fontsize=11)
ax2.set_ylabel("Speedup  (T₁ / Tₙ)", color=TEXT_SEC, fontsize=11)
ax2.set_title("Speedup al crescere dei thread",
              color=TEXT_PRI, fontsize=13, pad=12)
ax2.tick_params(colors=TEXT_SEC)
ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax2.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)
ax2.legend(facecolor=SURFACE, edgecolor=BORDER,
           labelcolor=TEXT_PRI, fontsize=9)

fig2.tight_layout()
fig2.savefig(OUT_SPEED, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Salvato: {OUT_SPEED}")

# ── riepilogo testuale ───────────────────────────────────────
print()
print(f"{'Thread':>8}  {'Media (ms)':>12}  {'Dev.std (ms)':>13}  {'Speedup':>9}")
print("-" * 52)
for t, m, s, sp in zip(threads_list, mean_times, std_times, speedup_real):
    print(f"{t:>8}  {m*1000:>12.3f}  {s*1000:>13.3f}  {sp:>9.3f}×")
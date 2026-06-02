import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Leggi il CSV
df = pd.read_csv("risultati_benchmark_cpu.csv")

# Estrai solo il nome dell'eseguibile per la legenda
df["Nome"] = df["Eseguibile"].apply(lambda x: Path(x).name)

# Calcolo speedup
df["Speedup"] = 0.0

for exe in df["Nome"].unique():
    t1 = df[(df["Nome"] == exe) & (df["Num_Threads"] == 1)]["Media_ms"].iloc[0]
    mask = df["Nome"] == exe
    df.loc[mask, "Speedup"] = t1 / df.loc[mask, "Media_ms"]

# Ordina per numero di thread
df = df.sort_values(["Nome", "Num_Threads"])

# -------------------------
# Grafico tempo medio
# -------------------------
plt.figure(figsize=(10, 6))

for exe in df["Nome"].unique():
    subset = df[df["Nome"] == exe]

    plt.errorbar(
        subset["Num_Threads"],
        subset["Media_ms"],
        yerr=subset["CI_95_ms"],
        marker="o",
        capsize=4,
        label=exe
    )

plt.xlabel("Numero di thread")
plt.ylabel("Tempo medio di esecuzione (ms)")
plt.title("Tempo medio di esecuzione")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig("tempo_esecuzione.png", dpi=300)
plt.show()

# -------------------------
# Grafico speedup
# -------------------------
plt.figure(figsize=(10, 6))

for exe in df["Nome"].unique():
    subset = df[df["Nome"] == exe]

    plt.plot(
        subset["Num_Threads"],
        subset["Speedup"],
        marker="o",
        linewidth=2,
        label=exe
    )

# Speedup ideale
max_threads = df["Num_Threads"].max()
plt.plot(
    range(1, max_threads + 1),
    range(1, max_threads + 1),
    linestyle="--",
    label="Speedup ideale"
)

plt.xlabel("Numero di thread")
plt.ylabel("Speedup")
plt.title("Speedup rispetto a 1 thread")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig("speedup.png", dpi=300)
plt.show()
import subprocess
import re
import statistics
import math
import csv
import os

# ==========================================
# 1. CONFIGURAZIONE BENCHMARK CPU
# ==========================================
# Inserisci qui i percorsi dei tuoi eseguibili C++
eseguibili = [
    r"/mnt/c/universita/Computer Architecture/progetto/Least-squares-algorithm-optimization-with-cpu-ang-gpu/src/algo2/least-squares-it-final.exe"
    r"/mnt/c/universita/Computer Architecture/progetto/Least-squares-algorithm-optimization-with-cpu-ang-gpu/src/least-squares-opt3-final.exe",
    #r"/mnt/c/universita/Computer Architecture/progetto/Least-squares-algorithm-optimization-with-cpu-ang-gpu/src/least-squares-opt3-3.exe",
    #r"/mnt/c/universita/Computer Architecture/progetto/Least-squares-algorithm-optimization-with-cpu-ang-gpu/src/algo2/least-squares-it2.exe"
]

M = "4000"
N = "3000"
iterazioni = 30 #a 30 per avere un Confidence Interval statisticamente valido

# File di salvataggio sicuro
file_output = "risultati_benchmark_cpu.csv"

# ==========================================
# 2. SISTEMA DI CHECKPOINT (Resilienza ai crash)
# ==========================================
def leggi_test_completati(nome_file):
    completati = set()
    if os.path.exists(nome_file):
        with open(nome_file, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # Salta l'intestazione
            for riga in reader:
                if len(riga) >= 2:
                    # Salva una tupla (Nome_Eseguibile, Numero_Thread)
                    completati.add((riga[0], int(riga[1])))
    return completati

test_fatti = leggi_test_completati(file_output)

# Crea il file con l'intestazione se è la prima volta che avvii lo script
if not os.path.exists(file_output):
    with open(file_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Eseguibile", "Num_Threads", "Media_ms", "CI_95_ms"])

# ==========================================
# 3. MOTORE DI ESECUZIONE
# ==========================================
print("--- AVVIO SUPER-BENCHMARK CPU ---")
print(f"Target: Matrice {M}x{N} | {iterazioni} Iterazioni per test")
print(f"Salvataggio automatico su: {file_output}\n")

for exe in eseguibili:
    if not os.path.exists(exe):
        print(f"[!] ATTENZIONE: File '{exe}' non trovato. Salto al prossimo...")
        continue

    print(f"\n>>> TESTANDO ESEGUIBILE: {os.path.basename(exe)}")
    
    # Loop sui Thread: da 1 a 10 inclusi
    for num_threads in range(1, 11):
        
        # Controllo anti-crash: test già fatto?
        if (exe, num_threads) in test_fatti:
            print(f"    - Thread: {num_threads:2d} -> [GIA' COMPLETATO. Skippato]")
            continue

        print(f"    - Thread: {num_threads:2d} -> Calcolo in corso...", end='', flush=True)
        tempi = []

        # Esecuzione delle iterazioni
        for i in range(iterazioni):
            # Passiamo M, N e i Thread
            comando = [exe, M, N, str(num_threads)]
            risultato = subprocess.run(comando, capture_output=True, text=True)
            
            # Regex case-insensitive globale per quella stringa
            match = re.search(r"tempo solo parallelo:\s*([0-9.]+)\s*(ms|s)", risultato.stdout, re.IGNORECASE)
            
            if match:
                # Sostituisce l'eventuale virgola e converte in float
                tempo_str = match.group(1).replace(',', '.')
                tempo = float(tempo_str)
                unita = match.group(2)
                
                # Normalizza tutto in millisecondi
                if unita == 's':
                    tempo *= 1000.0
                    
                tempi.append(tempo)
            else:
                print(f"\n[!] Errore iterazione {i+1}. Testo ricevuto:\n{risultato.stdout.strip()[:100]}...")

        # Elaborazione statistica e salvataggio
        if tempi:
            media = statistics.mean(tempi)
            stdev = statistics.stdev(tempi) if len(tempi) > 1 else 0.0
            
            # Calcolo dell'Intervallo di Confidenza al 95%
            ci_95 = 1.96 * (stdev / math.sqrt(len(tempi)))
            
            # Scrittura asincrona in Append
            with open(file_output, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([exe, num_threads, round(media, 3), round(ci_95, 3)])
            
            print(f" Fatto! (Media: {media:.2f} ms | CI 95%: ±{ci_95:.2f} ms)")
        else:
            print(" Fallito! (Nessun tempo valido registrato)")

print("\n=== TUTTI I BENCHMARK CPU SONO COMPLETATI ===")
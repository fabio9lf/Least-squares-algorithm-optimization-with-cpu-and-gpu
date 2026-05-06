@echo off
:: ============================================================
::  benchmark.bat
::  Uso: benchmark.bat [M] [N] [max_threads] [ripetizioni]
::  Default: M=5000 N=100 max_threads=16 ripetizioni=30
::
::  Requisiti:
::    - GCC installato (es. MinGW-w64 o TDM-GCC)
::      https://www.mingw-w64.org/  oppure  https://jmeubank.github.io/tdm-gcc/
::    - gcc.exe raggiungibile nel PATH
::
::  Metti questo file nella stessa cartella di:
::    least_squares_parallel.c
::    plot_results.py
:: ============================================================
setlocal enabledelayedexpansion

:: ── Parametri (usa argomenti CLI o default) ─────────────────
set M=%~1
set N=%~2
set MAX_T=%~3
set REPS=%~4

if "%M%"==""     set M=5000
if "%N%"==""     set N=100
if "%MAX_T%"=""  set MAX_T=16
if "%REPS%"=""   set REPS=30

set SCRIPT_DIR=%~dp0
set SRC=%SCRIPT_DIR%least_squares_parallel.c
set BIN=%SCRIPT_DIR%ls_parallel.exe
set CSV=%SCRIPT_DIR%results.csv

:: ── Compilazione ────────────────────────────────────────────
echo =^> Compilazione: gcc -O2 -march=native ...
gcc -O2 -march=native -o "%BIN%" "%SRC%" -lm -lpthread
if errorlevel 1 (
    echo.
    echo [ERRORE] Compilazione fallita.
    echo Verifica che gcc sia installato e nel PATH.
    echo Scarica MinGW-w64: https://www.mingw-w64.org/
    pause
    exit /b 1
)
echo     Binario: %BIN%
echo.

:: ── CSV header ──────────────────────────────────────────────
echo threads,run,time_s > "%CSV%"

echo =^> Parametri: M=%M%  N=%N%  thread da 1 a %MAX_T%  ripetizioni=%REPS%
echo.

:: ── Benchmark loop ──────────────────────────────────────────
for /l %%T in (1,1,%MAX_T%) do (
    set /a PAD=%%T
    <nul set /p "=Thread %%T/%MAX_T%  ["

    for /l %%R in (1,1,%REPS%) do (
        :: Esegui il binario e cattura tutto l'output in un file temporaneo
        "%BIN%" %M% %N% %%T > "%TEMP%\ls_out.txt" 2>nul

        :: Estrai il numero del tempo dalla riga "Tempo sezione parallela: X s"
        set TIME_VAL=
        for /f "tokens=4" %%L in ('findstr /c:"Tempo sezione parallela" "%TEMP%\ls_out.txt"') do (
            set TIME_VAL=%%L
        )

        :: Scrivi nel CSV solo se abbiamo un valore valido
        if not "!TIME_VAL!"=="" (
            echo %%T,%%R,!TIME_VAL! >> "%CSV%"
        )
        <nul set /p "=."
    )
    echo ]
)

:: ── Conta righe ─────────────────────────────────────────────
set COUNT=0
for /f %%C in ('find /c /v "" ^< "%CSV%"') do set COUNT=%%C

echo.
echo =^> Risultati salvati in: %CSV%
echo =^> Righe totali: %COUNT% (header incluso)
echo.

:: ── Lancia il plotter Python ────────────────────────────────
echo =^> Generazione grafici con Python...
python "%SCRIPT_DIR%plot_results.py"
if errorlevel 1 (
    echo.
    echo [ATTENZIONE] Python non trovato o errore nel plot.
    echo Installa matplotlib con:  pip install matplotlib
    echo Poi lancia manualmente:   python plot_results.py
)

echo.
echo =^> Tutto completato.
pause
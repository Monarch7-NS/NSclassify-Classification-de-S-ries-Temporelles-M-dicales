# Fin de l'etude 2 en parallele (reprise du pilote d'origine, arrete apres la recherche CNN).
# cnn_lstm : plis restants de la recherche en workers separes, puis finals (1 worker par seed).
$ErrorActionPreference = "Continue"
Set-Location "C:\NS\POLYTECH\DATA931\PROJET"
$env:EXPERIMENT_CONFIG = "configs/experiment_rr.json"
$env:OMP_NUM_THREADS = "2"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$py = ".\.venv\Scripts\python.exe"
$log = "results_etude2\logs"

function Start-Py($name, $argList) {
    Start-Process -FilePath $py -ArgumentList (@("scripts\etude2_cv.py") + $argList) -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$log\$name.out.log" -RedirectStandardError "$log\$name.err.log"
}
function Run-Py($name, $argList) { $p = Start-Py $name $argList; $p.WaitForExit() }

"FINISH-START $(Get-Date -Format s)" | Out-File "results_etude2\ETUDE2_STATUS.txt" -Append -Encoding ascii

# Recherche cnn_lstm : config 2 (plis 1-4) et config 5 (plis 3-4)
$w = @()
foreach ($k in 1,2,3,4) { $w += Start-Py "p_ls_c2_f$k" @("search","--family","cnn_lstm","--configs","2","--folds","$k") }
foreach ($k in 3,4)     { $w += Start-Py "p_ls_c5_f$k" @("search","--family","cnn_lstm","--configs","5","--folds","$k") }
# Finals CNN : un worker par seed (le worker sequentiel d'origine est arrete)
$cf = @()
foreach ($s in 42,43,44,45,46) { $cf += Start-Py "p_final_cnn_s$s" @("final","--family","cnn","--seeds","$s") }

foreach ($p in $w) { $p.WaitForExit() }

Run-Py "select_lstm" @("select","--family","cnn_lstm")
$env:OMP_NUM_THREADS = "3"
$lf = @()
foreach ($s in 42,43,44,45,46) { $lf += Start-Py "p_final_lstm_s$s" @("final","--family","cnn_lstm","--seeds","$s") }
foreach ($p in $lf) { $p.WaitForExit() }
foreach ($p in $cf) { $p.WaitForExit() }

Run-Py "select_all" @("select")
Run-Py "report" @("report")
"DONE $(Get-Date -Format s)" | Out-File "results_etude2\ETUDE2_DONE.txt" -Encoding ascii

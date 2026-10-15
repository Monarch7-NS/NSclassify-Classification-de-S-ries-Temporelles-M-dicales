# Étude 2 (etude2_cv_groupee) : recherche CV groupée -> sélection -> finals (5 seeds) -> rapport.
# Reprenable : chaque run (famille, config, pli) écrit son JSON ; les runs existants sont sautés, les finals aussi.
# Lancement détaché :  Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-File','results_etude2\run_etude2.ps1'
$ErrorActionPreference = "Continue"
Set-Location "C:\NS\POLYTECH\DATA931\PROJET"
$env:EXPERIMENT_CONFIG = "configs/experiment_rr.json"
$env:OMP_NUM_THREADS = "3"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$py = ".\.venv\Scripts\python.exe"
$log = "results_etude2\logs"
New-Item -ItemType Directory -Force -Path $log | Out-Null

function Start-Py($name, $argList) {
    Start-Process -FilePath $py -ArgumentList (@("scripts\etude2_cv.py") + $argList) -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$log\$name.out.log" -RedirectStandardError "$log\$name.err.log"
}
function Run-Py($name, $argList) {
    $p = Start-Py $name $argList
    $p.WaitForExit()
}

"START $(Get-Date -Format s)" | Out-File "results_etude2\ETUDE2_STATUS.txt" -Encoding utf8

# Phase 1 : recherche (parallèle)
$mlp  = Start-Py "search_mlp"    @("search", "--family", "mlp")
$cnnA = Start-Py "search_cnn_a"  @("search", "--family", "cnn", "--configs", "0,1,2,6")
$cnnB = Start-Py "search_cnn_b"  @("search", "--family", "cnn", "--configs", "3,4,5")
$lsA  = Start-Py "search_lstm_a" @("search", "--family", "cnn_lstm", "--configs", "0,1,2")
$lsB  = Start-Py "search_lstm_b" @("search", "--family", "cnn_lstm", "--configs", "3,4,5")
$lr   = Start-Py "search_lr"     @("search", "--family", "lr")
$rf   = Start-Py "search_rf"     @("search", "--family", "rf")

# Phase 2 : dès qu'une famille est complète -> sélection -> final
$finals = @()
$mlp.WaitForExit();  Run-Py "select_mlp" @("select", "--family", "mlp");  $finals += Start-Py "final_mlp" @("final", "--family", "mlp")
$lr.WaitForExit();   Run-Py "select_lr"  @("select", "--family", "lr");   $finals += Start-Py "final_lr"  @("final", "--family", "lr")
$rf.WaitForExit();   Run-Py "select_rf"  @("select", "--family", "rf");   $finals += Start-Py "final_rf"  @("final", "--family", "rf")
$cnnA.WaitForExit(); $cnnB.WaitForExit(); Run-Py "select_cnn" @("select", "--family", "cnn"); $finals += Start-Py "final_cnn" @("final", "--family", "cnn")
$lsA.WaitForExit();  $lsB.WaitForExit();  Run-Py "select_lstm" @("select", "--family", "cnn_lstm"); $finals += Start-Py "final_lstm" @("final", "--family", "cnn_lstm")
foreach ($f in $finals) { $f.WaitForExit() }

Run-Py "select_all" @("select")
Run-Py "report" @("report")
"DONE $(Get-Date -Format s)" | Out-File "results_etude2\ETUDE2_DONE.txt" -Encoding utf8

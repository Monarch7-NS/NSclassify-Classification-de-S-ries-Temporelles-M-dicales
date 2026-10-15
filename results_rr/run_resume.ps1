Set-Location C:\NS\POLYTECH\DATA931\PROJET
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONWARNINGS = 'ignore'
$env:EXPERIMENT_CONFIG = 'configs/experiment_rr.json'; $env:RESULTS_DIR = 'results_rr'
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path

function Set-Threads($n) { $env:OMP_NUM_THREADS = "$n"; $env:MKL_NUM_THREADS = "$n" }

function Run-Parallel($jobs) {
    $procs = foreach ($j in $jobs) {
        Start-Process -FilePath $py -ArgumentList $j.args -RedirectStandardOutput $j.log `
            -RedirectStandardError ($j.log + '.err') -NoNewWindow -PassThru
    }
    $procs | Wait-Process
}

$m = '-m', 'src.run_experiments'

# Reprise après interruption : les runs déjà "ok" (metrics.json) sont sautés par run_experiment.
Set-Threads 8
Run-Parallel @(
    @{ log = 'results_rr\stage_factorial_cnn_r.log';      args = $m + '--stage', 'factorial', '--family', 'cnn' },
    @{ log = 'results_rr\stage_factorial_cnn_lstm_r.log'; args = $m + '--stage', 'factorial', '--family', 'cnn_lstm' }
)

& $py @m --stage select 2>&1 | Out-File results_rr\stage_select.log -Encoding utf8

Set-Threads 4
Run-Parallel @(
    @{ log = 'results_rr\stage_final_mlp.log';      args = $m + '--stage', 'final', '--family', 'mlp' },
    @{ log = 'results_rr\stage_final_cnn.log';      args = $m + '--stage', 'final', '--family', 'cnn' },
    @{ log = 'results_rr\stage_final_cnn_lstm.log'; args = $m + '--stage', 'final', '--family', 'cnn_lstm' },
    @{ log = 'results_rr\stage_final_sk.log';       args = $m + '--stage', 'final', '--family', 'lr', 'rf' }
)

'CHAIN DONE' | Out-File results_rr\CHAIN_DONE.txt

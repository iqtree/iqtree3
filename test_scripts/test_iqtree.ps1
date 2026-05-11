param (
    [string]$IQTreeBin = "./build/iqtree3",
    [string]$LogFile   = "time_log.tsv",
    [string]$OutDir    = "test_scripts/test_data"
)

$SEED   = 73073
$LOGFILE = $LogFile
$WD     = "test_scripts/test_data"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

"Command`tRealTime(s)`tPeakMemory(MB)" | Out-File -FilePath $LOGFILE -Encoding utf8

function Measure-IQTree {
    param (
        [string]$CommandLine
    )

    Write-Host "`n===== RUNNING: $CommandLine ====="

    $startTime = Get-Date

    # Split command and arguments
    $exe, $args = $CommandLine -split '\s+', 2
    $tempOut = [System.IO.Path]::GetTempFileName()

    # Start the process with output redirection
    $proc = Start-Process -FilePath $exe -ArgumentList $args `
        -RedirectStandardOutput $tempOut `
        -NoNewWindow -PassThru

    $procId = $proc.Id
    $peakMemory = 0

    while (-not $proc.HasExited) {
        Start-Sleep -Milliseconds 200
        try {
            $currentMem = (Get-Process -Id $procId -ErrorAction Stop).WorkingSet64 / 1MB
            if ($currentMem -gt $peakMemory) {
                $peakMemory = $currentMem
            }
        } catch {
            break
        }
    }

    $endTime = Get-Date
    $elapsed = [math]::Round(($endTime - $startTime).TotalSeconds, 2)

    if ($proc.ExitCode -ne 0) {
        Write-Host "WARNING: command exited with code $($proc.ExitCode), logging 0 for time and memory"
        $elapsed    = 0
        $peakMemory = 0
    }

    # Show output
    Get-Content $tempOut

    # Log timing and memory
    "$CommandLine`t$elapsed`t$([math]::Round($peakMemory, 2))" | Out-File -FilePath $LOGFILE -Append -Encoding utf8

    # Cleanup
    Remove-Item $tempOut
}

Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -B 1000 -T 1 -seed $SEED --prefix $OutDir/turtle.fa"

Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -p $WD/turtle.nex -B 1000 -T 1 -seed $SEED --prefix $OutDir/turtle.nex"

Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -p $WD/turtle.nex -B 1000 -T 1 -m MFP+MERGE -rcluster 10 --prefix $OutDir/turtle.merge -seed $SEED"

Get-Content "$OutDir/turtle.fa.treefile", "$OutDir/turtle.nex.treefile" |
    Set-Content "$OutDir/turtle.trees"
Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -p $OutDir/turtle.merge.best_scheme.nex -z $OutDir/turtle.trees -zb 10000 -au -n 0 --prefix $OutDir/turtle.test -seed $SEED -T 1"

Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -m GTR+F+I+R3+T -te $OutDir/turtle.trees -T 1 --prefix $OutDir/turtle.mix -seed $SEED"

Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -p $OutDir/turtle.nex.best_scheme.nex -z $OutDir/turtle.trees -n 0 -wpl --prefix $OutDir/turtle.wpl -seed $SEED -T 1"

Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -S $WD/turtle.nex --prefix $OutDir/turtle.loci -T 1 -seed $SEED"

Measure-IQTree "$IQTreeBin -t $OutDir/turtle.nex.treefile --gcf $OutDir/turtle.loci.treefile -s $WD/turtle.fa --scf 100 -seed $SEED -T 1 --prefix $OutDir/turtle.nex.cf"

Measure-IQTree "$IQTreeBin -t $OutDir/turtle.fa.treefile --gcf $OutDir/turtle.loci.treefile -s $WD/turtle.fa --scf 100 -seed $SEED -T 1 --prefix $OutDir/turtle.fa.cf"

# link-exchange-rates model

Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -m `"MIX{GTR+FO,GTR+FO}`" --link-exchange-rates --prefix $OutDir/turtle.mix.link -seed $SEED -T 1"

Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -m `"MIX{GTR{1,1,1,1,1,1}+FO,GTR{1,1,1,1,1,1}+FO}`" --link-exchange-rates --prefix $OutDir/turtle.mix.jc.link -seed $SEED -T 1"

Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -p $WD/turtle.nex -g $WD/turtle.constr.tree --prefix $OutDir/turtle.nex.constr -T 1 -seed $SEED"

Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -p $WD/turtle.nex -g $WD/turtle.constr.tree2 -B 1000 -alrt 1000 --prefix $OutDir/turtle.nex.constr2 -T 1 -seed $SEED"

Measure-IQTree "$IQTreeBin -s $WD/turtle.fa -m `"MIX+MF`" --prefix $OutDir/turtle.mixfinder -T 1 -seed $SEED"

## amino acid test cases
Write-Host "Running amino acid test cases..."

Measure-IQTree "$IQTreeBin -s $WD/turtle_aa.fasta -B 1000 -T 1 -seed $SEED --prefix $OutDir/turtle_aa.fasta"

Measure-IQTree "$IQTreeBin -s $WD/turtle_aa.fasta -p $WD/turtle_aa.nex -B 1000 -T 1 -seed $SEED --prefix $OutDir/turtle_aa.nex"

Measure-IQTree "$IQTreeBin -s $WD/turtle_aa.fasta -p $WD/turtle_aa.nex -B 1000 -T 1 -m MFP+MERGE -rcluster 10 --prefix $OutDir/turtle_aa.merge -seed $SEED"

Get-Content "$OutDir/turtle_aa.fasta.treefile", "$OutDir/turtle_aa.nex.treefile" |
    Set-Content "$OutDir/turtle_aa.trees"
Measure-IQTree "$IQTreeBin -s $WD/turtle_aa.fasta -p $OutDir/turtle_aa.merge.best_scheme.nex -z $OutDir/turtle_aa.trees -zb 10000 -au -n 0 --prefix $OutDir/turtle_aa.test -seed $SEED -T 1"

Measure-IQTree "$IQTreeBin -s $WD/turtle_aa.fasta -p $OutDir/turtle_aa.nex.best_scheme.nex -z $OutDir/turtle_aa.trees -n 0 -wpl --prefix $OutDir/turtle_aa.wpl -seed $SEED -T 1"

Measure-IQTree "$IQTreeBin -s $WD/turtle_aa.fasta -S $WD/turtle_aa.nex --prefix $OutDir/turtle_aa.loci -T 1 -seed $SEED"

Measure-IQTree "$IQTreeBin -t $OutDir/turtle_aa.nex.treefile --gcf $OutDir/turtle_aa.loci.treefile -s $WD/turtle_aa.fasta --scf 100 -seed $SEED -T 1 --prefix $OutDir/turtle_aa.nex.cf"

Measure-IQTree "$IQTreeBin -t $OutDir/turtle_aa.fasta.treefile --gcf $OutDir/turtle_aa.loci.treefile -s $WD/turtle_aa.fasta --scf 100 -seed $SEED -T 1 --prefix $OutDir/turtle_aa.fasta.cf"

Measure-IQTree "$IQTreeBin -s $WD/turtle_aa.fasta -m `"MIX{LG+F,WAG+F}`" --prefix $OutDir/turtle_aa.mix -seed $SEED -T 1"

Measure-IQTree "$IQTreeBin -s $WD/turtle_aa.fasta -p $WD/turtle_aa.nex -g $WD/turtle.constr.tree --prefix $OutDir/turtle_aa.nex.constr -T 1 -seed $SEED"

Measure-IQTree "$IQTreeBin -s $WD/turtle_aa.fasta -p $WD/turtle_aa.nex -g $WD/turtle.constr.tree2 -B 1000 -alrt 1000 --prefix $OutDir/turtle_aa.nex.constr2 -T 1 -seed $SEED"

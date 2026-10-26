# Re-run one benchmark command; returns @{ Time; Mem; Ok }.
# Used to retry a check that breached. Mirrors Measure-IQTree so the retry is
# comparable, and clears the command's own outputs first.
function Measure-Once {
    param ([string]$CommandLine)

    $prefix = $null
    if ($CommandLine -match '--prefix\s+(\S+)') { $prefix = $Matches[1] }
    if ($prefix) {
        Get-ChildItem -Path "$prefix.*" -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }

    $exe, $cmdArgs = $CommandLine -split '\s+', 2
    $tempOut = [System.IO.Path]::GetTempFileName()
    $startTime = Get-Date

    $proc = Start-Process -FilePath $exe -ArgumentList $cmdArgs `
        -RedirectStandardOutput $tempOut -NoNewWindow -PassThru

    # Poll often: a command finishing inside one sleep interval would otherwise
    # never be sampled and report 0 MB.
    $peakMemory = 0
    while (-not $proc.HasExited) {
        try {
            $currentMem = (Get-Process -Id $proc.Id -ErrorAction Stop).WorkingSet64 / 1MB
            if ($currentMem -gt $peakMemory) { $peakMemory = $currentMem }
        } catch { break }
        Start-Sleep -Milliseconds 25
    }
    $proc.WaitForExit()

    # The OS-recorded peak is authoritative and survives a process too short-lived
    # to sample; fall back to the polled maximum where it is unavailable.
    try {
        $proc.Refresh()
        $osPeak = $proc.PeakWorkingSet64 / 1MB
        if ($osPeak -gt $peakMemory) { $peakMemory = $osPeak }
    } catch { }

    $elapsed = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 2)
    Remove-Item $tempOut -ErrorAction SilentlyContinue

    # Ok distinguishes "the command failed" from "it legitimately used very little".
    return @{ Time = $elapsed; Mem = [math]::Round($peakMemory, 2); Ok = ($proc.ExitCode -eq 0) }
}

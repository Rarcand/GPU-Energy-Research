$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Resolve VM ownership before sampling. CIM discovery can take tens of seconds;
# never run it inside the timing monitor's freshness-sensitive sampling loop.
$verifiedWslIds = @()
$identityCache = @{}
$identityError = $null
try {
    $systems = @(& "$env:WINDIR\System32\hcsdiag.exe" list 2>$null)
    $verifiedWslIds = @($systems | ForEach-Object {
        if ($_ -match '^\s*VM,\s*Running,\s*([0-9a-fA-F-]{36}),\s*WSL\s*$') {
            $Matches[1].ToLowerInvariant()
        }
    })
    foreach ($vm in @(Get-CimInstance Win32_Process -Filter "Name='vmwp.exe'" -ErrorAction Stop)) {
        if ($vm.CommandLine -match '\b([0-9a-fA-F-]{36})\b') {
            $vmId = $Matches[1].ToLowerInvariant()
            $process = Get-Process -Id $vm.ProcessId -ErrorAction Stop
            if ($verifiedWslIds -contains $vmId) {
                $identityCache["$($vm.ProcessId)/$($process.StartTime.ToUniversalTime().Ticks)"] = $vmId
            }
        }
    }
} catch { $identityError = $_.ToString() }
# Reopen the wildcard query each cycle: a long-lived PDH query can retain
# only the instances present when it was created, missing newly opened apps.
while ($true) {
    try {
        $sets = @(Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 2 -ErrorAction SilentlyContinue)
        if ($sets.Count -eq 0) { throw 'No GPU counter samples available' }
        $samples = @($sets[-1].CounterSamples)
        $valid = @($samples | Where-Object { $_.Status -eq 0 -or $_.Status -eq 1 })
        $activity = @()
        foreach ($counter in $valid) {
            if ($counter.CookedValue -gt 1.0 -and $counter.InstanceName -match '^pid_(\d+)_') {
                $processId = [int]$Matches[1]
                $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                $processName = if ($process) { $process.ProcessName } else { 'unknown' }
                $wslVmId = $null
                if ($processName -eq 'vmwp' -and $process) {
                    $cacheKey = "$processId/$($process.StartTime.ToUniversalTime().Ticks)"
                    if ($verifiedWslIds -contains $identityCache[$cacheKey]) {
                        $wslVmId = $identityCache[$cacheKey]
                    }
                }
                $activity += [pscustomobject]@{
                    pid = $processId
                    process = $processName
                    engine = $counter.InstanceName
                    utilization = [double]$counter.CookedValue
                    wsl_vm_id = $wslVmId
                }
            }
        }
        [pscustomobject]@{
            timestamp = [DateTime]::UtcNow.ToString('o')
            monitor_version = 'refresh-v4-wsl-preflight'
            verified_wsl_vm_ids = @($verifiedWslIds)
            identity_preflight_error = $identityError
            sample_ok = ($valid.Count -gt 0)
            invalid_counters = $samples.Count - $valid.Count
            active = @($activity)
        } | ConvertTo-Json -Compress -Depth 4
    } catch {
        [pscustomobject]@{
            timestamp = [DateTime]::UtcNow.ToString('o')
            monitor_version = 'refresh-v4-wsl-preflight'
            sample_ok = $false
            error = $_.Exception.Message
            active = @()
        } | ConvertTo-Json -Compress -Depth 4
        Start-Sleep -Seconds 2
    }
}

# 用 NSSM 把 face-cast server 注册成 Windows 服务.
# 需先跑过 install-server.ps1 + smoke 测试通过.

param(
    [string]$InstallDir = "E:\face-cast",
    [int]$Port = 9000,
    [string]$ServiceName = "face-cast"
)

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$ErrorActionPreference = "Stop"

$nssm = (Get-Command nssm -ErrorAction SilentlyContinue).Source
if (-not $nssm) { throw "nssm not found. winget install NSSM.NSSM" }

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv -or -not (Test-Path $uv)) {
    $candidates = @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe"
    )
    $uv = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $uv) { throw "uv not found. Install: winget install astral-sh.uv" }
Write-Host "uv: $uv"

$logDir = Join-Path $InstallDir "logs"
$modelDir = Join-Path $InstallDir "models"
New-Item -ItemType Directory -Force -Path $logDir, $modelDir | Out-Null

# 把交互安装时下到 ~/.insightface 的模型移到 install dir 下
# (LocalSystem 服务账户的 home 跟当前 the2n 不同, 不搬就要重下 280 MB)
$userModels = Join-Path $env:USERPROFILE ".insightface\models"
if (Test-Path $userModels) {
    Get-ChildItem $userModels -Directory | ForEach-Object {
        $dst = Join-Path $modelDir "models\$($_.Name)"
        if (-not (Test-Path $dst)) {
            New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
            Copy-Item -Recurse -Force $_.FullName $dst
            Write-Host "  copied model: $($_.Name) → $dst"
        }
    }
}

# 删除已有同名服务 (幂等). NSSM 找不到服务时返回非零, 这里忽略.
$ErrorActionPreference = "Continue"
& $nssm stop $ServiceName 2>&1 | Out-Null
& $nssm remove $ServiceName confirm 2>&1 | Out-Null
$ErrorActionPreference = "Stop"

# 注册
& $nssm install $ServiceName $uv "run" "face-server" "--host" "0.0.0.0" "--port" $Port
& $nssm set $ServiceName AppDirectory $InstallDir
& $nssm set $ServiceName Description "face-cast face recognition HTTP server (InsightFace + bottle + waitress)"
& $nssm set $ServiceName Start SERVICE_DEMAND_START   # 手动启动, 不开机自启 (避免 GPU 占用)
& $nssm set $ServiceName AppStdout (Join-Path $logDir "server.out.log")
& $nssm set $ServiceName AppStderr (Join-Path $logDir "server.err.log")
# InsightFace 模型根目录 (LocalSystem 看不到 the2n 的 home)
& $nssm set $ServiceName AppEnvironmentExtra "FACE_MODEL_ROOT=$modelDir"
# 日志轮转: 单文件 ≥10 MB 时切, 保留 7 份
& $nssm set $ServiceName AppRotateFiles 1
& $nssm set $ServiceName AppRotateOnline 1
& $nssm set $ServiceName AppRotateBytes 10485760
# 异常时重启策略
& $nssm set $ServiceName AppExit Default Restart
& $nssm set $ServiceName AppRestartDelay 5000

Write-Host ""
Write-Host "═══ 注册完成 ═══" -ForegroundColor Green
Write-Host ""
Write-Host "服务: $ServiceName"
Write-Host "启动: nssm start $ServiceName  (或 Start-Service $ServiceName)"
Write-Host "停止: nssm stop  $ServiceName"
Write-Host "状态: nssm status $ServiceName"
Write-Host "日志: $logDir\server.{out,err}.log"
Write-Host ""
Write-Host "现在启动 + 探活:"
& $nssm start $ServiceName
Start-Sleep -Seconds 12
& $nssm status $ServiceName
Write-Host ""
try {
    $r = Invoke-RestMethod -Uri "http://localhost:$Port/health" -TimeoutSec 5
    Write-Host "/health: $($r | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "/health 探活失败: $_" -ForegroundColor Red
    Write-Host "stderr 日志最后几行:"
    Get-Content (Join-Path $logDir "server.err.log") -Tail 10 -ErrorAction SilentlyContinue
}

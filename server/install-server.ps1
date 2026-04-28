# face-cast server install (Windows)
# 执行: powershell -ExecutionPolicy Bypass -File install-server.ps1
#
# 假设:
#   - NVIDIA 驱动已装 (跑 2080 Ti / 3090 / 4060+ 都行)
#   - E:\face-cast 作为安装目录 (可改 $InstallDir)

param(
    [string]$InstallDir = "E:\face-cast",
    [string]$RepoUrl = "https://github.com/naughtyGitCat/face-cast",
    [string]$Branch = "main",
    [int]$Port = 9000
)

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$ErrorActionPreference = "Stop"

Write-Host "═══ face-cast server install ═══" -ForegroundColor Cyan
Write-Host "InstallDir: $InstallDir"
Write-Host "Repo:       $RepoUrl ($Branch)"
Write-Host "Port:       $Port"
Write-Host ""

# ─── 1. 装 uv (Python 包管理) ────────────────────────────────────────────
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe"
}
if (-not (Test-Path $uv)) {
    Write-Host "[1/5] 装 uv ..." -ForegroundColor Yellow
    winget install --id=astral-sh.uv -e --silent --accept-source-agreements --accept-package-agreements
    $uv = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe"
}
Write-Host "[1/5] uv ok: $uv"

# ─── 2. 拉代码 ────────────────────────────────────────────────────────────
if (-not (Test-Path $InstallDir)) {
    Write-Host "[2/5] git clone ..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path (Split-Path $InstallDir -Parent) | Out-Null
    git clone -b $Branch $RepoUrl $InstallDir
} else {
    Write-Host "[2/5] git pull ..." -ForegroundColor Yellow
    Push-Location $InstallDir
    git pull --ff-only
    Pop-Location
}

# ─── 3. 创建 venv + 装依赖 (含 server extras) ────────────────────────────
Write-Host "[3/5] uv sync (含 server extras) ..." -ForegroundColor Yellow
Push-Location $InstallDir
& $uv sync --extra server
Pop-Location

# ─── 4. 预热模型 (首次会下载 ~280 MB buffalo_l) ──────────────────────────
Write-Host "[4/5] 预热 buffalo_l 模型 ..." -ForegroundColor Yellow
Push-Location $InstallDir
& $uv run python -c @"
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
fa.prepare(ctx_id=0)
print('model providers:', fa.models['detection'].session.get_providers())
"@
Pop-Location

# ─── 5. 防火墙开端口 ─────────────────────────────────────────────────────
Write-Host "[5/5] 防火墙规则 :$Port ..." -ForegroundColor Yellow
$ruleName = "face-cast :$Port"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort $Port -Profile Any | Out-Null
    Write-Host "  + rule added"
} else {
    Write-Host "  rule already exists"
}

# ─── done ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══ 安装完成 ═══" -ForegroundColor Green
Write-Host ""
Write-Host "启动服务:"
Write-Host "  cd $InstallDir"
Write-Host "  uv run face-server run --host 0.0.0.0 --port $Port" -ForegroundColor Cyan
Write-Host ""
Write-Host "或后台运行:"
Write-Host "  Start-Process -FilePath uv -ArgumentList 'run','face-server','run' -WindowStyle Hidden" -ForegroundColor Cyan
Write-Host ""
Write-Host "测试:"
Write-Host "  Invoke-RestMethod http://localhost:$Port/model/info" -ForegroundColor Cyan

# NSSM 服务化 face-cast ui (web UI 端).
# 在 .13 NAS 上跑, 24/7 监听 :9100.

param(
    [string]$InstallDir = "E:\face-cast",
    [int]$Port = 9100,
    [string]$DbPath = "E:\face-cast\face_cast.db",
    [string]$ServiceName = "face-cast-ui"
)

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$ErrorActionPreference = "Stop"

function Find-NSSM {
    $cmd = (Get-Command nssm -ErrorAction SilentlyContinue).Source
    if ($cmd) { return $cmd }
    $candidates = @(
        (Join-Path $InstallDir "bin\nssm.exe"),
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\nssm.exe",
        "C:\ProgramData\chocolatey\bin\nssm.exe"
    )
    $pkg = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Directory -Filter "NSSM.NSSM_*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pkg) {
        $candidates += (Get-ChildItem $pkg.FullName -Recurse -Filter "nssm.exe" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*win64*" } | Select-Object -First 1 -ExpandProperty FullName)
    }
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

function Install-NSSM-Direct {
    # 直接从 nssm.cc 下 zip, 解压 win64 nssm.exe 到 InstallDir\bin
    $binDir = Join-Path $InstallDir "bin"
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
    $zip = Join-Path $env:TEMP "nssm.zip"
    $extracted = Join-Path $env:TEMP "nssm-extracted"
    Write-Host "  下 nssm zip ..."
    Invoke-WebRequest -Uri "https://nssm.cc/release/nssm-2.24.zip" -OutFile $zip -UseBasicParsing
    if (Test-Path $extracted) { Remove-Item -Recurse -Force $extracted }
    Expand-Archive -Path $zip -DestinationPath $extracted -Force
    $exe = Get-ChildItem $extracted -Recurse -Filter "nssm.exe" | Where-Object { $_.FullName -like "*win64*" } | Select-Object -First 1
    if (-not $exe) { throw "nssm.exe (win64) 没找到" }
    Copy-Item $exe.FullName -Destination (Join-Path $binDir "nssm.exe") -Force
    Remove-Item -Recurse -Force $extracted
    Remove-Item -Force $zip
    return (Join-Path $binDir "nssm.exe")
}

$nssm = Find-NSSM
if (-not $nssm) {
    Write-Host "装 NSSM (直接下 zip) ..." -ForegroundColor Yellow
    $nssm = Install-NSSM-Direct
}
if (-not $nssm) { throw "nssm not found" }
Write-Host "nssm: $nssm"

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = "$env:USERPROFILE\.local\bin\uv.exe"
}
if (-not (Test-Path $uv)) { throw "uv not found" }

$logDir = Join-Path $InstallDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 防火墙放行端口
$ruleName = "face-cast UI :$Port"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort $Port -Profile Any | Out-Null
    Write-Host "防火墙规则已加: $ruleName"
}

# 删除已有 (幂等)
$ErrorActionPreference = "Continue"
& $nssm stop $ServiceName 2>&1 | Out-Null
& $nssm remove $ServiceName confirm 2>&1 | Out-Null
$ErrorActionPreference = "Stop"

# 注册服务
& $nssm install $ServiceName $uv "run" "face-cast" "ui" "--db" $DbPath "--host" "0.0.0.0" "--port" $Port
& $nssm set $ServiceName AppDirectory $InstallDir
& $nssm set $ServiceName Description "face-cast Web UI (浏览/合并/拆分 person, 推 Jellyfin)"
& $nssm set $ServiceName Start SERVICE_AUTO_START   # 开机自启 (跟 Jellyfin 一样常驻)
& $nssm set $ServiceName AppStdout (Join-Path $logDir "ui.out.log")
& $nssm set $ServiceName AppStderr (Join-Path $logDir "ui.err.log")
& $nssm set $ServiceName AppRotateFiles 1
& $nssm set $ServiceName AppRotateOnline 1
& $nssm set $ServiceName AppRotateBytes 10485760
& $nssm set $ServiceName AppExit Default Restart
& $nssm set $ServiceName AppRestartDelay 5000
& $nssm set $ServiceName AppEnvironmentExtra "PYTHONIOENCODING=utf-8"

Write-Host ""
Write-Host "═══ 注册完成 ═══" -ForegroundColor Green
Write-Host ""
Write-Host "服务: $ServiceName"
Write-Host "网址: http://localhost:$Port  (LAN: http://10.100.100.13:$Port)"
Write-Host "日志: $logDir\ui.{out,err}.log"
Write-Host ""
& $nssm start $ServiceName
Start-Sleep -Seconds 5
& $nssm status $ServiceName

try {
    $r = Invoke-WebRequest -Uri "http://localhost:$Port/" -TimeoutSec 5 -UseBasicParsing
    Write-Host ""
    Write-Host "✓ UI HTTP $($r.StatusCode) ($($r.Content.Length) bytes)" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "探活失败: $_" -ForegroundColor Red
    Get-Content (Join-Path $logDir "ui.err.log") -Tail 10 -ErrorAction SilentlyContinue
}

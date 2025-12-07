# ===========================
# Electrodrive BEM Runner (strict governance, GPU-aware, with log tailing)
# Full script – save as run_bem_env.ps1
# ===========================
param(
    [string]$RepoRoot = ".",
    [string]$VenvPath = ".\.venv",
    [string]$Mode = "bem",
    [string]$OutRoot = ".\run_out",
    [switch]$SkipSphere,

    # Gates
    [double]$ThresholdBc = 1e-7,
    [double]$ThresholdDual = 1e-8,
    [double]$ThresholdPde = 1e-6,
    [double]$ThresholdEnergy = 1e-6,
    [int]$MaxGmresProgress = 250,

    # Governance inputs (can be omitted; auto-detects hw7.pdf & checksum)
    [string]$EvalPdf = $null,
    [string]$EvalSha256 = $null
)

# Your expected HW7 checksum
$EMBEDDED_EXPECTED_SHA = "727DA2B8061C1F37B8A9938716D7822456356F106B42C1CD0F87114D0530BE58"

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Info($m){ Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Write-Ok($m){ Write-Host "[ OK ]  $m" -ForegroundColor Green }
function Write-Warn($m){ Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Write-Err($m){ Write-Host "[FAIL]  $m" -ForegroundColor Red }

function Test-Exe([string]$Path){ if(-not (Test-Path $Path)){ throw "Executable not found: $Path" } }
function Get-PythonExe([string]$Venv){ $py = Join-Path $Venv "Scripts\python.exe"; Test-Exe $py; return $py }

function New-CaseOutDir([string]$OutRoot, [string]$Name){
  $ts = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
  $dir = Join-Path $OutRoot "$($Name)_$ts"
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  return $dir
}

function New-PlaneSpec([string]$Path){
  $spec = @{
    domain="R3"
    conductors=@(@{type="plane"; z=0; potential=0})
    dielectrics=@()
    charges=@(@{type="point"; q=1e-9; pos=@(0,0,0.05)})
    BCs="Dirichlet"
    symmetry=@("axial")
    queries=@("potential","field")
    symbols=@{d="length"; q="charge"}
  } | ConvertTo-Json -Depth 6
  Set-Content -Path $Path -Encoding UTF8 -Value $spec
}

function New-SphereSpec([string]$Path){
  $spec = @{
    domain="R3"
    conductors=@(@{type="sphere"; center=@(0,0,0); radius=1.0; potential=0})
    dielectrics=@()
    charges=@(@{type="point"; q=1e-9; pos=@(0.0,0.0,0.3)})
    BCs="Dirichlet"
    symmetry=@("axial")
    queries=@("potential","field","force_on_charge")
    symbols=@{a="length"; d="length"; q="charge"}
  } | ConvertTo-Json -Depth 6
  Set-Content -Path $Path -Encoding UTF8 -Value $spec
}

function Sha256([string]$Path){
  if(-not (Test-Path $Path)){ throw "File not found for SHA-256: $Path" }
  (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToUpperInvariant()
}

function Get-CLIHelp([string]$Py){ try { & $Py -m electrodrive.cli solve -h 2>&1 | Out-String } catch { "" } }

function Detect-GovMode([string]$Help){
  if($Help -match "--eval_sha256"){ return "flag_underscore" }
  if($Help -match "--eval-sha256"){ return "flag_dash" }
  if($Help -match "(?m)^\s+eval_sha256\b"){ return "positional_underscore" }
  if($Help -match "(?m)^\s+eval-sha256\b"){ return "positional_dash" }
  return "unknown"
}

function Build-GovArgs([string]$Mode, [string]$PdfPath, [string]$Hash){
  $args = @()
  if($PdfPath){ $args += @("--eval-pdf",$PdfPath) }
  switch($Mode){
    "flag_underscore"   { $args += @("--eval_sha256",$Hash) }
    "flag_dash"         { $args += @("--eval-sha256",$Hash) }
    "positional_underscore" { $args += @($Hash) }
    "positional_dash"       { $args += @($Hash) }
    default             { }
  }
  return ,$args
}

function Show-RecentLogs([string]$OutDir){
  $hp = Join-Path $OutDir "human.log"
  if(Test-Path $hp){
    Write-Info "----- tail: human.log (${hp}) -----"
    try { Get-Content $hp -Tail 120 | ForEach-Object { Write-Host "       $_" } } catch {}
  } else {
    Write-Warn "human.log not found in $OutDir"
  }
}

function Read-Metrics([string]$OutDir){
  $mp = Join-Path $OutDir "metrics.json"
  if(-not (Test-Path $mp)){ return $null }
  try { Get-Content $mp -Raw | ConvertFrom-Json } catch { $null }
}

function Count-Gmres([string]$OutDir){
  $hp = Join-Path $OutDir "human.log"
  if(-not (Test-Path $hp)){ return 0 }
  ((Select-String -Path $hp -Pattern "GMRES progress") | Measure-Object).Count
}

function Get-Metric($obj, [string]$name){
  if($null -eq $obj){ return $null }
  $p = $obj.PSObject.Properties[$name]
  if($null -eq $p){ return $null }
  $p.Value
}

function Test-Metric([string]$Name, $Value, [double]$Threshold){
  if($null -eq $Value){ Write-Err "$Name missing"; return $false }
  if(($Value -is [double]) -and ([Double]::IsNaN($Value) -or [Double]::IsInfinity($Value))){
    if($Name -eq "energy_rel_diff"){ Write-Warn "$Name not computed (NaN)"; return $false }
    Write-Err "$Name is NaN/Inf ($Value)"; return $false
  }
  if([double]$Value -le $Threshold){ Write-Ok ("{0} = {1:E3} <= {2:E3}" -f $Name,[double]$Value,$Threshold); return $true }
  Write-Err ("{0} = {1:E3} > {2:E3}" -f $Name,[double]$Value,$Threshold); return $false
}

function Invoke-EDE(
  [string]$Py, [string]$RepoRoot, [string]$SpecPath, [string]$Mode, [string]$OutDir,
  [string[]]$GovArgs, [string]$GovLabel
){
  Push-Location $RepoRoot
  try {
    # Encourage GPU (harmless if solver ignores)
    $env:EDE_USE_GPU = "1"
    if (-not $env:EDE_TILE_SIZE -or $env:EDE_TILE_SIZE -eq "") { $env:EDE_TILE_SIZE = "0" }

    $base = @(
      "-X","dev","-u","-m","electrodrive.cli","solve",
      "--problem",(Resolve-Path $SpecPath).ProviderPath,
      "--mode",$Mode,
      "--out",(Resolve-Path $OutDir).ProviderPath,
      "--cert"
    )
    if($GovArgs){ $base += $GovArgs }

    Write-Info ("Invoking EDE ({0}): python {1}" -f $GovLabel, ($base -join " "))
    & $Py @base
    return $LASTEXITCODE
  } finally { Pop-Location }
}

# ---------------- MAIN ----------------
$repo = (Resolve-Path $RepoRoot).ProviderPath
$resolvedOutRoot = Resolve-Path $OutRoot -ErrorAction SilentlyContinue
if(-not $resolvedOutRoot){
  Write-Info "Creating output directory: $OutRoot"
  $resolvedOutRoot = New-Item -ItemType Directory -Force -Path $OutRoot | Select-Object -ExpandProperty FullName
}

$py = Get-PythonExe $VenvPath
Write-Info "Using Python: $py"
Write-Info "Repo root: $repo"
Write-Info "Out root: $resolvedOutRoot"

# Locate hw7.pdf if not provided
if(-not $EvalPdf){
  $cand = Join-Path $repo "hw7.pdf"
  if(Test-Path $cand){ $EvalPdf = $cand; Write-Info "No --EvalPdf passed; found hw7.pdf in repo root. Using: $EvalPdf" }
}
# Calculate actual SHA if we have a PDF
$actualSha = $null
if($EvalPdf){
  $actualSha = Sha256 $EvalPdf
  Write-Info "Computed SHA256 for EvalPdf: $actualSha"
}

# Decide expected SHA
$expectedSha = if($EvalSha256){ $EvalSha256 } elseif($actualSha){ $EMBEDDED_EXPECTED_SHA } else { $null }
if($actualSha -and $expectedSha){
  if($actualSha.ToUpperInvariant() -eq $expectedSha.ToUpperInvariant()){ Write-Ok "EvalPdf hash matches supplied/expected SHA." }
  else { Write-Warn "EvalPdf hash does not match expected SHA." }
}

# Probe CLI help (to pick a first governance mode), then prepare alternates for retry
$help = Get-CLIHelp $py
$primary = Detect-GovMode $help
switch($primary){
  "flag_underscore"   { Write-Info "CLI supports: --eval-pdf / --eval_sha256" }
  "flag_dash"         { Write-Info "CLI supports: --eval-pdf / --eval-sha256" }
  "positional_underscore" { Write-Info "CLI expects positional governance hash: eval_sha256" }
  "positional_dash"       { Write-Info "CLI expects positional governance hash: eval-sha256" }
  default             { Write-Warn "CLI help did not expose governance flags; will try common modes." }
}

# Candidate governance modes to try (primary first, then fallbacks)
$modes = @()
if($primary -ne "unknown"){ $modes += $primary }
$modes += @("flag_dash","flag_underscore","positional_dash","positional_underscore") | Where-Object { $_ -ne $primary } | Select-Object -Unique

# Build gov args for each mode (only if we have both pieces)
$govMatrix = @()
foreach($m in $modes){
  $args = @()
  if($EvalPdf -and $expectedSha){ $args = Build-GovArgs -Mode $m -PdfPath (Resolve-Path $EvalPdf).ProviderPath -Hash $expectedSha }
  $govMatrix += ,@($m, $args)
}

$overallPass = $true
$results = @()

# Helper: run one case with auto-retry across governance modes
function Run-Case([string]$CaseName, [scriptblock]$SpecBuilder){
  $spec = Join-Path $env:TEMP ("ede_{0}_spec.json" -f $CaseName)
  & $SpecBuilder $spec
  $outDir = New-CaseOutDir $resolvedOutRoot $CaseName

  $exit = $null
  $usedMode = $null
  $usedArgs = $null

  foreach($pair in $govMatrix){
    $modeName = $pair[0]; $gargs = $pair[1]
    $label = if($gargs -and $gargs.Count) { $modeName } else { "$modeName (no-args)" }
    $exit = Invoke-EDE -Py $py -RepoRoot $repo -SpecPath $spec -Mode $Mode -OutDir $outDir -GovArgs $gargs -GovLabel $label
    $usedMode = $modeName; $usedArgs = $gargs

    # If exit==0, or metrics file exists, stop retrying
    $mx = Read-Metrics $outDir
    if(($exit -eq 0) -or ($mx -ne $null)){ break }
    Write-Warn "Retrying with next governance style..."
  }

  $metrics = Read-Metrics $outDir
  $gmres = Count-Gmres $outDir

  # Strict: fail if CLI non-zero OR no metrics OR no metrics.metrics
  $passed = $false
  if(($exit -ne 0) -or ($metrics -eq $null) -or (-not $metrics.metrics)){
    Write-Err ("{0} run did not produce usable metrics. (exit={1}, metrics.json present={2}, metrics.metrics present={3})" -f $CaseName, $exit, ($metrics -ne $null), ($metrics -ne $null -and $metrics.metrics -ne $null))
    Show-RecentLogs $outDir
    $passed = $false
  } else {
    $bc  = Get-Metric $metrics.metrics 'bc_residual_linf'
    $dual= Get-Metric $metrics.metrics 'dual_route_l2_boundary'
    $pde = Get-Metric $metrics.metrics 'pde_residual_linf'
    $en  = Get-Metric $metrics.metrics 'energy_rel_diff'

    Write-Info "Evaluating metrics ($CaseName)"
    $ok1 = Test-Metric "bc_residual_linf" $bc $ThresholdBc
    $ok2 = Test-Metric "dual_route_l2_boundary" $dual $ThresholdDual
    $ok3 = Test-Metric "pde_residual_linf" $pde $ThresholdPde
    if($gmres -le $MaxGmresProgress){ Write-Ok "GMRES progress lines: $gmres <= $MaxGmresProgress" } else { Write-Err "GMRES progress lines: $gmres > $MaxGmresProgress" }
    $ok4 = ($gmres -le $MaxGmresProgress)
    $ok5 = Test-Metric "energy_rel_diff" $en $ThresholdEnergy
    $passed = $ok1 -and $ok2 -and $ok3 -and $ok4 -and $ok5
    if(-not $passed){ Show-RecentLogs $outDir }
  }

  $results += [pscustomobject]@{
    Case    = $CaseName
    OutDir  = $outDir
    Pass    = $passed
    CLIExit = $exit
    GovMode = $usedMode
    GMRES   = $gmres
  }
  return $passed
}

# ---- Execute cases ----
$pPass = Run-Case -CaseName "plane" -SpecBuilder ${function:New-PlaneSpec}
$sPass = $true
if(-not $SkipSphere){
  $sPass = Run-Case -CaseName "sphere" -SpecBuilder ${function:New-SphereSpec}
}

$overallPass = $pPass -and $sPass

"`n==== Summary ====" | Write-Host -ForegroundColor DarkCyan
$results | Format-Table -AutoSize | Out-Host

if($overallPass){ Write-Ok "All checks passed."; exit 0 } else { Write-Err "One or more checks failed."; exit 2 }

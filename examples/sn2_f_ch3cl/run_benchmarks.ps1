# Benchmarks along the AIMNet2 IRC of F- + CH3Cl -> CH3F + Cl- (charge -1).
# Run from the repository root with SAMSON's Python; outputs go to $Work.
param(
    [string]$Work = "D:\MLIP_Work_Folder\sn2_F_CH3Cl",
    [string]$Python = "$env:USERPROFILE\OneAngstrom\SAMSON-Application\11.0.1\Binaries\python.exe",
    [string]$Aimnet = "D:\MLIP_Work_Folder\envs\mlip\python.exe",
    [string]$AimnetModel = "D:\MLIP_Work_Folder\cache\aimnet\aimnet2_wb97m_d3_0.pt",
    [string]$Mace = "D:\MLIP_Work_Folder\cache\mace\20231210mace128L0_energy_epoch249model",
    [int]$Points = 30
)
$irc = Join-Path $Work "aimnet2_irc.extxyz"
$common = @("-m", "samson_mlip_visualizer.benchmark", $irc, "--points", $Points, "--charge", "-1",
            "--ends", "F-...CH3Cl,FCH3...Cl-", "--aimnet-model", $AimnetModel)
$runs = @{
    "aimnet2_vs_pbe"   = @("--backend", "aimnet2", "--model", $Aimnet, "--reference", "psi4",
                           "--psi4-method", "pbe", "--basis", "def2-tzvpd")
    "mace_vs_pbe"      = @("--backend", "mace", "--model", $Mace, "--reference", "psi4",
                           "--psi4-method", "pbe", "--basis", "def2-tzvpd")
    "xtb_vs_pbe"       = @("--backend", "xtb", "--model", "auto", "--reference", "psi4",
                           "--psi4-method", "pbe", "--basis", "def2-tzvpd")
    "aimnet2_vs_wb97m" = @("--backend", "aimnet2", "--model", $Aimnet, "--reference", "psi4",
                           "--psi4-method", "wb97m-d3bj", "--basis", "def2-tzvppd")
}
$jobs = foreach ($name in $runs.Keys) {
    $out = Join-Path $Work $name
    $args = $common + $runs[$name] + @("-o", $out)
    Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -PassThru `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
}
$jobs | Wait-Process
foreach ($name in $runs.Keys) { "== $name"; Get-Content (Join-Path $Work "$name.log") -Tail 12 }

# escape=`
FROM mcr.microsoft.com/windows/servercore:ltsc2025

SHELL ["powershell","-NoLogo","-NoProfile","-Command"]

# 1) Chocolatey
RUN Set-ExecutionPolicy Bypass -Scope Process -Force; `
    [System.Net.ServicePointManager]::SecurityProtocol = `
        [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1')); `
    choco feature enable -n=usePackageExitCodes; `
    choco feature enable -n=allowGlobalConfirmation;

# 2) Toolchain & utils
RUN choco install -y git;

# 3) Visual Studio Build Tools + components
RUN choco install visualstudio2022buildtools -y `
      --package-parameters '--add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --add Microsoft.VisualStudio.Component.VC.CMake.Project --includeRecommended';

# 4) Add MSBuild and cl.exe to PATH (single RUN!)
RUN $msbuildToolsPath = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin'; `
    $clexePath = Get-ChildItem 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64' | Select-Object -First 1 -ExpandProperty FullName; `
    $currentPath = [Environment]::GetEnvironmentVariable('Path', 'Machine'); `
    [Environment]::SetEnvironmentVariable('Path', \"$currentPath;$msbuildToolsPath;$clexePath\", 'Machine'); `
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine'); `
    msbuild -version; `
    cl.exe;
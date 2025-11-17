# escape=`
FROM mcr.microsoft.com/dotnet/framework/runtime:4.8-windowsservercore-ltsc2019

SHELL ["powershell","-NoLogo","-NoProfile","-Command","$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';"]

# 1) Chocolatey (4.8 is already present in this base image)
RUN Set-ExecutionPolicy Bypass -Scope Process -Force; `
    [System.Net.ServicePointManager]::SecurityProtocol = `
      [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1')); `
    choco feature enable -n=usePackageExitCodes; `
    choco feature enable -n=allowGlobalConfirmation

# 2) Toolchain & utils  (NOTE: no backtick on the last line)
RUN choco install -y git; `
    choco install -y cmake --installargs '"ADD_CMAKE_TO_PATH=System"';

RUN choco install -y ninja;

# 3) PATH fixups (single line to avoid parse errors)
RUN $p = [Environment]::GetEnvironmentVariable('Path','Machine'); `
    $extra = 'C:\Program Files\CMake\bin;C:\ProgramData\chocolatey\lib\ninja\tools'; `
    [Environment]::SetEnvironmentVariable('Path', $p + ';' + $extra, 'Machine')


# Optional sanity checks
RUN cmake --version; ctest --version; 
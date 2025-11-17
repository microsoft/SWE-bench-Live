# escape=`
FROM karinali20011210/windows_server:ltsc2025_vsbuildtools

SHELL ["powershell","-NoLogo","-NoProfile","-Command"]

# 1) Chocolatey
RUN choco install -y cmake --installargs '"ADD_CMAKE_TO_PATH=System"';

RUN choco install -y ninja;

RUN $p = [Environment]::GetEnvironmentVariable('Path','Machine'); `
    $extra = 'C:\Program Files\CMake\bin;C:\ProgramData\chocolatey\lib\ninja\tools'; `
    [Environment]::SetEnvironmentVariable('Path', $p + ';' + $extra, 'Machine');

RUN cmake --version; ninja --version;  git --version;
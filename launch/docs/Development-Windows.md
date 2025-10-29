# Development on Windows

## To run on windows, you need to

(1) Download docker desktop on windows.

You need to turn on virtualization support on windows in BIOS when starting your computer.

(2) If you want to build repo on windows image, switch to Windows containers in docker settings.

(3) Enable long path in git. Open windows powershell as administrator,

```powershell
git config --global core.longpaths true
```

(4) Enable long path in Windows system.

and you can now run all the steps in the pipeline.

## Windows Notes
The way to export environment variables in windows is

```powershell
$env:OPENAI_API_KEY="your_key"
```
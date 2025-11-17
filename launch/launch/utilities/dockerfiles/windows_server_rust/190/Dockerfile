# escape=`
FROM karinali20011210/windows_server:ltsc2025_vsbuildtools

SHELL ["powershell","-NoLogo","-NoProfile","-Command","$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';"]

# Install Rust
RUN Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "rustup-init.exe"; `
	./rustup-init.exe -y --default-toolchain 1.90.0; `
	Remove-Item rustup-init.exe;

RUN $cargoPath = 'C:\Users\ContainerAdministrator\.cargo\bin'; `
	setx PATH \"$env:PATH;$cargoPath\"; `
	$env:PATH = \"$env:PATH;$cargoPath\"; `
	rustc --version; `
	cargo --version

# Create, build, and run a test Rust project, then clean up
RUN cargo new cargotest; `
	cd cargotest; `
	cargo build; `
	cargo run; `
	cd ..; `
	Remove-Item -Recurse -Force cargotest;
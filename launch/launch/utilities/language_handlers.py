from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from launch.core.runtime import SetupRuntime
from launch.utilities.timemachine import start_timemachine


class LanguageHandler(ABC):
    """Abstract base class for language-specific setup handlers."""
    
    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language name."""
        pass
    
    @abstractmethod
    def base_images(self, platform = "linux") -> List[str]:
        """Return candidate base Docker images for this language."""
        pass
    
    @abstractmethod
    def setup_environment(self, session: SetupRuntime, date: Optional[str] = None) -> Optional[Any]:
        pass
    
    @abstractmethod
    def get_setup_instructions(self, base_image: str, platform: str = "linux") -> str:
        """Get language-specific setup instructions for the agent."""
        pass
    
    @abstractmethod
    def cleanup_environment(self, session: SetupRuntime, server: Optional[Any] = None):
        """Cleanup language-specific resources."""
        pass


class PythonHandler(LanguageHandler):
    """Handler for Python projects."""
    
    @property
    def language(self) -> str:
        return "python"
    
    def base_images(self, platform = "linux") -> List[str]:
        return [f"python:3.{v}" for v in range(6, 12)]
    
    def setup_environment(self, session: SetupRuntime, date: Optional[str] = None) -> Optional[Any]:
        """Setup Python environment with optional timemachine."""
        if not date:
            return None
        
        # Start pypi-timemachine server for historical package resolution
        return start_timemachine(session, date)
    
    def get_setup_instructions(self, base_image: str, platform: str = "linux") -> str:
        return """
### Python-Specific Instructions:
- Make sure the package is installed from source in editable mode before running tests (e.g., `pip install -e .`)
- Avoid using tox to run tests if possible as it's designed for CI. Read tox.ini to understand setup
- Check requirements.txt, setup.py, or pyproject.toml for dependencies
"""
    
    def cleanup_environment(self, session: SetupRuntime, server: Optional[Any] = None):
        """Cleanup Python environment."""
        if server:
            try:
                session.send_command("pip config unset global.index-url")
                session.send_command("pip config unset global.trusted-host")
                server.stop()
            except Exception:
                pass


class JavaScriptHandler(LanguageHandler):
    """Handler for JavaScript/Node.js projects."""
    
    @property
    def language(self) -> str:
        return "javascript"
    
    def base_images(self, platform = "linux") -> List[str]:
        return [f"node:{v}" for v in ["18", "20", "22"]]
    
    def setup_environment(self, session: SetupRuntime, date: Optional[str] = None) -> Optional[Any]:
        """Setup Node.js environment."""
        # No special server needed for JavaScript
        return None
    
    def get_setup_instructions(self, base_image: str, platform: str = "linux") -> str:
        return """
### JavaScript/Node.js-Specific Instructions:
- Use npm, yarn, or pnpm to install dependencies (check package.json and lockfiles)
- Run `npm install` or `yarn install` to install dependencies
- Check package.json for test scripts and build commands
- Consider using `npm ci` for faster, reproducible builds if package-lock.json exists
- Install global dependencies if needed (e.g., `npm install -g typescript`)
"""
    
    def cleanup_environment(self, session: SetupRuntime, server: Optional[Any] = None):
        """Cleanup JavaScript environment."""
        # No special cleanup needed for JavaScript
        pass


class TypeScriptHandler(JavaScriptHandler):
    """Handler for TypeScript projects (inherits from JavaScript)."""
    
    @property
    def language(self) -> str:
        return "typescript"
    
    def get_setup_instructions(self, base_image: str, platform: str = "linux") -> str:
        return """
### TypeScript-Specific Instructions:
- Use npm, yarn, or pnpm to install dependencies (check package.json and lockfiles)
- Run `npm install` or `yarn install` to install dependencies
- Check package.json for test scripts and build commands
- Consider using `npm ci` for faster, reproducible builds if package-lock.json exists
- Install TypeScript globally if needed (e.g., `npm install -g typescript`)
"""


class RustHandler(LanguageHandler):
    """Handler for Rust projects."""
    
    @property
    def language(self) -> str:
        return "rust"
    
    def base_images(self, platform = "linux") -> List[str]:
        return [f"rust:1.7{v}" for v in range(0, 10)]
    
    def setup_environment(self, session: SetupRuntime, date: Optional[str] = None) -> Optional[Any]:
        """Setup Rust environment."""
        # No special server needed for Rust
        return None
    
    def get_setup_instructions(self, base_image: str, platform: str = "linux") -> str:
        return """
### Rust-Specific Instructions:
- Use `cargo build` to build the project
- Use `cargo test` to run tests
- Use `cargo check` for faster compilation checks
- Install system dependencies if needed (check Cargo.toml for sys crates)
- Consider using `cargo install` for binary dependencies
"""
    
    def cleanup_environment(self, session: SetupRuntime, server: Optional[Any] = None):
        """Cleanup Rust environment."""
        # No special cleanup needed for Rust
        pass


class JavaHandler(LanguageHandler):
    """Handler for Java projects."""
    
    @property
    def language(self) -> str:
        return "java"
    
    def base_images(self, platform = "linux") -> List[str]:
        if platform == "linux":
            return [f"eclipse-temurin:{v}-jdk-noble" for v in ["11", "17", "21"]]
        if platform == "windows":
            return [f"eclipse-temurin:{v}-jdk-windowsservercore-ltsc2022" for v in ["11", "17", "21"]]
    
    def setup_environment(self, session: SetupRuntime, date: Optional[str] = None) -> Optional[Any]:
        """Setup Java environment."""
        # No special server needed for Java
        return None
    
    def get_setup_instructions(self, base_image: str, platform: str = "linux") -> str:
        return """
### Java-Specific Instructions:
- Use Maven (`mvn test`) or Gradle (`gradle test`) to run tests
- Use `mvn install` or `gradle build` to build the project
- Check pom.xml (Maven) or build.gradle (Gradle) for dependencies
- Install system dependencies if needed
- Use `mvn dependency:resolve` to download dependencies
"""
    
    def cleanup_environment(self, session: SetupRuntime, server: Optional[Any] = None):
        """Cleanup Java environment."""
        # No special cleanup needed for Java
        pass


class GoHandler(LanguageHandler):
    """Handler for Go projects."""
    
    @property
    def language(self) -> str:
        return "go"
    
    def base_images(self, platform = "linux") -> List[str]:
        if platform == "linux":
            return [f"golang:1.{v}" for v in ["19", "20", "21", "22", "23", "24", "25"]]
        if platform == "windows":
            return [f"golang:1.{v}" for v in ["19.0-windowsservercore", 
                                              "20.0-windowsservercore", 
                                              "21.0-windowsservercore", 
                                              "22.0-windowsservercore",
                                              "23.0-windowsservercore",
                                              "24.0-windowsservercore",
                                              "25.0-windowsservercore"]]
    
    def setup_environment(self, session: SetupRuntime, date: Optional[str] = None) -> Optional[Any]:
        """Setup Go environment."""
        # No special server needed for Go
        return None
    
    def get_setup_instructions(self, base_image: str, platform: str = "linux") -> str:
        return """
### Go-Specific Instructions:
- Use `go mod download` to download dependencies
- Use `go test ./...` to run all tests
- Use `go build` to build the project
- Check go.mod for module dependencies
- Use `go get` to install missing dependencies
"""
    
    def cleanup_environment(self, session: SetupRuntime, server: Optional[Any] = None):
        """Cleanup Go environment."""
        # No special cleanup needed for Go
        pass


class CSharpHandler(LanguageHandler):
    """Handler for C# projects."""
    
    @property
    def language(self) -> str:
        return "csharp"
    
    def base_images(self, platform = "linux") -> List[str]:
        if platform == "linux":
            return [f"mcr.microsoft.com/dotnet/sdk:{v}" for v in ["6.0", "7.0", "8.0", "9.0"]]
        elif platform == "windows":
            return [f"mcr.microsoft.com/dotnet/sdk:{v}" for v in [
                "9.0-windowsservercore-ltsc2022",
                "8.0-windowsservercore-ltsc2022",
                "9.0-windowsservercore-ltsc2019",
                "8.0-windowsservercore-ltsc2019",
            ]]
    
    def setup_environment(self, session: SetupRuntime, date: Optional[str] = None) -> Optional[Any]:
        """Setup C# environment."""
        # No special server needed for C#
        return None
    
    def get_setup_instructions(self, base_image: str, platform: str = "linux") -> str:
        return """
### C#-Specific Instructions:
- Use `dotnet restore` to restore NuGet packages
- Use `dotnet build` to build the project
- Use `dotnet test` to run tests
- Check .csproj or .sln files for project configuration
- Use `dotnet run` to run the application
- Consider using `dotnet publish` for deployment builds
"""
    
    def cleanup_environment(self, session: SetupRuntime, server: Optional[Any] = None):
        """Cleanup C# environment."""
        # No special cleanup needed for C#
        pass



class CppHandler(LanguageHandler):
    """Handler for C++ projects."""
    
    @property
    def language(self) -> str:
        return "c++"
    
    def base_images(self, platform = "linux") -> List[str]:
        if platform == "linux": 
            return [
                f"mcr.microsoft.com/devcontainers/cpp:{v}"
                for v in [
                    "1-ubuntu-20.04",
                    "1-ubuntu-22.04",
                    "1-ubuntu-24.04"
                ]
            ]
        if platform == "windows":
            return [
                "karinali20011210/windows_server:ltsc2019",
                "karinali20011210/windows_server:ltsc2022"
            ]
    
    def setup_environment(self, session: SetupRuntime, date: Optional[str] = None) -> Optional[Any]:
        """Setup C/C++ environment."""
        # No special server needed for C#
        return None
    
    def get_setup_instructions(self, base_image: str, platform: str = "linux") -> str:
        if platform == "linux":
            return """
### C/C++ Specific Instructions:
- Verify tools: 
  - `cl ; gcc --version ; g++ --version ; clang --version ; cmake --version ; ctest --version ; ninja --version`
- Configure with CMake:
  - `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=23`
    - Use 20/17/11 if your project requires it
    - Force a compiler if needed:
      - GCC: `-DCMAKE_CXX_COMPILER=g++`
      - Clang: `-DCMAKE_CXX_COMPILER=clang++`
- Build the project:
  - `cmake --build build --parallel`
- Run tests:
  - `ctest --test-dir build --output-on-failure`
- Dependencies:
  - `vcpkg` is available; integrate via `-DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg.cmake` or your preset.
    (Conan works too if present in your repo.)
- Run the app:
  - `./build/<target_name>`
- For other c/cpp repository variants not covered, decide how to build the repository yourself.
"""
        if platform == "windows":
            return r"""
### C/C++ Specific Instructions:
This is a minimal windows server image with only git, choco, cmake and ninja installed.
You need to figure out how to install the required dependencies yourself. You can use web search if you are not sure.
choco is the preferred way to install packages. For example, choco install -y mingw, choco install -y llvm
Some dependencies may not be supported by choco and have to be installed through its official source. For example, git clone https://github.com/microsoft/vcpkg.git; .\vcpkg\bootstrap-vcpkg.bat; $env:VCPKG_ROOT = "C:\path\to\vcpkg"; $env:PATH = "$env:VCPKG_ROOT;$env:PATH";

Some dependency install examples using choco:

-- MSVC dependencies: 
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --add Microsoft.VisualStudio.Component.VC.CMake.Project --includeRecommended"
# Add MSBuild to PATH (as mentioned in gist.github.com)
$buildToolsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if ($currentPath -notlike "*$buildToolsPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$buildToolsPath", "Machine")
}

# Refresh environment variables in current session
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine")

# Verify cl.exe is available
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ cl.exe found and working"
} else {
    Write-Host "❌ cl.exe not found or not working"
}

# Verify MSBuild
msbuild -version

-- Qt with MSVC: 
(install MSVC as above)
choco install -y aqt --no-progress; refreshenv; 
aqt install-qt --outputdir C:\Qt windows desktop 6.8.0 win64_msvc2019_64;
aqt install-tool --outputdir C:\Qt windows desktop tools_qtcreator;
aqt install-tool --outputdir C:\Qt windows desktop tools_cmake;

-- Qt with MinGW: 
choco install -y qt6-base-dev mingw; refreshenv;

The installed packages often do not know the existence of each other. You need to link them manually if errors occur.

Examples to build a repo:
- Configure with CMake:
  - `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=23`
    - Use 20/17/11 if your project requires it
    - Force a compiler if needed:
      - GCC: `-DCMAKE_CXX_COMPILER=g++`
      - Clang: `-DCMAKE_CXX_COMPILER=clang++`
- Build the project:
  - `cmake --build build --parallel`
- Run tests:
  - `ctest --test-dir build --output-on-failure`
- Run the app:
  - `./build/<target_name>`
"""
    
    def cleanup_environment(self, session: SetupRuntime, server: Optional[Any] = None):
        """Cleanup C/C++ environment."""
        # No special cleanup needed for C++
        pass


class CHandler(CppHandler):
    @property
    def language(self) -> str:
        return "c"


LANGUAGE_HANDLERS: Dict[str, LanguageHandler] = {
    "python": PythonHandler(),
    "javascript": JavaScriptHandler(),
    "typescript": TypeScriptHandler(),
    "rust": RustHandler(),
    "java": JavaHandler(),
    "go": GoHandler(),
    "c#": CSharpHandler(),
    "c++":  CppHandler(),
    "c": CHandler(),
}


def get_language_handler(language: str) -> LanguageHandler:
    if language not in LANGUAGE_HANDLERS:
        raise ValueError(f"Language '{language}' is not supported. Available languages: {list(LANGUAGE_HANDLERS.keys())}")
    
    return LANGUAGE_HANDLERS[language]


def get_supported_languages() -> List[str]:
    """Get list of supported programming languages."""
    return list(LANGUAGE_HANDLERS.keys())

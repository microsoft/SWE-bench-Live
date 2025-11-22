"""
Docker runtime management for repository setup and command execution.

Provides containerized environment for repository testing with command execution,
file operations, and state management capabilities.
"""

from __future__ import annotations
import io
import json
import os
import queue
import re
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import docker
from docker.models.containers import Container
from typing_extensions import Self

CMD_OUTPUT_PS1_BEGIN = "\n###PS1JSON###\n"
CMD_OUTPUT_PS1_END = "\n###PS1END###"
CMD_OUTPUT_METADATA_PS1_REGEX = re.compile(
    r"(?m)^\s*" + re.escape(CMD_OUTPUT_PS1_BEGIN.strip()) + r"\s*(.*?)\s*" + re.escape(CMD_OUTPUT_PS1_END.strip()),
    re.DOTALL
)
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

TIMEOUT_EXIT_CODE = 124

MEM_LIMIT = "8g"
CPU_CORES = 4


VAR_PATTERNS = {
    'exit_code': re.compile(r'"exit_code":\s*(-?\d+)\s*(?:,|\})'),
    'username': re.compile(r'"username":\s*"([^"]*)"'),
    'hostname': re.compile(r'"hostname":\s*"([^"]*)"'),
    'working_dir': re.compile(r'"working_dir":\s*"([^"]*)"'),
    'py_interpreter_path': re.compile(r'"py_interpreter_path":\s*"([^"]*)"'),
}

class RegWrapper:
    def __init__(self, reg: str):
        self.reg = reg
    def group(self, idx=0):
        return self.reg

@dataclass
class CmdOutputMetadata:
    """
    Additional metadata captured from PS1 shell prompt.
    
    Provides context about command execution environment including
    exit codes, user info, working directory, and Python interpreter.
    """

    exit_code: int = -1
    username: str | None = None
    hostname: str | None = None
    working_dir: str | None = None
    py_interpreter_path: str | None = None

    @classmethod
    def to_ps1_prompt(cls) -> str:
        """
        Convert metadata requirements into a PS1 prompt string.
        
        Returns:
            str: PS1 prompt configuration for capturing metadata
        """
        prompt = CMD_OUTPUT_PS1_BEGIN
        json_str = json.dumps(
            {
                "exit_code": "$?",
                "username": r"\u",
                "hostname": r"\h",
                "working_dir": r"$(pwd)",
                "py_interpreter_path": r'$(which python 2>/dev/null || echo "")',
            },
            indent=2,
        )
        # Make sure we escape double quotes in the JSON string
        # So that PS1 will keep them as part of the output
        prompt += json_str.replace('"', r"\"")
        prompt += CMD_OUTPUT_PS1_END + "\n"  # Ensure there's a newline at the end
        return prompt

    @classmethod
    def matches_ps1_metadata(cls, output: str) -> list[re.Match[str]]:
        matches = []
        for match in CMD_OUTPUT_METADATA_PS1_REGEX.finditer(output):
            scope = match.group(1).strip()
            try:
                d = json.loads(scope)  # Try to parse as JSON
                matches.append(match)
            except json.JSONDecodeError:
                d = cls.best_effort_match(scope)
                if len(d) > 0:
                    matches.append(match)
        return matches
    
    @classmethod
    def best_effort_match(cls, scope: str) -> dict:
        out = {}
        for field, pattern in VAR_PATTERNS.items():
            m = pattern.search(scope)
            if m:
                out[field] = m.group(1)
            else:
                out[field] = ""
        return out

    @classmethod
    def from_ps1_match(cls, match: re.Match[str]) -> Self:
        """
        Extract metadata from a PS1 prompt regex match.
        
        Args:
            match (re.Match[str]): Regex match containing JSON metadata
            
        Returns:
            Self: CmdOutputMetadata instance with parsed values
        """
        try:
            metadata = json.loads(match.group(1)) 
        except:
            metadata = cls.best_effort_match(match.group(1))
        # Create a copy of metadata to avoid modifying the original
        processed = metadata.copy()
        # Convert numeric fields
        if "exit_code" in metadata:
            try:
                processed["exit_code"] = int(float(str(metadata["exit_code"])))
            except (ValueError, TypeError):
                processed["exit_code"] = -1
        return cls(**processed)


@dataclass
class CommandResult:
    """
    Result of a command execution with output and metadata.
    
    Attributes:
        output (str): Command output text
        metadata (Optional[CmdOutputMetadata]): Execution context metadata
    """
    output: str
    metadata: Optional[CmdOutputMetadata]

    def to_observation(self, strip: bool = True) -> str:
        """
        Convert command result to formatted observation string.
        
        Args:
            strip (bool): Whether to truncate long output
            
        Returns:
            str: Formatted observation with output and context
        """
        # compile regex once for efficiency
        ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        
        output = ANSI_ESCAPE.sub("", self.output).replace("\r", "")

        if len(output) > 1024 * 16 and strip:
            output = (
                output[: 1024 * 8]
                + "....stripped due to length....\n"
                + output[-1024 * 8 :]
            )

        if self.metadata is None:
            return f"\n{output}\n"
        return f"""{output}
{self.metadata.username}@{self.metadata.hostname}:{self.metadata.working_dir} $

exit code: {self.metadata.exit_code}
"""


class SetupRuntime:
    """
    Docker container runtime for repository setup and testing.
    
    Manages a Docker container with persistent bash session, command execution,
    file operations, and container lifecycle management.
    """
    def __init__(self, container: Container, container_platform: str = "linux"):
        """
        Initialize runtime with an existing Docker container.
        
        Args:
            container (Container): Docker container instance to manage
        """
        self.container = container
        self.platform = container_platform
        self.sock = self.container.attach_socket(
            params={"stdin": 1, "stdout": 1, "stderr": 1, "stream": 1}
        )
        self.output_queue = queue.Queue()
        self._start_output_thread()
        self._clear_initial_prompt()
        if self.platform == "windows":
            self.send_command(r'''
function prompt {
  try { $ec = $global:LASTEXITCODE } catch { $ec = $null }
  $ok = $ExecutionContext.SessionState.PSVariable.GetValue('?', $true)
  if ($null -eq $ec) { $ec = if ($ok) { 0 } else { 1 } }
  $u  = $env:USERNAME
  $h  = $env:COMPUTERNAME
  $wd = (Get-Location).Path
  $pyCmd = Get-Command python -ErrorAction SilentlyContinue
  $py = if ($pyCmd) {
    if ($pyCmd.PSObject.Properties.Match('Path').Count -gt 0 -and $pyCmd.Path) { $pyCmd.Path }
    elseif ($pyCmd.PSObject.Properties.Match('Source').Count -gt 0 -and $pyCmd.Source) { $pyCmd.Source }
    else { '' }
  } else { '' }
  Write-Output ""
  Write-Output "###PS1JSON###"
  $obj = [ordered]@{
    exit_code = $ec
    username = $u
    hostname = $h
    working_dir = $wd
    py_interpreter_path = $py
  }
  $obj | ConvertTo-Json -Compress
  Write-Output "###PS1END###"
  "PS $wd> "
}
''')
        elif self.platform == "linux":
            json_str = json.dumps(
                {
                    "exit_code": "$?",
                    "username": r"\u",
                    "hostname": r"\h",
                    "working_dir": r"$(pwd)",
                    "py_interpreter_path": r'$(which python 2>/dev/null || echo "")',
                },
                indent=2,
            ).replace('"', r"\"")
            ps1 = CMD_OUTPUT_PS1_BEGIN + json_str + CMD_OUTPUT_PS1_END + "\n"
            self.send_command(
                f'export PROMPT_COMMAND=\'export PS1="{ps1}"\'; export PS2=""'
            )
        self.stopped = False

    def _stream_output(self):
        while True:
            try:
                output = self._recv_bytes(4096)
                if not output:
                    break
                self.output_queue.put(output)
            except (OSError, ConnectionError) as e:
                print(f"Connection error in _stream_output: {e}")
                break
            except Exception as e:
                # print(f"Unexpected error in _stream_output: {e}")
                break

    def _start_output_thread(self):
        self.output_thread = threading.Thread(target=self._stream_output, daemon=True)
        self.output_thread.start()

    def _clear_initial_prompt(self):
        time.sleep(0.5)
        while not self.output_queue.empty():
            self.output_queue.get()

    def _read_raw_output(self, timeout=30) -> tuple[str, Optional[CmdOutputMetadata]]:
        accumulated_output = ""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                chunk = self.output_queue.get(timeout=0.1)
                accumulated_output += chunk.decode("utf-8", errors="ignore")
                # PSReadLine injects ANSI + cursor control; normalize before matching
                accumulated_clean = ANSI_ESCAPE.sub("", accumulated_output).replace("\r", "")
                ps1_matches = CmdOutputMetadata.matches_ps1_metadata(accumulated_clean)
                if ps1_matches:
                    break
            except queue.Empty:
                continue
        accumulated_output = ANSI_ESCAPE.sub("", accumulated_output).replace("\r", "")
        ps1_matches = CmdOutputMetadata.matches_ps1_metadata(accumulated_output)
        metadata = (
            CmdOutputMetadata.from_ps1_match(ps1_matches[-1]) if ps1_matches else None
        )
        output = self._combine_outputs_between_matches(
            accumulated_output,
            ps1_matches,
        )
        return output, metadata

    def _combine_outputs_between_matches(
        self, pane_content: str, ps1_matches: list[re.Match[str]]
    ) -> str:
        if len(ps1_matches) == 1:
            return pane_content[: ps1_matches[0].start()]
        elif len(ps1_matches) == 0:
            return pane_content
        output_segments = []
        for i in range(len(ps1_matches) - 1):
            output_segment = pane_content[
                ps1_matches[i].end() + 1 : ps1_matches[i + 1].start()
            ]
            output_segments.append(output_segment)
        return "\n".join(output_segments) + "\n" if output_segments else ""

    def _recv_bytes(self, n=4096) -> bytes:
        # Prefer the public API on whatever object the SDK returns
        for m in ("recv", "read"):
            if hasattr(self.sock, m):
                return getattr(self.sock, m)(n)
        # Last-resort fallback for odd wrappers that still expose ._sock
        if hasattr(self.sock, "_sock"):
            for m in ("recv", "read"):
                if hasattr(self.sock._sock, m):
                    return getattr(self.sock._sock, m)(n)
        raise TypeError(f"Don't know how to read from {type(self.sock).__name__}")

    def _send_bytes(self, data: bytes) -> None:
        if hasattr(self.sock, "_sock"):
            for m in ("send", "sendall", "write"):
                if hasattr(self.sock._sock, m):
                    getattr(self.sock._sock, m)(data)
                    return
        for m in ("send", "sendall", "write"):
            if hasattr(self.sock, m):
                getattr(self.sock, m)(data)
                return

        raise TypeError(f"Don't know how to write to {type(self.sock).__name__}")

    def send_command(self, command: str, timeout: float = 20 * 60) -> CommandResult:
        # Normalize newline semantics for interactive shells
        if self.platform == "windows":
            # For PowerShell, ensure CRLF line endings
            command = command.strip().replace("\r\n", "\n").replace("\n", "\r\n")
            # Add extra CRLF for multi-line blocks to signal completion
            command += "\r\n\r\nprompt\r\n\r\n"
        else:
            if not command.endswith("\n"):
                command += "\n"

        while not self.output_queue.empty():
            self.output_queue.get()

        self._send_bytes(command.encode())

        output, metadata = self._read_raw_output(timeout=timeout)
        if metadata is not None:
            return CommandResult(output=output, metadata=metadata)

        # handle timeout
        self._send_bytes(b"\x03")

        kill_timeout = 5.0
        kill_output, kill_metadata = self._read_raw_output(timeout=kill_timeout)

        output = output + kill_output + "\n**Exited due to timeout**\n"
        if kill_metadata is not None:
            kill_metadata.exit_code = TIMEOUT_EXIT_CODE
            return CommandResult(output=output, metadata=kill_metadata)

        fallback_metadata = CmdOutputMetadata(
            exit_code=TIMEOUT_EXIT_CODE,
        )

        return CommandResult(output=output, metadata=fallback_metadata)

    def copy_to_container(self, src: str, dest: str) -> None:
        """
        Copy local file or directory 'src' into the container at path 'dest'.

        If 'src' is a directory, all files within that directory (recursively)
        are placed inside 'dest' in the container. If 'src' is a single file,
        it is placed inside 'dest' (which is typically a directory).
        """
        tar_stream = io.BytesIO()
        src = os.path.abspath(src)

        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            if os.path.isdir(src):
                # Add directory contents so they appear directly under `dest`.
                tar.add(src, arcname=".")
            else:
                # Add a single file using its basename.
                tar.add(src, arcname=os.path.basename(src))

        tar_stream.seek(0)

        # Put the archive into the container. `dest` must exist and be a directory
        # when copying directories, or you'll need to ensure it's the appropriate file path
        # when copying a single file.
        self.container.put_archive(dest, tar_stream)

    def copy_dir_to_container(self, src: str, dest: str) -> None:

        src = Path(src)
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for file_path in src.rglob("*"):
                arcname = file_path.relative_to(src)
                tar.add(str(file_path), arcname=str(arcname))
        tar_stream.seek(0)

        self.container.put_archive(path=dest, data=tar_stream.read())
        if self.platform == "linux":
            self.send_command(f'chown -R root:root "{dest}"')

    def cleanup(self, prune_dangling = True) -> None:
        if self.stopped:
            return
        try:
            self.container.stop()
            self.container.remove(force=True)
            self.stopped = True
        except Exception as e:
            print(f"Failed to stop container: {e}")
        if prune_dangling:
            try:
                client = docker.from_env()
                client.images.prune(filters={'dangling': True})
            except Exception as e:
                print(e, "...Skipping...")

    def commit(self, image_name: str, tag: str = "latest", push: bool = False) -> str:
        self.container.stop()

        self.container.commit(
            repository=image_name,
            tag=tag,
        )
        print(f"Image {image_name}:{tag} created successfully.")

        if push:
            client = docker.from_env()
            client.images.push(image_name, tag=tag)
            print(f"Image {image_name}:{tag} pushed successfully.")

        self.container.start()
        return f"{image_name}:{tag}"

    def __del__(self):
        self.cleanup()


    @staticmethod
    def pull_image(image_name: str) -> bool:
        """
        Pull Docker image from registry.
        
        Args:
            image_name (str): Name of the Docker image to pull
            
        Returns:
            bool: True if successful, False if image not found
        """
        client = docker.from_env()
        try:
            # Check if image exists locally
            client.images.get(image_name)
            return True
        except docker.errors.ImageNotFound:
            # Image doesn't exist locally, try to pull it
            try:
                client.images.pull(image_name)
                return True
            except docker.errors.ImageNotFound:
                raise ValueError(f"Image {image_name} not found in registry")

    @classmethod
    def from_launch_image(
        cls,
        image_name: str,
        instance_id: str,
        platform: Literal["linux", "windows"] = "linux",
    ) -> SetupRuntime:
        """
        Start a Docker container session for repository testing.
        
        Args:
            image_name (str): Base Docker image name
            instance (dict): SWE-bench instance data with repo info
            platform: the platform of the container, linux or windows
            
        Returns:
            SetupRuntime: Configured runtime session ready for command execution
            
        Raises:
            RuntimeError: If Docker is not available
        """
        try:
            docker.from_env().ping()
        except docker.errors.DockerException:
            raise RuntimeError("Docker is not installed or not running.")

        _ = cls.pull_image(image_name)
        client = docker.from_env(timeout=600)
        container_name = f"git-launch-{instance_id}-{str(uuid.uuid4())[:4]}"
        info = client.version()
        engine_os = (info.get("Os") or info.get("OSType") or "").lower() 
        # which operating system this code is running on, note windows can run linux containers, so engine_os != (container) platform
        extra_hosts = {"host.docker.internal": "host-gateway"} if "linux" in engine_os else None
        
        if platform == "windows":
            shell_command = r"powershell -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -NoExit"
            working_dir = r"C:\testbed"
            run_kwargs = {
                "cpu_count": CPU_CORES,  # cpu_quota is Linux-only
                "mem_limit": MEM_LIMIT,
            }
        else:
            shell_command = "/bin/bash"
            working_dir = "/testbed"
            run_kwargs = {
                "cpu_quota": int(CPU_CORES * 100000),
                "mem_limit": MEM_LIMIT,
            }

        container = client.containers.run(
            image_name,
            name=container_name,
            command=shell_command,
            stdin_open=True,
            tty=True,
            detach=True,
            environment={
                "TERM": "xterm-mono",
            },
            working_dir=working_dir,
            extra_hosts=extra_hosts,
            **run_kwargs,
        )

        session = cls(container, container_platform=platform)

        return session



    @classmethod
    def from_base_image(
        cls,
        image_name: str,
        instance: dict,
        platform: Literal["linux", "windows"] = "linux",
    ) -> SetupRuntime:
        """
        Start a Docker container session for repository testing.
        
        Args:
            image_name (str): Base Docker image name
            instance (dict): SWE-bench instance data with repo info
            platform: the platform of the container, linux or windows
            
        Returns:
            SetupRuntime: Configured runtime session ready for command execution
            
        Raises:
            RuntimeError: If Docker is not available
        """
        try:
            docker.from_env().ping()
        except docker.errors.DockerException:
            raise RuntimeError("Docker is not installed or not running.")

        _ = cls.pull_image(image_name)
        client = docker.from_env(timeout=600)
        container_id = instance["instance_id"]
        container_name = f"git-launch-{container_id}-{str(uuid.uuid4())[:4]}"
        info = client.version()
        engine_os = (info.get("Os") or info.get("OSType") or "").lower() 
        # which operating system this code is running on, note windows can run linux containers, so engine_os != (container) platform
        extra_hosts = {"host.docker.internal": "host-gateway"} if "linux" in engine_os else None
        
        if platform == "windows":
            shell_command = r"powershell -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -NoExit"
            working_dir = r"C:\testbed"
            run_kwargs = {
                "cpu_count": CPU_CORES,  # cpu_quota is Linux-only
                "mem_limit": MEM_LIMIT,
            }
        else:
            shell_command = "/bin/bash"
            working_dir = "/testbed"
            run_kwargs = {
                "cpu_quota": int(CPU_CORES * 100000),
                "mem_limit": MEM_LIMIT,
            }

        container = client.containers.run(
            image_name,
            name=container_name,
            command=shell_command,
            stdin_open=True,
            tty=True,
            detach=True,
            environment={
                "TERM": "xterm-mono",
            },
            working_dir=working_dir,
            extra_hosts=extra_hosts,
            **run_kwargs,
        )

        session = cls(container, container_platform=platform)

        # We avoid copying due to performance issues
        # session.copy_dir_to_container(str(workspace), "/workspace")

        url = f'https://github.com/{instance["repo"]}.git'
        base_commit = instance["base_commit"]

        if platform == "windows":
            # 2) Ensure Git is installed (Chocolatey if possible; fallback to official silent installer).
            #    - Chocolatey official install script: https://community.chocolatey.org/install.ps1
            #    - git.install package params include /GitOnlyOnPath, /GitAndUnixToolsOnPath, /NoAutoCrlf, etc.
            #    - Git for Windows silent flags are documented by the project itself.
            session.send_command(r'''
# Skip if git already present
if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
  try {
    # Prefer Chocolatey (cleaner package mgmt)
    if (-not (Get-Command choco.exe -ErrorAction SilentlyContinue)) {
      Set-ExecutionPolicy Bypass -Scope Process -Force
      [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
      Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    }

    choco install git.install -y --no-progress --params '"/GitOnlyOnPath /NoAutoCrlf"'
  }
  catch {
    Write-Host "Chocolatey install failed: $($_.Exception.Message)  -> falling back to Git for Windows installer"

    # Fallback: Official Git for Windows silent install
    $ProgressPreference = 'SilentlyContinue'
    $temp = Join-Path $env:TEMP 'git-installer.exe'
    # 'latest' link maintained by Git for Windows; resolves to current amd64 EXE
    $url  = 'https://github.com/git-for-windows/git/releases/latest/download/Git-64-bit.exe'
    try {
      Invoke-WebRequest -Uri $url -OutFile $temp
    } catch {
      Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $temp
    }

    # Silent/unattended flags per Git for Windows docs (Inno Setup):
    # /VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS
    # Optional components: icons, ext\reg\shellhere, assoc, assoc_sh, gitlfs, windowsterminal, scalar
    Start-Process -FilePath $temp -ArgumentList `
      '/VERYSILENT','/NORESTART','/NOCANCEL','/SP-','/CLOSEAPPLICATIONS','/RESTARTAPPLICATIONS',`
      '/COMPONENTS="icons,ext\reg\shellhere,assoc,assoc_sh,gitlfs,windowsterminal,scalar"' `
      -Wait
  }

  # Ensure PATH is updated in this running session (Chocolatey/Git installers update registry only)
  $gitCmd = 'C:\Program Files\Git\cmd'
  $gitBin = 'C:\Program Files\Git\bin'
  if (Test-Path $gitCmd) { $env:PATH = "$gitCmd;$gitBin;$env:PATH" }
}
''')
            res: CommandResult = session.send_command(
                r'git clone {url} "C:\testbed"; cd "C:\testbed"; git reset --hard {base}'.format(
                    url=url, base=base_commit
            )
        )
        else: 
            session.send_command("apt update && apt install -y git")
            res: CommandResult = session.send_command(
                f"git clone {url} /testbed && cd /testbed && git reset --hard {base_commit}"
            )
        

        return session


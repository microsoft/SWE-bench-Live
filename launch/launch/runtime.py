"""
Docker runtime management for repository setup and command execution.

Provides containerized environment for repository testing with command execution,
file operations, and state management capabilities.
"""
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
from typing import Optional

import docker
from docker.models.containers import Container
from typing_extensions import Self

CMD_OUTPUT_PS1_BEGIN = "\n###PS1JSON###\n"
CMD_OUTPUT_PS1_END = "\n###PS1END###"
CMD_OUTPUT_METADATA_PS1_REGEX = re.compile(
    f"^{CMD_OUTPUT_PS1_BEGIN.strip()}(.*?){CMD_OUTPUT_PS1_END.strip()}",
    re.DOTALL | re.MULTILINE,
)

TIMEOUT_EXIT_CODE = 124

MEM_LIMIT = "8g"
CPU_CORES = 4


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
            try:
                json.loads(match.group(1).strip())  # Try to parse as JSON
                matches.append(match)
            except json.JSONDecodeError:
                continue  # Skip if not valid JSON
        return matches

    @classmethod
    def from_ps1_match(cls, match: re.Match[str]) -> Self:
        """
        Extract metadata from a PS1 prompt regex match.
        
        Args:
            match (re.Match[str]): Regex match containing JSON metadata
            
        Returns:
            Self: CmdOutputMetadata instance with parsed values
        """
        metadata = json.loads(match.group(1))
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

        if len(output) > 1024 * 8 and strip:
            output = (
                output[: 1024 * 4]
                + "....stripped due to length....\n"
                + output[-1024 * 4 :]
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
    def __init__(self, container: Container):
        """
        Initialize runtime with an existing Docker container.
        
        Args:
            container (Container): Docker container instance to manage
        """
        self.container = container
        self.sock = self.container.attach_socket(
            params={"stdin": 1, "stdout": 1, "stderr": 1, "stream": 1}
        )

        self.output_queue = queue.Queue()
        self._start_output_thread()
        self._clear_initial_prompt()
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
        self.send_command("apt update && apt install -y git")
        self.stopped = False

    def _stream_output(self):
        while True:
            try:
                output = self.sock._sock.recv(4096)
                if not output:
                    break
                self.output_queue.put(output)
            except (OSError, ConnectionError) as e:
                print(f"Connection error in _stream_output: {e}")
                break
            except Exception as e:
                print(f"Unexpected error in _stream_output: {e}")
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
                ps1_matches = CmdOutputMetadata.matches_ps1_metadata(accumulated_output)
                if ps1_matches:
                    break
            except queue.Empty:
                continue
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

    def send_command(self, command: str, timeout: float = 20 * 60) -> CommandResult:
        if not command.endswith("\n"):
            command += "\n"

        while not self.output_queue.empty():
            self.output_queue.get()

        self.sock._sock.send(command.encode())

        output, metadata = self._read_raw_output(timeout=timeout)
        if metadata is not None:
            return CommandResult(output=output, metadata=metadata)

        # handle timeout
        self.sock._sock.send(b"\x03")

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
        self.send_command(f"chown -R root:root {dest}")

    def cleanup(self) -> None:
        if self.stopped:
            return
        try:
            self.container.stop()
            self.container.remove(force=True)
            self.stopped = True
        except Exception as e:
            print(f"Failed to stop container: {e}")

    def commit(self, image_name: str, tag: str = "latest", push: bool = False) -> str:
        self.container.commit(
            repository=image_name,
            tag=tag,
        )
        print(f"Image {image_name}:{tag} created successfully.")

        if push:
            client = docker.from_env()
            client.images.push(image_name, tag=tag)
            print(f"Image {image_name}:{tag} pushed successfully.")

        return f"{image_name}:{tag}"

    def __del__(self):
        self.cleanup()


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
        client.images.pull(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False


def start_session(
    image_name: str,
    instance: dict,
) -> SetupRuntime:
    """
    Start a Docker container session for repository testing.
    
    Args:
        image_name (str): Base Docker image name
        instance (dict): SWE-bench instance data with repo info
        
    Returns:
        SetupRuntime: Configured runtime session ready for command execution
        
    Raises:
        RuntimeError: If Docker is not available
    """
    try:
        docker.from_env().ping()
    except docker.errors.DockerException:
        raise RuntimeError("Docker is not installed or not running.")

    _ = pull_image(image_name)
    client = docker.from_env(timeout=600)
    container_id = instance["instance_id"]
    container_name = f"git-launch-{container_id}-{str(uuid.uuid4())[:4]}"
    container = client.containers.run(
        image_name,
        name=container_name,
        command="/bin/bash",
        stdin_open=True,
        tty=True,
        detach=True,
        environment={
            "TERM": "xterm-mono",
        },
        working_dir="/testbed",
        extra_hosts={"host.docker.internal": "host-gateway"},
        # resources
        mem_limit=MEM_LIMIT,
        cpu_quota=int(CPU_CORES * 100000),
        # volumes={str(workspace.absolute()): {"bind": "/workspace", "mode": "rw"}},
    )

    session = SetupRuntime(container)

    # We avoid copying due to performance issues
    # session.copy_dir_to_container(str(workspace), "/workspace")

    url = f'https://github.com/{instance["repo"]}.git'
    base_commit = instance["base_commit"]

    session.send_command(
        f"git clone {url} /testbed && cd /testbed && git reset --hard {base_commit}"
    )

    return session


if __name__ == "__main__":
    client = docker.from_env()
    container = client.containers.run(
        "python",
        name="launch-runtime-test",
        command="/bin/bash",
        stdin_open=True,
        tty=True,
        detach=True,
        environment={
            "TERM": "xterm-mono",
        },
    )
    docker_tty = SetupRuntime(container)
    try:
        # Test 1: Basic command with prompt parsing
        result = docker_tty.send_command('echo "Hello World"')
        assert result.metadata.exit_code == 0
        assert result.metadata.working_dir == "/"
        assert result.metadata.username == "root"
        print(result.output)

        # Test 2: Command with multiple lines
        result = docker_tty.send_command('for i in {1..3}; do echo "Line $i"; done')
        assert result.metadata.exit_code == 0
        assert result.metadata.working_dir == "/"
        print(result.output)

        # Test 3: Change directory and verify prompt update
        result = docker_tty.send_command("cd /tmp && pwd")
        assert (
            result.metadata.working_dir == "/tmp"
        ), f"Working directory should be /tmp, got {result.metadata.working_dir}"
        assert result.metadata.exit_code == 0
        print(result.output)

        # Test 4: Command with error
        result = docker_tty.send_command("ls /nonexistent")
        assert (
            "No such file or directory" in result.output
        ), f"Expected error message not found in '{result.output}'"
        assert (
            result.metadata.exit_code == 2
        ), f"Expected exit code 2, got {result.metadata.exit_code}"
        assert (
            result.metadata.working_dir == "/tmp"
        ), "Working directory should remain /tmp"
        print(result.output)

        # Test 5: Timeout
        start_time = time.time()
        result = docker_tty.send_command("sleep 5", timeout=2)
        end_time = time.time()
        assert (
            end_time - start_time < 3
        ), f"Command should have timed out, took {end_time - start_time} seconds"
        print(result.to_observation())
    finally:
        docker_tty.cleanup()

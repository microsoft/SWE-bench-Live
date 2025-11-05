#!/usr/bin/env python3
"""
Modal deployment for SWE-bench-Live with self-hosted Qwen model.

This version runs the Qwen model directly on Modal's GPUs (no external API needed!).
Optimized for Qwen3-Coder-30B-A3B-Instruct with tensor parallelism.

UPDATED: Now uses proper SWE-bench Docker environments during patch generation
for fairness and consistency with evaluation.
"""

import modal
from pathlib import Path

app = modal.App("swebench-qwen-selfhosted")

# GPU configuration for Qwen model
# A100-80GB with 4 GPUs for tensor parallelism
GPU_CONFIG = "A100-80GB:4"  # Modal 1.0 syntax

# Model configuration - Qwen3 30B A3B (optimized for agentic tasks)
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MODEL_REVISION = "main"

# Create image with model dependencies
model_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.10.0",  # Pinned to 0.10.0 to prevent errors
        "transformers>=4.40.0",
        "torch>=2.1.0",
        "accelerate>=0.25.0",
        "datasets",
        "pyyaml",
        "tenacity",
    )
)

# Environment image for running code
# Install local swebench in editable mode to use the latest code
env_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "mini-swe-agent",
        "datasets",
        "pyyaml",
        "tenacity",
        # Don't install swebench from PyPI - we'll use local version
    )
    # Install local swebench package (includes fix for SWE-bench-Live instances)
    .add_local_dir(".", remote_path="/root/swebench-repo", copy=True)
    .run_commands(
        "cd /root/swebench-repo && pip install -e ."
    )
)


# Model inference class
@app.cls(
    image=model_image,
    gpu=GPU_CONFIG,
    timeout=3600 * 2,  # 2 hours
    scaledown_window=600,  # Keep warm for 10 minutes
)
class QwenModel:
    """Self-hosted Qwen model on Modal with vLLM for fast inference."""


    @modal.enter()
    def load_model(self):
        """Load the model when container starts."""
        from vllm import LLM, SamplingParams
        import torch

        print(f"Loading model: {MODEL_NAME}")
        print(f"GPU config: {GPU_CONFIG}")

        # vLLM automatically handles tensor parallelism with multiple GPUs
        self.llm = LLM(
            model=MODEL_NAME,
            revision=MODEL_REVISION,
            tensor_parallel_size=4,  # Use all 4 GPUs
            gpu_memory_utilization=0.85,  # Use 85% (reduced from 95%)
            max_model_len=32768,  # 32K tokens for longer conversations
            trust_remote_code=True,  # Required for Qwen models
            dtype="bfloat16",  # Use bfloat16 for stability
        )

        # Default sampling parameters
        self.default_sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic for reproducibility
            max_tokens=4096,
            top_p=1.0,
        )

        print("Model loaded successfully!")

    @modal.method()
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        from vllm import SamplingParams

        # Format messages into a prompt (Qwen chat template)
        prompt = self._format_messages(messages)

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0 if temperature == 0.0 else 0.95,
        )

        # Generate
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages into Qwen chat template."""
        # Qwen uses <|im_start|> and <|im_end|> tokens
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        # Add assistant start token
        formatted += "<|im_start|>assistant\n"
        return formatted


# Global model instance (one per worker container, stays warm)
_model_instance = None

def get_model():
    """Get or create the singleton model instance for this worker."""
    global _model_instance
    if _model_instance is None:
        print("Loading model for this worker (happens once per container)...")
        _model_instance = QwenModel()
        print("Model loaded and ready!")
    return _model_instance

# Agent execution on Modal Sandbox
@app.function(
    image=env_image,
    cpu=4,
    timeout=3600,  # 1 hour per instance
    include_source=True,  # Include local swebench code for TestSpec generation
)
def run_instance_with_model(
    instance: dict,
    extract_mode: str = "post-agent",
):
    """
    Run a single SWE-bench instance with the self-hosted model.

    Args:
        instance: Full SWE-bench instance dict with all metadata
        extract_mode: Patch extraction mode - "original" or "post-agent" (default)
    """
    import json
    import time
    import traceback
    import yaml
    from pathlib import Path
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.modal_eval.run_evaluation_modal import ModalSandboxRuntime

    instance_id = instance["instance_id"]
    problem_statement = instance["problem_statement"]

    print(f"Processing instance: {instance_id}")

    try:
        # Environment that uses proper SWE-bench Docker setup via Modal sandbox
        class ModalSandboxEnvironment:
            """
            Environment for mini-swe-agent that uses ModalSandboxRuntime.
            Provides the exact same environment used during evaluation.
            """

            def __init__(self, sandbox_runtime: ModalSandboxRuntime):
                self.sandbox_runtime = sandbox_runtime
                self.config = type('Config', (), {
                    'cwd': '/testbed',  # Repository is set up in /testbed
                    'timeout': 60,
                    'env': {
                        'PAGER': 'cat',
                        'MANPAGER': 'cat',
                    }
                })()

            def execute(self, command: str, cwd: str = "", *, timeout: int | None = None):
                """Execute bash command in the Modal sandbox."""
                # ModalSandboxRuntime.exec() expects a single command string
                # and returns (output, returncode) tuple
                cwd = cwd or self.config.cwd

                # Prepend cd command if cwd is different from default
                if cwd != self.config.cwd:
                    command = f"cd {cwd} && {command}"

                try:
                    output, returncode = self.sandbox_runtime.exec(command)
                    return {
                        "output": output,
                        "returncode": returncode,
                    }
                except Exception as e:
                    return {
                        "output": f"Error executing command: {str(e)}",
                        "returncode": 1,
                    }

            def get_template_vars(self):
                return {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Cleanup sandbox
                self.sandbox_runtime.__exit__(exc_type, exc_val, exc_tb)

        # Create a model wrapper that uses the Modal-hosted model
        class ModalModelWrapper:
            """Wrapper to make Modal model work like mini-swe-agent model."""

            def __init__(self, modal_model_instance):
                # Use the passed model instance instead of creating a new one
                self.modal_model = modal_model_instance
                self.cost = 0.0  # Free since it's self-hosted!
                self.n_calls = 0
                self.config = type('Config', (), {
                    'model_name': MODEL_NAME,
                })()

            def query(self, messages: list[dict], **kwargs) -> dict:
                """Query the Modal-hosted model."""
                self.n_calls += 1

                # Get generation parameters
                temperature = kwargs.get('temperature', 0.0)
                max_tokens = kwargs.get('max_tokens', 4096)

                # Call the model
                response = self.modal_model.generate.remote(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                return {
                    "content": response,
                    "extra": {},
                }

            def get_template_vars(self):
                return {
                    'model_name': MODEL_NAME,
                    'n_model_calls': self.n_calls,
                    'model_cost': self.cost,
                }

        # Configure submission command based on extraction mode
        if extract_mode == "original":
            extraction_command = " && git add -A && git diff --cached"
            extraction_description = """This command will stage all changes and display the patch.
The patch will be captured from the command output."""
        else:  # post-agent mode (default)
            extraction_command = ""
            extraction_description = """This command marks task completion.
The patch will be extracted automatically after you submit."""

        # SWE-bench agent config (embedded directly to avoid file upload issues)
        config = {
            "agent": {
                "step_limit": 250,
                "cost_limit": 3.0,
                # Proper SWE-bench templates
                "system_template": """You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.""",
                "instance_template": f"""<pr_description>
Consider the following PR description:
{{{{task}}}}
</pr_description>

<instructions>
# Task Instructions
You're a software engineer helping implement necessary changes to meet requirements in the PR description.
Your task is to make changes to non-test files to fix the issue described.

For each response:
1. Include a THOUGHT section explaining your reasoning
2. Provide exactly ONE bash command in triple backticks

## Important Boundaries
- MODIFY: Regular source code files in /testbed
- DO NOT MODIFY: Tests, configuration files

## Recommended Workflow
1. Analyze the codebase
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works

## Submission
When you've completed your work, issue exactly this command:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT{extraction_command}
```
{extraction_description}
</instructions>""",
                "action_observation_template": """<returncode>{{output.returncode}}</returncode>
{% if output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>Output too long. Use head/tail/sed for selective viewing.</warning>
<output_head>{{ output.output[:5000] }}</output_head>
<elided_chars>{{ output.output | length - 10000 }} characters elided</elided_chars>
<output_tail>{{ output.output[-5000:] }}</output_tail>
{%- endif -%}""",
            },
            "environment": {
                "cwd": "/testbed",
                "timeout": 60,
                "env": {
                    "PAGER": "cat",
                    "MANPAGER": "cat",
                },
            },
        }

        # Create TestSpec from instance - this generates the proper Docker environment
        print(f"Creating TestSpec for {instance_id}...")
        # Use pre-built images from DockerHub namespace
        test_spec = make_test_spec(instance, namespace="starryzhang")
        print(f"TestSpec created: repo={test_spec.repo}, arch={test_spec.arch}")
        print(f"Using pre-built image: {test_spec.instance_image_key}")

        # Create Modal sandbox runtime with proper Docker environment
        # This builds an image with:
        # - Correct Python version
        # - All dependencies from requirements.txt
        # - Repository cloned at base_commit
        # - Full git history
        print(f"Creating Modal sandbox with proper environment...")
        sandbox_runtime = ModalSandboxRuntime(
            test_spec=test_spec,
            timeout=3600,  # 1 hour timeout
            verbose=True
        )
        print(f"Sandbox created successfully! Repository ready in /testbed")

        # Use context manager to ensure proper cleanup of sandbox
        with ModalSandboxEnvironment(sandbox_runtime) as env:
            # Get the singleton model instance for this worker
            model_instance = get_model()

            # Create model wrapper
            model_wrapper = ModalModelWrapper(model_instance)

            # Create agent
            from minisweagent.agents.default import DefaultAgent

            agent = DefaultAgent(
                model_wrapper,
                env,
                **config.get("agent", {}),
            )

            # Run the agent
            print(f"Running agent on {instance_id}...")
            start_time = time.time()
            exit_status, message = agent.run(problem_statement)
            duration = time.time() - start_time

            print(f"Completed in {duration:.2f}s - Status: {exit_status}")
            print(f"Model calls: {model_wrapper.n_calls}")

            # Extract patch based on extraction mode
            if extract_mode == "original":
                # Original mode: Extract from agent's message (stream capture)
                print("Using original extraction mode (stream capture from agent message)...")
                patch = message.strip()
                print(f"Captured patch from agent submission: {len(patch)} bytes")
            else:
                # Post-agent mode (default): Extract directly after agent completes
                print("Using post-agent extraction mode (direct git diff execution)...")
                print("Staging all changes and generating patch...")
                patch_output, rc = sandbox_runtime.exec("git add -A && git diff --cached")
                patch = patch_output.strip()
                print(f"Generated patch: {len(patch)} bytes, return code: {rc}")

            # Verify patch completeness by checking for proper git diff format
            if patch:
                lines = patch.split('\n')
                has_diff_header = any(line.startswith('diff --git') for line in lines[:10])

                print(f"Patch has diff header: {has_diff_header}")

                # Show first and last few lines for verification
                first_lines = '\n'.join(lines[:3])
                last_lines = '\n'.join(lines[-3:]) if len(lines) > 3 else ''
                print(f"Patch preview (first 3 lines):\n{first_lines}")
                if last_lines:
                    print(f"Patch preview (last 3 lines):\n{last_lines}")

                if not has_diff_header:
                    print("Warning: Patch missing git diff header!")
            else:
                print(f"Warning: No patch generated!")

        # Sandbox is now cleaned up
        print(f"Sandbox terminated successfully")

        return {
            "instance_id": instance_id,
            "model_name_or_path": MODEL_NAME,
            "model_patch": patch,
            "exit_status": exit_status,
            "duration": duration,
            "model_calls": model_wrapper.n_calls,
            "success": True,
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            "instance_id": instance_id,
            "model_name_or_path": MODEL_NAME,
            "model_patch": "",
            "exit_status": f"error: {str(e)}",
            "error": error_msg,
            "success": False,
        }


@app.function(
    image=env_image,
    timeout=3600 * 24,  # 24 hours for batch
    include_source=True,  # Include local code
)
def run_batch_evaluation(
    dataset_name: str = "SWE-bench-Live/SWE-bench-Live",
    split: str = "lite",
    slice_spec: str = "",
    extract_mode: str = "post-agent",
):
    """Run batch evaluation on multiple instances.

    Args:
        extract_mode: Patch extraction mode - "original" or "post-agent" (default)
    """
    from datasets import load_dataset
    import json

    print(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split)

    # Apply slice
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        start = values[0] if len(values) > 0 and values[0] is not None else 0
        end = values[1] if len(values) > 1 and values[1] is not None else len(dataset)
        dataset = dataset.select(range(start, end))

    print(f"Processing {len(dataset)} instances with self-hosted Qwen model")
    print("Note: Model will load once per worker and stay warm for 10 minutes (scaledown_window)")
    print("Using proper SWE-bench Docker environments for patch generation")
    print(f"Patch extraction mode: {extract_mode}")

    # Prepare inputs: full instance dicts + extract_mode
    # Model will be created once per worker container and reused via Modal's warmth
    # Each instance needs full metadata to create TestSpec
    inputs = [
        (inst, extract_mode)  # Pass instance dict and extract_mode for starmap
        for inst in dataset
    ]

    # Run instances in parallel using starmap
    # Modal will reuse warm containers, so model stays loaded between calls
    print("Starting parallel execution with proper Docker environments...")
    results = list(run_instance_with_model.starmap(inputs))

    # Summary
    successful = sum(1 for r in results if r.get("success", False))
    total_calls = sum(r.get("model_calls", 0) for r in results)
    total_duration = sum(r.get("duration", 0) for r in results)

    print(f"\n{'='*60}")
    print(f"Batch Evaluation Complete!")
    print(f"Total: {len(results)} | Success: {successful} | Failed: {len(results) - successful}")
    print(f"Total model calls: {total_calls}")
    print(f"Total duration: {total_duration/60:.1f} minutes")
    print(f"Cost: $0 (self-hosted on Modal!) 🎉")
    print(f"{'='*60}\n")

    return {r["instance_id"]: r for r in results}


@app.local_entrypoint()
def main(
    dataset: str = "SWE-bench-Live/SWE-bench-Live",
    split: str = "lite",
    slice: str = "0:5",
    docker_mode: str = "modal",
    gpu: str = "modal",
    extract: str = "post-agent",
):
    """
    Main entry point for self-hosted Qwen evaluation.

    Args:
        extract: Patch extraction mode - "original" (stream capture) or "post-agent" (direct git diff, default)

    Usage:
        # Run on Modal (default - GPU + Docker on Modal)
        modal run run_swebench_modal_selfhosted.py
        modal run run_swebench_modal_selfhosted.py --slice 0:1
        modal run run_swebench_modal_selfhosted.py --split verified --slice 0:10

        # Run with local Docker but Modal GPU (hybrid)
        modal run run_swebench_modal_selfhosted.py --docker-mode local --slice 0:1

        # Run with local GPU but Modal Docker
        modal run run_swebench_modal_selfhosted.py --gpu local --slice 0:1

        # Use original extraction mode (for testing/comparison)
        modal run run_swebench_modal_selfhosted.py --extract original --slice 0:1

        # Run completely locally (GPU + Docker local)
        modal run run_swebench_modal_selfhosted.py --gpu local --docker-mode local --slice 0:1

    Args:
        docker_mode: "modal" (use Modal Sandbox) or "local" (use local Docker)
        gpu: "modal" (use Modal GPU) or "local" (use local GPU with vLLM)
    """
    print("="*60)
    print("SWE-bench-Live with Self-Hosted Qwen")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"GPU: {GPU_CONFIG if gpu == 'modal' else 'Local GPU'}")
    print(f"Dataset: {dataset}")
    print(f"Split: {split}")
    print(f"Slice: {slice}")
    print(f"Docker Mode: {docker_mode}")
    print(f"GPU Mode: {gpu}")
    print("="*60)

    # Check GPU mode
    if gpu == "local":
        print("\n⚠️  Local GPU mode requested")
        print("Note: Local GPU mode requires vLLM and CUDA installed locally")
        print("This mode is currently experimental.\n")
        # TODO: Implement local GPU execution
        # For now, we'll error out and ask user to use Modal GPU
        raise NotImplementedError(
            "Local GPU mode is not yet fully implemented. "
            "Please use --gpu modal (default) for now. "
            "The patch extraction fix works with Modal GPU."
        )

    # Check Docker mode
    if docker_mode == "local":
        # Run locally with Docker (not on Modal Sandbox)
        print("\n⚠️  Running with LOCAL Docker execution")
        print("Note: This will pull Docker images from DockerHub and run them locally")
        print("Make sure Docker is installed and running!\n")

        # Import local execution function
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from run_swebench_local import run_batch_evaluation_local

        results = run_batch_evaluation_local(
            dataset_name=dataset,
            split=split,
            slice_spec=slice,
            extract_mode=extract,
        )
    else:
        # Run on Modal (default)
        print("\n✓ Running on Modal GPU with Docker Sandboxes\n")
        print(f"Patch extraction mode: {extract}\n")
        results = run_batch_evaluation.remote(
            dataset_name=dataset,
            split=split,
            slice_spec=slice,
            extract_mode=extract,
        )

    print(f"\n✓ Completed! Processed {len(results)} instances")

    # Save results locally
    import json
    output_file = f"predictions_selfhosted_{split}_{slice.replace(':', '-')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_file}")
    if docker_mode == "modal":
        print(f"✓ Cost: $0 for API calls (self-hosted!) + Modal GPU time")
    else:
        print(f"✓ Cost: $0 (completely free - local execution!)")


# Test function to verify model loading
@app.local_entrypoint()
def test_model():
    """Test that the model loads and runs correctly."""
    print("Testing model loading and inference...")

    # Instantiate the model class - load_model will run automatically via @modal.enter()
    model = QwenModel()

    # Test generation
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a hello world program in Python."},
    ]

    print("Calling model for test generation...")
    response = model.generate.remote(messages)
    print(f"Test response:\n{response[:500]}...")
    print("\n✓ Model test successful!")

    return response


# Test function to verify Docker sandbox setup
@app.local_entrypoint()
def test_sandbox():
    """Test that Modal sandbox with Docker environment works correctly."""
    from datasets import load_dataset

    print("Testing Modal sandbox with proper Docker environment...")
    print("="*60)

    # Load a single test instance from SWE-bench-Live
    print("Loading test instance from SWE-bench-Live...")
    dataset = load_dataset("SWE-bench-Live/SWE-bench-Live", split="lite")
    test_instance = dataset[0]

    print(f"Test instance: {test_instance['instance_id']}")
    print(f"Repository: {test_instance['repo']}")
    print(f"Base commit: {test_instance['base_commit'][:8]}")

    # Test the sandbox setup with this instance
    result = test_sandbox_instance.remote(test_instance)

    if result["success"]:
        print("\n✓ Sandbox test successful!")
        print(f"  - Docker image built successfully")
        print(f"  - Repository cloned at /testbed")
        print(f"  - Python version: {result['python_version']}")
        print(f"  - Git status: {result['git_status'][:100]}...")
    else:
        print(f"\n✗ Sandbox test failed: {result['error']}")

    print("="*60)
    return result


@app.function(
    image=env_image,
    timeout=1800,  # 30 minutes for test
    include_source=True,
)
def test_sandbox_instance(instance: dict):
    """Test Modal sandbox setup for a single instance."""
    import traceback
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.modal_eval.run_evaluation_modal import ModalSandboxRuntime

    instance_id = instance["instance_id"]

    try:
        print(f"Creating TestSpec for {instance_id}...")
        test_spec = make_test_spec(instance, namespace="starryzhang")
        print(f"✓ TestSpec created: repo={test_spec.repo}, arch={test_spec.arch}")
        print(f"✓ Using pre-built image: {test_spec.instance_image_key}")

        print(f"Creating Modal sandbox with Docker environment...")
        sandbox_runtime = ModalSandboxRuntime(
            test_spec=test_spec,
            timeout=600,  # 10 minutes
            verbose=True
        )
        print(f"✓ Sandbox created successfully!")

        # Test some basic commands
        print("Testing basic commands in sandbox...")

        # Check Python version
        python_output, rc = sandbox_runtime.exec("python --version")
        print(f"Python version: {python_output.strip()}")

        # Check if we're in /testbed
        pwd_output, rc = sandbox_runtime.exec("pwd")
        print(f"Current directory: {pwd_output.strip()}")

        # Check git status
        git_output, rc = sandbox_runtime.exec("cd /testbed && git status")
        print(f"Git status:\n{git_output[:200]}")

        # Check if repository files exist
        ls_output, rc = sandbox_runtime.exec("cd /testbed && ls -la | head -20")
        print(f"Repository contents:\n{ls_output[:300]}")

        # Cleanup
        sandbox_runtime.__exit__(None, None, None)

        return {
            "success": True,
            "instance_id": instance_id,
            "python_version": python_output.strip(),
            "git_status": git_output,
        }
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            "success": False,
            "instance_id": instance_id,
            "error": error_msg,
        }

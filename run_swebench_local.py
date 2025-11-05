#!/usr/bin/env python3
"""
Local Docker execution for SWE-bench-Live with self-hosted Qwen model.

This version runs the Qwen model on Modal GPUs but executes mini-swe-agent
in local Docker containers built from SWE-bench TestSpec (same as evaluation).

IMPORTANT: This requires Docker to be installed and running locally.
"""

import sys
import time
import traceback
from pathlib import Path
from datasets import load_dataset
import modal

# Import from the Modal script for model access
from run_swebench_modal_selfhosted import MODEL_NAME, QwenModel, app

# Import SWE-bench TestSpec to build proper Docker images
from swebench.harness.test_spec.test_spec import make_test_spec

# Add mini-swe-agent to path
mini_swe_agent_path = Path(__file__).parent / "mini-swe-agent" / "src"
if mini_swe_agent_path.exists():
    sys.path.insert(0, str(mini_swe_agent_path))

from minisweagent.agents.default import DefaultAgent


def get_sb_environment_local(instance: dict):
    """
    Create environment using the SAME Docker approach as mini-swe-agent's SWE-bench support.
    This uses TestSpec-based Docker images, identical to evaluation.
    """
    # Import here to avoid circular dependencies
    from minisweagent.run.extra.swebench import get_sb_environment

    # Default config that will use Docker with TestSpec
    config = {
        "environment": {
            "environment_class": "docker",
        }
    }

    # This function uses TestSpec to determine the correct Docker image
    # and sets up the environment exactly as needed
    return get_sb_environment(config, instance)


def run_instance_local(instance: dict, model_instance, extract_mode: str = "post-agent") -> dict:
    """
    Run a single SWE-bench instance locally with Docker.

    Args:
        instance: Full SWE-bench instance dict with all metadata
        model_instance: Modal-hosted model instance
        extract_mode: Patch extraction mode - "original" or "post-agent" (default)

    Returns:
        Result dict with instance_id, model_patch, exit_status, etc.
    """
    instance_id = instance["instance_id"]
    problem_statement = instance["problem_statement"]

    print(f"Processing instance: {instance_id}")

    try:
        # Create TestSpec to determine proper Docker environment
        # This is the SAME process used by evaluation
        # Use pre-built images from DockerHub namespace
        print(f"Creating TestSpec for proper Docker setup...")
        test_spec = make_test_spec(instance, namespace="starryzhang")
        print(f"TestSpec created: repo={test_spec.repo}, version={test_spec.version}")
        print(f"Using pre-built image: {test_spec.instance_image_key}")

        # Create model wrapper that uses the Modal-hosted model
        class ModalModelWrapper:
            """Wrapper to make Modal model work like mini-swe-agent model."""

            def __init__(self, modal_model_instance):
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

                # Call the model on Modal
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

        # Agent config (same as Modal mode)
        config = {
            "agent": {
                "step_limit": 250,
                "cost_limit": 3.0,
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
        }

        # Create Docker environment using TestSpec (same as evaluation!)
        # This will use the proper SWE-bench Docker image based on the instance
        print(f"Creating local Docker environment from TestSpec...")
        # Add the correct image name from TestSpec so mini-swe-agent uses it
        instance_with_image = {**instance, "image_name": test_spec.instance_image_key}
        env = get_sb_environment_local(instance_with_image)

        # Create model wrapper
        model_wrapper = ModalModelWrapper(model_instance)

        # Create agent
        print(f"Creating agent...")
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
            result = env.execute("git add -A && git diff --cached", cwd="/testbed")
            patch = result["output"].strip()
            print(f"Generated patch: {len(patch)} bytes, return code: {result['returncode']}")

        # Log what we captured
        if patch and len(patch) > 0:
            # Show first few lines to verify it's a proper git diff
            first_lines = '\n'.join(patch.split('\n')[:3])
            print(f"Patch preview: {first_lines}")
        else:
            print(f"Warning: No patch generated!")

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


def run_batch_evaluation_local(
    dataset_name: str = "SWE-bench-Live/SWE-bench-Live",
    split: str = "lite",
    slice_spec: str = "",
    extract_mode: str = "post-agent",
) -> dict:
    """
    Run batch evaluation locally with Docker containers.

    The model still runs on Modal (for GPU), but the agent execution
    happens in local Docker containers.

    Args:
        extract_mode: Patch extraction mode - "original" or "post-agent" (default)
    """
    print(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split)

    # Apply slice
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        start = values[0] if len(values) > 0 and values[0] is not None else 0
        end = values[1] if len(values) > 1 and values[1] is not None else len(dataset)
        dataset = dataset.select(range(start, end))

    print(f"Processing {len(dataset)} instances")
    print("Model runs on Modal GPUs, agent executes in local Docker")
    print("Make sure Docker is installed and running!")
    print(f"Patch extraction mode: {extract_mode}")

    # Create Modal model instance (this will be reused for all instances)
    print("\nInitializing Modal-hosted model...")

    # When called from main() with --docker-mode local, we're always in Modal context
    # Just instantiate the model directly - Modal handles the context
    model_instance = QwenModel()

    # Process instances sequentially (can be parallelized later)
    results = []
    for i, inst in enumerate(dataset):
        print(f"\n{'='*60}")
        print(f"Instance {i+1}/{len(dataset)}")
        print(f"{'='*60}")
        result = run_instance_local(inst, model_instance, extract_mode)
        results.append(result)

    # Summary
    successful = sum(1 for r in results if r.get("success", False))
    total_calls = sum(r.get("model_calls", 0) for r in results)
    total_duration = sum(r.get("duration", 0) for r in results)

    print(f"\n{'='*60}")
    print(f"Batch Evaluation Complete!")
    print(f"Total: {len(results)} | Success: {successful} | Failed: {len(results) - successful}")
    print(f"Total model calls: {total_calls}")
    print(f"Total duration: {total_duration/60:.1f} minutes")
    print(f"Cost: $0 for API + Modal GPU time")
    print(f"{'='*60}\n")

    return {r["instance_id"]: r for r in results}

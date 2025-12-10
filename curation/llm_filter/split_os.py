#!/usr/bin/env python3
import json
import argparse
import re
from launch.utilities.llm import LLMProvider
from fire import Fire
from typing import Literal
import json
from langchain_core.messages import HumanMessage, SystemMessage

# Keywords that indicate possible Windows-specific problems
WINDOWS_KEYWORDS = [
    "windows", "win32", "win64", 
    "visual studio", "powershell", "cmd.exe", "msvc",
    "winerror", "dll", "registry", "com object",
    "directx", "wsl"
]

PROMPT_TEMPLATE = """You are a classifier. 
Decide if the following bug/problem statement is **Windows-specific** or **General (cross-platform / not limited to Windows)**.
Return only one label: "windows" for problems that only occur on Windows or "general" for problems not limited to Windows.

Problem statement:
---
{problem}
---
Now output your answer. You answer should be only one word, "windows" or "general".
Answer:"""

class Judge:
    def __init__(self, provider: str, model_name: str):
        self.llm = LLMProvider(provider, model_name = model_name, temperature = None)

    def call_llm(self, problem_statement: str) -> str:
        """Call LLM to classify problem statement when Windows keyword is detected."""

        prompt = PROMPT_TEMPLATE.format(problem=problem_statement)

        test_chat_message = [{"role": "user", "content": prompt}]
        response = self.llm.invoke([
            SystemMessage("You are an expert software engineer."),
            HumanMessage(prompt),
        ]).content

        label = response.strip().lower()
        print(label)
        if "windows" in label:
            return "windows"
        return "general"


    def classify_problem(self, problem_statement: str) -> str:
        """Keyword prefilter: only call LLM if statement mentions Windows-related terms."""
        for kw in WINDOWS_KEYWORDS:
            if kw in problem_statement.lower():
                return self.call_llm(problem_statement)
        return "general"


    def process_file(self, input_file, windows_file, general_file):
        with open(input_file, "r") as f:
            lines = f.readlines()

        with open(windows_file, "w") as windows_out, open(general_file, "w") as general_out:
            for line in lines:
                task = json.loads(line)
                problem_statement = task.get("problem_statement", "")

                label = self.classify_problem(problem_statement)

                if label == "windows":
                    windows_out.write(json.dumps(task) + "\n")
                else:
                    general_out.write(json.dumps(task) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to input JSONL file with tasks")
    parser.add_argument("--windows_file", default="windows.jsonl", help="Output file for Windows-specific tasks")
    parser.add_argument("--general_file", default="general.jsonl", help="Output file for general tasks")
    parser.add_argument("--llm_provider", choices = ["AOAI", "OpenAI", "Anthropic"], default="AOAI", help="LLM provider")
    parser.add_argument("--model_name", default="gpt-5-20250807", help="model name")

    args = parser.parse_args()
    judge = Judge(args.llm_provider, args.model_name)
    judge.process_file(args.input_file, args.windows_file, args.general_file)


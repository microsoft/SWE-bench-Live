import sys
sys.path.insert(0, "../launch")
from launch.utilities.llm import LLMProvider
from fire import Fire
from typing import Literal
import json
from langchain_core.messages import HumanMessage, SystemMessage

class Judge:
    def __init__(self, provider: str, model_name: str):
        self.llm = LLMProvider(provider, model_name = model_name, temperature = None)

    def verify(self, description, gold_patch, test_patch) -> bool:
        if len(description) > 100000:
            print(0)
            return False
        if len(test_patch) > 300000:
            print("Warning! test patch truncated due to length.")
            test_patch = test_patch[:300000] + "......truncated."
        test_prompt = f"Testcases to evaluate whether the problem is solved: \n{test_patch}\n==============================================" if test_patch.strip() else ""
        solution_prompt = f"The ground truth patch to solve the problem: \n{gold_patch}\n=============================================="
        prompt = f"""
    This is a SWE task for coding agents to solve. 
    But we are evaluating the quality of this task.

    Let's look at the task first.
    Task description:
    {description}
    ==============================================
    {test_prompt if test_prompt.strip() else solution_prompt}

    You need to classify the task into the following categories:
    1. The description is too vague so we cannot infer the {'evaluation tests' if test_prompt.strip() else 'solution patch'} from the task description.
    2. The {'evaluation tests' if test_prompt.strip() else 'solution patch'} contain requirements not required in the task description, so if the coding agent only sees the task description it is unable to meet all the requirements in the {'evaluation tests' if test_prompt.strip() else 'solution patch'}.
    3. The decription already tells how to implement the solution, either in natural language or in code, so the task is too easy.
    4. The task does not have these problems.

    Your answer should be only one category number: 1,2,3 or 4. Be cautious to output 1,2,3 because it would delete this task. These tasks are expensive to create, so be tolerant and always output 4 if you are not sure.
    Your answer:
    """
        for i in range(3):
            response = self.llm.invoke([
                SystemMessage("You are an expert software engineer."),
                HumanMessage(prompt),
            ]).content
            if "4" in response:
                print(4)
                return True
        print(response)
        return False

def main(input_dir: str, 
            output_dir: str, 
            llm_provider: Literal["AOAI", "OpenAI", "Anthropic"],
            model_name: str = "gpt-5-20250807"):
    judge = Judge(llm_provider, model_name)
    with open(input_dir) as f:
        original=[json.loads(i) for i in f]
    filtered=[i for i in original \
            if judge.verify(i["problem_statement"], i["patch"], i["test_patch"])]
    print(f"Retained {len(filtered)} / {len(original)} instances.")
    with open(output_dir, "w") as f:
        for i in filtered:
            f.write(json.dumps(i)+"\n")

if __name__ == "__main__":
    Fire(main)

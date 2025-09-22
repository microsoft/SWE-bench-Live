import io
import sys
import traceback
import json
from functools import wraps
from typing import Callable

def capture_output(func: Callable) -> Callable:
    """Decorator to capture stdout/stderr and handle exceptions during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> str:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = buf_out, buf_err
            # TODO: wrap the execution in some safe container to avoid dangerous LLM hallucinations
            result = func(*args, **kwargs)
            return result if result else {}
        except Exception:
            # Combine redirected stderr + traceback
            captured_err = buf_err.getvalue()
            tb = traceback.format_exc()
            return f"stderr:\n{captured_err}\ntraceback:\n{tb}"
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    return wrapper

@capture_output
def run_get_pertest_cmd(script: str, test_name_list: list[str]) -> dict[str, str]:
    # Create a namespace for exec
    namespace = {}
    exec(script, namespace)
    
    # Check if get_pertest_cmd was defined in the script
    if 'get_pertest_cmd' not in namespace:
        raise NotImplementedError("Script must define 'get_pertest_cmd' function")
    
    # Call the function from the namespace
    return namespace['get_pertest_cmd'](test_name_list)

@capture_output
def run_parser(script: str, log: str) -> dict[str, str]:
    # Create a namespace for exec
    namespace = {}
    exec(script, namespace)
    
    # Check if parser was defined in the script
    if 'parser' not in namespace:
        raise NotImplementedError("Script must define 'parser' function")
    
    # Call the function from the namespace
    return namespace['parser'](log)
import regex  # note: install via "pip install regex"
import json
import logging
"""
LLM can return a JSON object like this:
```json
{
  "is_valid": true
}
```

We want to extract the JSON object from the response.   
"""

# Get logger
logger = logging.getLogger('utils')

def convert_json_values(obj):
    """
    Recursively convert string values in a JSON object to their proper Python types.
    Handles:
    - "true"/"false" -> bool
    - Numeric strings -> int/float
    - Lists and nested objects
    """
    if isinstance(obj, dict):
        return {key: convert_json_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_json_values(item) for item in obj]
    elif isinstance(obj, str):
        # Convert boolean strings
        if obj.lower() == "true":
            return True
        elif obj.lower() == "false":
            return False
        # Convert numeric strings
        try:
            if '.' in obj:
                return float(obj)
            return int(obj)
        except (ValueError, TypeError):
            return obj
    return obj


def extract_json_from_response(response: str):
    """
    Extract a JSON object from a response string.
    First, try to extract a JSON code block (```json ... ```).
    If not found, use a recursive regex to match a balanced JSON object.
    """
    # 1. Try to find a JSON code block
    code_block_pattern =r'```(?:json)?\s*\n?(.*?)\n?```'
    code_block_match = regex.search(code_block_pattern, response, flags=regex.DOTALL | regex.IGNORECASE)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError("Found JSON code block but failed to parse it") from e
    
    # 2. Fallback: try to find a balanced JSON object anywhere in the text.
    # This recursive regex pattern matches a string that starts with { and has properly nested {}.
    balanced_pattern = r'(?P<brace>\{(?:[^{}]+|(?&brace))*\})'
    balanced_match = regex.search(balanced_pattern, response)
    if balanced_match:
        json_str = balanced_match.group("brace")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError("Found a balanced JSON object but failed to parse it") from e

    # 3. If neither approach works, raise an error.
    raise ValueError("No valid JSON object found in the response")

import re
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

def extract_json_from_response(response:str):
    """
    Extract and parse JSON from an LLM response.
    Handles both code blocks and raw JSON.
    Converts string values to appropriate Python types.
    """
    logger.debug(f"Attempting to extract JSON from response:\n{response}")
    
    # First try to find JSON within ```json blocks, being lenient with whitespace and backticks
    json_block_match = re.search(r'```\s*json\s*\n?(.*?)\n?\s*```', response, re.DOTALL | re.IGNORECASE)
    if json_block_match:
        try:
            json_str = json_block_match.group(1).strip()
            logger.debug(f"Found JSON block:\n{json_str}")
            
            # Handle double-escaped backslashes in LaTeX commands
            json_str = json_str.replace('\\\\', '\\')
            logger.debug(f"After escape handling:\n{json_str}")
            
            try:
                parsed = json.loads(json_str)
                logger.debug(f"Successfully parsed JSON:\n{json.dumps(parsed, indent=2)}")
                return convert_json_values(parsed)
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode error: {str(e)}")
                # Try parsing with raw string
                try:
                    parsed = json.loads(json_str.encode('utf-8').decode('unicode-escape'))
                    logger.debug(f"Successfully parsed JSON with unicode escape:\n{json.dumps(parsed, indent=2)}")
                    return convert_json_values(parsed)
                except json.JSONDecodeError as e2:
                    logger.debug(f"JSON decode error with unicode escape: {str(e2)}")
                    raise
        except Exception as e:
            logger.debug(f"Failed to parse JSON block: {str(e)}")
            pass

    # If no valid JSON in code blocks, look for any {...} patterns
    brace_matches = re.findall(r'\{[^{]*?\}', response, re.DOTALL)
    logger.debug(f"Found {len(brace_matches)} brace patterns")
    
    # If only one {...} pattern exists, try parsing it
    if len(brace_matches) == 1:
        try:
            match = brace_matches[0].replace('\\\\', '\\')
            logger.debug(f"Attempting to parse single brace pattern:\n{match}")
            return convert_json_values(json.loads(match))
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse single brace pattern: {str(e)}")
            pass
    
    # If multiple {...} patterns exist, try each one
    for i, match in enumerate(brace_matches):
        try:
            match = match.replace('\\\\', '\\')
            logger.debug(f"Attempting to parse brace pattern #{i+1}:\n{match}")
            return convert_json_values(json.loads(match))
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse brace pattern #{i+1}: {str(e)}")
            continue
    
    # If no valid JSON found, raise an exception
    logger.error("No valid JSON object found in the response")
    raise ValueError("No valid JSON object found in the response")
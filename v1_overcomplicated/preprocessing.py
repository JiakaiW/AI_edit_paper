import json
from ollama import chat
import re
from logging_config import setup_logging
import logging
from utils import extract_json_from_response

# Get logger for preprocessing
logger = logging.getLogger('preprocessing')

def extract_coarse_sentence_from_queue(text, max_chunk=2000):
    """
    Extracts the first complete sentence from the beginning of the text.
    A chunk (up to max_chunk characters) is sent to the LLM, which returns
    the first complete sentence exactly as it appears in the text.
    The end_index is then computed in Python by finding the sentence in the chunk.
    
    Returns:
        tuple: (sentence, end_index) where sentence is the extracted coarse sentence
               and end_index is its position in the text
    """
    chunk = text[:max_chunk]
    prompt = f"""
You are a text segmentation assistant for a LaTeX document.
Please extract the first complete sentence exactly as it appears in the text below.
A complete sentence ends with a period (.), question mark (?) or exclamation mark (!).
Include any LaTeX commands that are part of the sentence.
Return only the sentence with no additional commentary.

Text:
\"\"\"{chunk}\"\"\"
"""
    logger.debug(f"Extracting coarse sentence from chunk:\n{chunk}")
    
    try:
        response = chat(
            model="llama3.1",
            messages=[{'role': 'user', 'content': prompt}],
        )
        
        try:
            sentence = response['message']['content'].strip()
            logger.debug(f"LLM extracted sentence:\n{sentence}")
            
            # Use Python to compute the end index of the sentence in the chunk
            idx = chunk.find(sentence)
            if idx == -1:
                error_msg = f"Could not find exact sentence match in chunk: Response was:\n{response['message']['content']}"
                logger.warning(error_msg)
                # If not found, assume the entire chunk was returned
                end_index = len(chunk)
            else:
                end_index = idx + len(sentence)
                logger.debug(f"Found sentence location: index {idx}, ending at {end_index}")
            return sentence, end_index
            
        except KeyError as e:
            error_msg = f"Failed to extract sentence from response: {str(e)}\nFull response was:\n{response}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
    except Exception as e:
        error_msg = f"Error during coarse sentence extraction: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def extract_fine_sentence_from_coarse_sentence(coarse_sentence):
    """
    Enhances coarse sentences by removing or replacing complex LaTeX commands.
    Preserves inline math while removing complex environments.
    
    Returns:
        dict: Analysis result containing the fine sentence and processing info
    """
    logger.debug(f"Processing coarse sentence for fine extraction:\n{coarse_sentence}")
    
    prompt = f"""
The following sentence from a LaTeX document, it may contain complex LaTeX environments, section commands, and other LaTeX commands such as \\begin{{itemize}}, \\begin{{enumerate}}, \\begin{{align}}, etc.
Your task is to remove or replace the complex LaTeX environments and section commands. However, keep the inline math ($...$) and simple commands such as $\\ket{{f}}$$. So that the sentence is still readable and understandable. 

Do not reply methodological commentary, do not return how to write a program to do this, do not return any other text than the fine-grained sentence. It is very unlikely that you need to modify anything in the sentence at all, unless it is far away from natural language. 

Return a JSON object with exactly two keys:
  "fine_sentence": the processed sentence with complex LaTeX removed
  "should_check": boolean indicating if grammar checking is needed (false if the entire sentence is pure math/commands)

Input sentence:
\"\"\"{coarse_sentence}\"\"\"

Return in the following format:
```json
{{
  "fine_sentence": "<processed text>",
  "should_check": true/false
}}
```
"""
    try:
        response = chat(
            model="llama3.1",
            messages=[{'role': 'user', 'content': prompt}],
        )
               
        try:
            raw_response = response['message']['content']
            logger.debug(f"""Raw LLM response received:
========== BEGIN RESPONSE ==========
{raw_response}
========== END RESPONSE ==========
""")
            
            # Try to extract JSON and log the extracted part
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
                if json_match:
#                     logger.debug(f"""Found JSON block:
# ========== BEGIN JSON ==========
# {json_match.group(1)}
# ========== END JSON ==========
# """)              
                    logger.debug(f"Found JSON block successfully")
                    
                else:
                    logger.debug("No ```json block found in response")
                    
                # Try to find any {...} patterns
                brace_matches = re.findall(r'\{.*?\}', raw_response, re.DOTALL)
                if brace_matches:
                    for i, match in enumerate(brace_matches):
                        logger.debug(f"""Found JSON-like pattern #{i+1}:
========== BEGIN PATTERN ==========
{match}
========== END PATTERN ==========
""")
                else:
                    logger.debug("No {...} patterns found in response")
            except Exception as e:
                logger.error(f"Error while analyzing response format: {str(e)}")
            
            result = extract_json_from_response(raw_response)
            logger.debug(f"Successfully parsed JSON result: \n{json.dumps(result, indent=2)}")
            return result
            
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            error_msg = f"""Failed to parse fine sentence extraction response.
Error type: {type(e).__name__}
Error message: {str(e)}

Raw LLM response:
========== BEGIN RESPONSE ==========
{response.get('message', {}).get('content', 'No content found in response')}
========== END RESPONSE ==========
"""
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
    except Exception as e:
        error_msg = f"Error during fine sentence extraction: {str(e)}"
        logger.error(f"error_msg: {error_msg}, prompt: {prompt}, raw_response:{raw_response}")
        raise RuntimeError(error_msg) from e

def double_check_preprocessing(coarse_sentence, fine_sentence):
    """
    Verifies that the fine sentence aligns properly with the coarse sentence
    and no critical information was lost in preprocessing.
    
    Returns:
        dict: Verification result with potentially corrected fine sentence
    """
    logger.debug(f"Double checking preprocessing:\nCoarse: {coarse_sentence}\nFine: {fine_sentence}")
    
    prompt = f"""
You are a LaTeX verification assistant. Compare the following coarse and fine sentences:

Coarse: \"\"\"{coarse_sentence}\"\"\"
Fine: \"\"\"{fine_sentence}\"\"\"

Verify that:
1. No critical meaning was lost in the fine sentence
2. Inline math and simple LaTeX commands are preserved
3. Only complex environments and section commands were removed

Do not reply methodological commentary, do not return how to write a program to do this, do not return any other text than the JSON object.

Return a JSON object with:
  "is_valid": boolean indicating whether the fine sentence is acceptable

Return the JSON object in the following format:
```json
{{
  "is_valid": true/false
}}
```
"""
    
    try:
        response = chat(
            model="llama3.1",
            messages=[{'role': 'user', 'content': prompt}],
        )
        
        try:
            result = extract_json_from_response(response['message']['content'])
            logger.debug(f"Verification result: \n{json.dumps(result, indent=2)}")
            return result
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            error_msg = f"Failed to parse verification response: {str(e)}\nFull response was:\n{response['message']['content']}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
    except Exception as e:
        error_msg = f"Error during verification: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def calculate_coarse_sentence_length(coarse_sentence):
    """
    Calculates lengths of both coarse and fine sentences.
    
    Returns:
        dict: Length information for both sentences
    """
    length = len(coarse_sentence)
    logger.debug(f"Calculated coarse sentence length: {length}")
    return {
        "coarse_length": length
    }

def preprocessing_main(latex_file):
    """
    Main preprocessing function that coordinates the entire preprocessing workflow.
    
    Returns:
        list: List of JSON objects containing coarse and fine sentences
    """
    logger.info(f"Starting preprocessing of file: {latex_file}")
    
    with open(latex_file, 'r', encoding='utf-8') as f:
        document = f.read()
    
    logger.info(f"Loaded document of length: {len(document)}")
    
    results = []
    while document.strip():
        initial_sentence_length_upper_bound = 500
        max_attempts = 5
        attempt = 0
        verification = {"is_valid": False}
        
        logger.info("Starting new sentence processing cycle")
        
        while not verification["is_valid"] and attempt < max_attempts:
            logger.debug(f"Attempt {attempt + 1}/{max_attempts} with length bound {initial_sentence_length_upper_bound}")
            
            # Step 1: Extract coarse sentence
            coarse_sentence, end_index = extract_coarse_sentence_from_queue(document, initial_sentence_length_upper_bound)
            
            # Step 2: Create fine sentence
            fine_result = extract_fine_sentence_from_coarse_sentence(coarse_sentence)
            fine_sentence = fine_result["fine_sentence"]
            
            # Step 3: Double-check preprocessing
            verification = double_check_preprocessing(coarse_sentence, fine_sentence)
            attempt += 1
            initial_sentence_length_upper_bound += 200
            
            if not verification["is_valid"]:
                logger.warning(f"Verification failed: Attempt {attempt}")
        
        # If all attempts failed, use the last attempt
        if not verification["is_valid"]:
            logger.warning(f"All verification attempts failed: Using coarse sentence as is")
            fine_sentence = coarse_sentence
            fine_result["should_check"] = True
        
        # Step 4: Calculate lengths
        lengths = calculate_coarse_sentence_length(coarse_sentence)
        
        # Store results
        result_entry = {
            "coarse_sentence": coarse_sentence,
            "fine_sentence": fine_sentence,
            "should_check": fine_result["should_check"],
            "lengths": lengths
        }
        results.append(result_entry)
        logger.debug(f"Added result entry:\n{json.dumps(result_entry, indent=2)}")
        
        # Advance document position
        document = document[end_index:]
        logger.debug(f"Remaining document length: {len(document)}")
        
        # Safety check for very short remaining text
        if len(document.strip()) < 3:
            logger.info("Reached end of document (less than 3 characters remaining)")
            break
    
    logger.info(f"Preprocessing complete. Processed {len(results)} sentences.")
    return results


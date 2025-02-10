import json
from ollama import chat
import re
from logging_config import setup_logging
import logging
from utils import extract_json_from_response
from models import PreprocessingVerification

# Get logger for preprocessing
logger = logging.getLogger('preprocessing')

def extract_sentence_from_queue(text, max_chunk=2000):
    """
    Extracts the first complete sentence from the beginning of the text.
    A chunk (up to max_chunk characters) is sent to the LLM, which returns
    the first complete sentence exactly as it appears in the text.
    
    Returns:
        tuple: (sentence, end_index) where sentence is the extracted sentence
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
{chunk}
"""
    logger.debug(f"Extracting sentence from chunk:\n{chunk}")
    
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
            error_msg = f"Failed to extract sentence from response: {str(e)}\nFull response was:\n{response['message']['content']}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
    except Exception as e:
        error_msg = f"Error during sentence extraction: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def double_check_preprocessing(sentence):
    """
    Verifies this is the first sentence in the chunk to ensure accuracy and prevent errors.
    
    Returns:
        dict: Verification result with validation status
    """
    logger.debug(f"Double checking preprocessing for sentence: {sentence}")
    
    prompt = f"""
You are a LaTeX verification assistant. Verify that the following is a complete, valid sentence:

Sentence: {sentence}

Verify that:
1. This is a complete sentence with proper beginning and end"""
    
    response = chat(
        model="llama3.1",
        messages=[{'role': 'user', 'content': prompt}],
        format=PreprocessingVerification.model_json_schema()
    )
    
    try:
        raw_response = response['message']['content']
        result = json.loads(raw_response)
        logger.debug(f"Verification result: \n{json.dumps(result, indent=2)}")
        return result
    except:
        return {"is_valid": False}


def calculate_sentence_length(sentence):
    """
    Calculates length of the sentence.
    
    Returns:
        dict: Length information for the sentence
    """
    length = len(sentence)
    logger.debug(f"Calculated sentence length: {length}")
    return {
        "length": length
    }

def preprocessing_main(latex_file):
    """
    Main preprocessing function that coordinates the entire preprocessing workflow.
    
    Returns:
        list: List of JSON objects containing sentences and their metadata
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
            
            # Step 1: Extract sentence
            sentence, end_index = extract_sentence_from_queue(document, initial_sentence_length_upper_bound)
            
            # Step 2: Double-check preprocessing
            verification = double_check_preprocessing(sentence)
            attempt += 1
            initial_sentence_length_upper_bound += 200
            
            if not verification["is_valid"]:
                logger.warning(f"Verification failed: Attempt {attempt}")
        
        # If all attempts failed, use the last attempt
        if not verification["is_valid"]:
            logger.warning(f"All verification attempts failed: Using sentence as is")
        
        # Step 3: Calculate length
        lengths = calculate_sentence_length(sentence)
        
        # Store results
        result_entry = {
            "sentence": sentence,
            "length": lengths["length"]
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


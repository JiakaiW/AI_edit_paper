import json
from ollama import chat
import re
from logging_config import setup_logging
from utils import extract_json_from_response

import logging
# Get logger for correction
logger = logging.getLogger('correction')

def grammar_correct_proposal(sentence, model="llama3.1"):
    """
    Single Sentence Grammar Correction Proposal step.
    Uses LLM to analyze and correct grammatical errors in a single sentence.
    
    Args:
        sentence: The sentence to correct
        model: The LLM model to use
        
    Returns:
        dict: JSON containing the proposed corrections and analysis
    """
    logger.debug(f"Proposing grammar corrections for sentence:\n{sentence}")
    
    prompt = f"""You are a grammar checker for LaTeX academic papers.
Analyze and correct any grammatical errors in the following sentence.
Preserve all LaTeX commands, math expressions (delimited by $ or $$), and citations.
Return a JSON object with the following keys:
  - "corrected_sentence": the grammatically corrected version
  - "confidence": float between 0-1 indicating confidence in the corrections
  - "explanation": brief explanation of changes (if any)

Input sentence: \"\"\"{sentence}\"\"\"

Return the JSON object in the following format:
```json
{
  "corrected_sentence": "...",
  "confidence": 0.0,
  "explanation": "..."
}
```"""
    
    try:
        response = chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
        )
        
        try:
            raw_response = response['message']['content']
            logger.debug(f"Raw LLM response:\n{raw_response}")
            result = extract_json_from_response(raw_response)
            logger.debug(f"Grammar correction proposal: \n{json.dumps(result, indent=2)}")
            return result
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            error_msg = f"""Failed to parse grammar correction response.
Error: {str(e)}
Raw LLM response:
{response.get('message', {}).get('content', 'No content found in response')}

Full response object:
{json.dumps(response, indent=2)}"""
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
    except Exception as e:
        error_msg = f"Error during grammar correction: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def incorporate_corrections(coarse_sentence, original_fine, corrected_fine, model="llama3.1"):
    """
    Incorporates grammar corrections from the fine sentence back into the coarse sentence.
    
    Args:
        coarse_sentence: Original sentence with all LaTeX commands
        original_fine: Original preprocessed sentence
        corrected_fine: Grammar-corrected version of the fine sentence
        model: The LLM model to use
        
    Returns:
        dict: JSON containing the final corrected sentence and metadata
    """
    logger.debug(f"Incorporating corrections:\nCoarse: {coarse_sentence}\nOriginal Fine: {original_fine}\nCorrected Fine: {corrected_fine}")
    
    prompt = f"""You are a LaTeX grammar correction expert.
Compare the following sentences and integrate grammar corrections into the original LaTeX:

1. Original LaTeX: \"\"\"{coarse_sentence}\"\"\"
2. Original Preprocessed: \"\"\"{original_fine}\"\"\"
3. Corrected Preprocessed: \"\"\"{corrected_fine}\"\"\"

Your task:
1. Identify grammar corrections between sentences 2 and 3
2. Apply these corrections to sentence 1 while preserving ALL LaTeX commands in sentence 1
3. Return a JSON object with:
   - "final_sentence": the corrected LaTeX with all commands preserved
   - "parsing_error": boolean indicating if parsing failed

Return the JSON object in the following format:
```json
{
  "final_sentence": "...",
  "parsing_error": false
}
```"""
    
    try:
        response = chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
        )
        
        try:
            raw_response = response['message']['content']
            logger.debug(f"Raw LLM response:\n{raw_response}")
            result = extract_json_from_response(raw_response)
            result["parsing_error"] = False
            logger.debug(f"Incorporation result: \n{json.dumps(result, indent=2)}")
            return result
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            error_msg = f"""Failed to parse incorporation response.
Error: {str(e)}
Raw LLM response:
{response.get('message', {}).get('content', 'No content found in response')}

Full response object:
{json.dumps(response, indent=2)}"""
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
    except Exception as e:
        error_msg = f"Error during incorporation: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def quality_assurance_check(original_sentence, corrected_sentence, model="llama3.1"):
    """
    Quality assurance step to ensure grammar corrections are valid and maintain meaning.
    
    Args:
        original_sentence: The original LaTeX sentence before corrections
        corrected_sentence: The sentence after grammar corrections
        model: The LLM model to use
        
    Returns:
        dict: JSON containing quality assessment results
    """
    logger.debug(f"Performing QA check:\nOriginal: {original_sentence}\nCorrected: {corrected_sentence}")
    
    prompt = f"""You are a LaTeX quality assurance expert.
Compare the following original and corrected sentences to ensure grammar corrections are valid
and do not alter the original meaning:

Original: \"\"\"{original_sentence}\"\"\"
Corrected: \"\"\"{corrected_sentence}\"\"\"

Analyze for:
1. Grammar improvement without meaning change
2. Preservation of technical accuracy
3. LaTeX command integrity

Return a JSON object with:
  - "is_valid": boolean indicating if corrections are acceptable
  - "maintains_meaning": boolean indicating if original meaning is preserved
  - "technical_accuracy": boolean indicating if technical content is preserved
  - "concerns": list of any identified issues (empty if none)

Return the JSON object in the following format:
```json
{
  "is_valid": true/false,
  "maintains_meaning": true/false,
  "technical_accuracy": true/false,
  "concerns": []
}
```"""
    
    try:
        response = chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
        )
        
        try:
            raw_response = response['message']['content']
            logger.debug(f"Raw LLM response:\n{raw_response}")
            result = extract_json_from_response(raw_response)
            logger.debug(f"QA check result: \n{json.dumps(result, indent=2)}")
            return result
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            error_msg = f"""Failed to parse QA response.
Error: {str(e)}
Raw LLM response:
{response.get('message', {}).get('content', 'No content found in response')}

Full response object:
{json.dumps(response, indent=2)}"""
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
    except Exception as e:
        error_msg = f"Error during QA check: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def grammar_correction_main(preprocessed_sentences, model="llama3.1"):
    """
    Main grammar correction function that processes the entire preprocessed paper.
    
    Args:
        preprocessed_sentences: List of dicts containing coarse and fine sentences
        model: The LLM model to use
        
    Returns:
        list: List of corrected sentences with metadata
    """
    logger.info(f"Starting grammar correction on {len(preprocessed_sentences)} sentences")
    corrected_results = []
    
    for i, sentence_info in enumerate(preprocessed_sentences, 1):
        logger.info(f"Processing sentence {i}/{len(preprocessed_sentences)}")
        
        preprocessing_proposed_skipping = sentence_info.get("should_check", True)
        preprocessing_proposed_skipping_didnt_pass_quality_assurance = False
        QA_passed = False
        
        max_attempts_in_getting_checked_correction = 5
        attempt_in_getting_checked_correction = 0
        
        while not QA_passed and attempt_in_getting_checked_correction < max_attempts_in_getting_checked_correction:
            logger.debug(f"Attempt {attempt_in_getting_checked_correction + 1}/{max_attempts_in_getting_checked_correction}")
            
            final_version = {
                "final_sentence": sentence_info["coarse_sentence"],
            }
            
            if preprocessing_proposed_skipping and not preprocessing_proposed_skipping_didnt_pass_quality_assurance:
                logger.info("Preprocessing suggested skipping this sentence")
                continue
            else:
                # Step 1: Get grammar correction proposal
                corrector_confident = False
                max_attempts_in_getting_confident_correction = 4
                attempt_in_getting_confident_correction = 0
                
                while not corrector_confident and attempt_in_getting_confident_correction < max_attempts_in_getting_confident_correction:
                    logger.debug(f"Grammar correction attempt {attempt_in_getting_confident_correction + 1}/{max_attempts_in_getting_confident_correction}")
                    correction = grammar_correct_proposal(
                        sentence_info["fine_sentence"],
                        model=model
                    )
                    corrector_confident = float(correction["confidence"]) > 0.5
                    if not corrector_confident:
                        logger.debug(f"Low confidence in correction: {correction['confidence']}")
                    attempt_in_getting_confident_correction += 1
                
                # Step 2: Incorporate corrections
                incorporation_accepted = False
                max_attempts_in_incorporating_corrections = 4
                attempt_in_incorporating_corrections = 0
                
                while not incorporation_accepted and attempt_in_incorporating_corrections < max_attempts_in_incorporating_corrections:
                    logger.debug(f"Incorporation attempt: {attempt_in_incorporating_corrections + 1}/{max_attempts_in_incorporating_corrections}")
                    final_version = incorporate_corrections(
                        sentence_info["coarse_sentence"],
                        sentence_info["fine_sentence"],
                        correction["corrected_sentence"],
                        model=model
                    )
                    incorporation_accepted = not final_version["parsing_error"]
                    if not incorporation_accepted:
                        logger.debug(f"Incorporation failed: Parsing error occurred")
                    attempt_in_incorporating_corrections += 1
            
            # Step 3: Quality assurance check
            qa_result = quality_assurance_check(
                sentence_info["coarse_sentence"],
                final_version["final_sentence"],
                model=model
            )
            
            if qa_result["is_valid"] and qa_result["maintains_meaning"] and qa_result["technical_accuracy"]:
                has_changes = True
                QA_passed = True
                logger.info(f"QA check passed successfully")
                break
            else:
                logger.warning(f"QA check failed: Concerns were: {qa_result.get('concerns', [])}")
                attempt_in_getting_checked_correction += 1
                if preprocessing_proposed_skipping:
                    preprocessing_proposed_skipping_didnt_pass_quality_assurance = True
        
        # Store the results
        result_entry = {
            "original": sentence_info["coarse_sentence"],
            "final": final_version["final_sentence"],
            "has_changes": has_changes,
            "qa_concerns": qa_result.get("concerns", [])
        }
        corrected_results.append(result_entry)
        logger.debug(f"Added result entry:\n{json.dumps(result_entry, indent=2)}")
    
    logger.info(f"Grammar correction complete. Processed {len(corrected_results)} sentences.")
    return corrected_results

def assemble_corrected_paper(corrected_results):
    """
    Assembles the final corrected paper from the list of corrected sentences.
    
    Args:
        corrected_results: List of dicts containing original and corrected sentences
        
    Returns:
        str: The complete corrected paper
    """
    logger.info("Assembling final paper")
    final_paper = " ".join(result["final"] for result in corrected_results)
    logger.info(f"Final paper length: {len(final_paper)}")
    return final_paper

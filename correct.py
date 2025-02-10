import json
from ollama import chat
import re
from logging_config import setup_logging
import logging
from utils import extract_json_from_response
from models import GrammarCorrection, QualityAssurance

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
    
    prompt = f'''You are a grammar checker for LaTeX academic papers. Note that we use present tense in scientific paper, not past tense.
Analyze and correct obvious grammatical errors in the following sentence.
Do not change any scientific notation, name of quantum systems,LaTeX commands, math expressions (delimited by $ or $$), and citations. 

Input sentence: {sentence}'''
    
    response = chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        format=GrammarCorrection.model_json_schema()
    )
    
    raw_response = response['message']['content']
    logger.debug(f"Raw LLM response:\n{raw_response}")
    result = json.loads(raw_response)
    logger.debug(f"Grammar correction proposal: \n{json.dumps(result, indent=2)}")
    return result

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
    
    prompt = f'''You are a LaTeX quality assurance expert.
Compare the following original and corrected sentences to ensure grammar corrections are valid
and do not alter the original meaning:

Original: {original_sentence}
Corrected: {corrected_sentence}

Analyze for:
1. Grammar improvement without meaning change
2. Preservation of technical accuracy
3. LaTeX command integrity'''
    
    response = chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        format=QualityAssurance.model_json_schema()
    )
    
    raw_response = response['message']['content']
    logger.debug(f"Raw LLM response:\n{raw_response}")
    result = json.loads(raw_response)
    logger.debug(f"QA check result: \n{json.dumps(result, indent=2)}")
    return result

def grammar_correction_main(preprocessed_sentences, model="llama3.1"):
    """
    Main grammar correction function that processes the entire preprocessed paper.
    
    Args:
        preprocessed_sentences: List of dicts containing sentences and their metadata
        model: The LLM model to use
        
    Returns:
        list: List of corrected sentences with metadata
    """
    logger.info(f"Starting grammar correction on {len(preprocessed_sentences)} sentences")
    corrected_results = []
    
    for i, sentence_info in enumerate(preprocessed_sentences, 1):
        logger.info(f"Processing sentence {i}/{len(preprocessed_sentences)}")
        
        original_sentence = sentence_info["sentence"]
        QA_passed = False
        max_attempts = 5
        attempt = 0
        
        while not QA_passed and attempt < max_attempts:
            logger.debug(f"Attempt {attempt + 1}/{max_attempts}")
            
            # Step 1: Get grammar correction proposal
            correction = grammar_correct_proposal(original_sentence, model=model)
            
            if float(correction["confidence"]) <= 0.5:
                logger.debug(f"Low confidence in correction: {correction['confidence']}")
                attempt += 1
                continue
                
            # Step 2: Quality assurance check
            num_attempts = 0
            max_qa_attempts = 5
            qa_yielded_result = False
            while num_attempts < max_qa_attempts:
                qa_result = quality_assurance_check(
                    original_sentence,
                    correction["corrected_sentence"],
                    model=model
                    )
                if 'is_valid' in qa_result and 'maintains_meaning' in qa_result and 'technical_accuracy' in qa_result:
                    if qa_result["is_valid"] and qa_result["maintains_meaning"] and qa_result["technical_accuracy"]:
                        QA_passed = True
                        logger.info(f"QA check passed successfully")
                        break
                    else:
                        logger.warning(f"QA check failed: Concerns were: {qa_result.get('concerns', [])}")
                        attempt += 1
                    qa_yielded_result = True
                if qa_yielded_result:
                    break
            
        # Store the results
        has_changes = original_sentence != correction["corrected_sentence"]
        result_entry = {
            "original": original_sentence,
            "final": correction["corrected_sentence"] if QA_passed else original_sentence,
            "has_changes": has_changes and QA_passed,
            "qa_concerns": qa_result.get("concerns", []) if QA_passed else ["Failed to find valid correction"]
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

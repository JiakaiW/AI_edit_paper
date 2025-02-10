from correct import grammar_correct_proposal, quality_assurance_check
from preprocessing import extract_sentence_from_queue, double_check_preprocessing, calculate_sentence_length
from logging_config import setup_logging
import logging
import os
from datetime import datetime

def process_sentence(document, model="llama3.1"):
    """
    Process a single sentence through preprocessing and correction.
    
    Returns:
        tuple: (processed_sentence_info, remaining_document, end_index)
    """
    initial_sentence_length_upper_bound = 500
    max_attempts = 5
    attempt = 0
    verification = {"is_valid": False}
    
    while not verification["is_valid"] and attempt < max_attempts:
        # Step 1: Extract sentence
        sentence, end_index = extract_sentence_from_queue(document, initial_sentence_length_upper_bound)
        
        # Step 2: Double-check preprocessing
        verification = double_check_preprocessing(sentence)
        attempt += 1
        initial_sentence_length_upper_bound += 200
    
    # If all attempts failed, use the last attempt
    if not verification["is_valid"]:
        logging.warning(f"All verification attempts failed: Using sentence as is")
    
    # Step 3: Calculate length
    lengths = calculate_sentence_length(sentence)
    
    # Step 4: Grammar correction
    QA_passed = False
    max_correction_attempts = 8
    correction_attempt = 0
    
    while not QA_passed and correction_attempt < max_correction_attempts:
        # Get grammar correction proposal
        correction = grammar_correct_proposal(sentence, model=model)
        if "confidence" not in correction:
            logging.warning(f"No confidence score in correction: {correction}")
            correction_attempt += 1
            continue

        if float(correction["confidence"]) <= 0.5:
            logging.debug(f"Low confidence in correction: {correction['confidence']}")
            correction_attempt += 1
            continue
            
        # Quality assurance check
        qa_result = quality_assurance_check(
            sentence,
            correction["corrected_sentence"],
            model=model
        )
        if "is_valid" not in qa_result or "maintains_meaning" not in qa_result or "technical_accuracy" not in qa_result:
            logging.warning(f"QA check failed: Missing required keys in QA result: {qa_result}")
            correction_attempt += 1
            continue
        if qa_result["is_valid"] and qa_result["maintains_meaning"] and qa_result["technical_accuracy"]:
            QA_passed = True
            logging.info(f"QA check passed successfully")
            break
        else:
            logging.warning(f"QA check failed: Concerns were: {qa_result.get('concerns', [])}")
            correction_attempt += 1
    
    # Prepare result
    has_changes = sentence != correction["corrected_sentence"] if QA_passed else False
    result = {
        "original": sentence,
        "final": correction["corrected_sentence"] if QA_passed else sentence,
        "has_changes": has_changes,
        "qa_concerns": qa_result.get("concerns", []) if QA_passed else ["Failed to find valid correction"],
        "length": lengths["length"],
        "explanation": correction.get("explanation", "") if QA_passed else ""
    }
    
    return result, document[end_index:], end_index

def main():
    # Set up logging
    preprocess_logger, correction_logger = setup_logging()
    
    # Create progress directory if it doesn't exist
    progress_dir = "progress"
    if not os.path.exists(progress_dir):
        os.makedirs(progress_dir)
    
    # Create timestamped progress file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_file = os.path.join(progress_dir, f"correction_log_{timestamp}.txt")
    
    # Read input file
    with open("main.tex", 'r', encoding='utf-8') as f:
        document = f.read()
    
    logging.info(f"Loaded document of length: {len(document)}")
    
    # Process sentences one by one
    sentence_count = 0
    total_chars = len(document)
    chars_processed = 0
    
    # Initialize progress file with header
    with open(progress_file, 'w', encoding='utf-8') as f:
        f.write("Grammar Correction Log\n")
        f.write("====================\n\n")
        f.write(f"Started at: {timestamp}\n\n")
    
    while document.strip():
        sentence_count += 1
        logging.info(f"\nProcessing sentence {sentence_count}")
        
        # Process one sentence
        result, document, chars_consumed = process_sentence(document)
        chars_processed += chars_consumed
        
        # Log progress
        progress = (chars_processed / total_chars) * 100
        logging.info(f"Progress: {progress:.2f}% ({chars_processed}/{total_chars} characters)")
        
        # Append to progress file
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"\nSentence {sentence_count}:\n")
            f.write("-" * 40 + "\n")
            if result["has_changes"]:
                f.write("CHANGED:\n")
                f.write(f"Original : {result['original']}\n")
                f.write(f"Corrected: {result['final']}\n")
                if result["explanation"]:
                    f.write(f"Explanation: {result['explanation']}\n")
                if result["qa_concerns"]:
                    f.write(f"QA Notes: {', '.join(result['qa_concerns'])}\n")
            else:
                f.write("NO CHANGES NEEDED:\n")
                f.write(f"{result['original']}\n")
                if result["qa_concerns"]:
                    f.write(f"QA Notes: {', '.join(result['qa_concerns'])}\n")
            f.write("\n")
        
        # Log changes for this sentence
        if result["has_changes"]:
            logging.info("Changes made in this sentence:")
            logging.info(f"Original: {result['original']}")
            logging.info(f"Corrected: {result['final']}")
            if result["explanation"]:
                logging.info(f"Explanation: {result['explanation']}")
        else:
            logging.info("No changes needed for this sentence")
    
    # Write summary at the end
    with open(progress_file, 'a', encoding='utf-8') as f:
        f.write("\nSummary\n")
        f.write("=======\n")
        f.write(f"Total sentences processed: {sentence_count}\n")
        f.write(f"Total characters processed: {chars_processed}\n")
        f.write(f"Completed at: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
    
    logging.info(f"\nProcessing complete. Processed {sentence_count} sentences.")
    logging.info(f"Detailed correction log saved to: {progress_file}")

if __name__ == "__main__":
    main()
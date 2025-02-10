from correct import grammar_correction_main, assemble_corrected_paper
from preprocessing import preprocessing_main
from logging_config import setup_logging

def main():
    # Set up logging
    preprocess_logger, correction_logger = setup_logging()
    
    sentences = preprocessing_main("main.tex")
    corrected_results = grammar_correction_main(sentences)
    
    with open("corrected_main.tex", "w") as f:
        f.write(assemble_corrected_paper(corrected_results))

if __name__ == "__main__":
    main()
# Design Document: Minimalistic Python Script Using LLM for LaTeX Paper Preprocessing

This design outlines a Python script intended to preprocess a LaTeX paper using an LLM (Large Language Model) in a minimalistic manner,
ensuring efficiency and clarity for automated implementation.

# 1. Main Preprocessing Function
- **Function Name:** `preprocessing_main()`
- **Return Type:** `List[json {"sentence": "..."}]`
- **Purpose:** It recursively calls the preprocessing steps to preprocess the whole paper. In each step, it determines the starting point of the next grammar correction step by calculating the length of the sentence, and then consumes the next chunk of the paper. It returns a list of json objects, each containing a sentence. This return object is the preprocessed version of the whole paper.

## Preprocessing Steps

The preprocessing workflow consists of the following steps, each designed for efficiency and clarity:

### a) Sentence Extraction
- **Function Name:** `extract_sentence_from_queue`
- **Purpose:** Breaks down the paper into manageable sentences. Different sentences consecutively make up the whole paper.

### b) Redundancy Check
- **Function Name:** `double_check_preprocessing`
- **Purpose:** Verifies this is the first sentence in the chunk to ensure accuracy and prevent errors.

### c) Sentence Length Calculation
- **Function Name:** `calculate_sentence_length`
- **Purpose:** Calculates lengths of both sentences for optimal processing in subsequent steps. This step is done using python functions not LLM. This length is used to determine the starting point of the next grammar correction step.


# 2. Main Grammar Correction Function
- **Function Name:** `grammar_correction_main()`
- **Purpose:** After preprocessing which turns the whole paper into a list of json objects, it recursively calls the grammar correction steps to correct the whole paper. It returns a list of json objects, each containing a corrected sentence. This return object is the preprocessed version of the whole paper.

## Grammar Correction Steps

### a) Single Sentence Grammar Correction Proposal
- **Function Name:** `grammar_correct_proposal`
- **Objective:** Correct grammatical errors in a single sentence using an LLM.
- **Process:**
  - Input: A single sentence (a segment of the paper).
  - LLM will analyze and correct grammatical errors, producing a JSON-formatted output with error corrections.
- **Output Format:** JSON containing the input sentence corrected for identified grammatical issues.

### b) Quality Assurance of Grammar Corrections
- **Function Name:** `quality_assurance_check`
- **Objective:** Ensure the grammar corrections are correct and not harmful to the original meaning of the sentence.
- **Process:**
  - Input: The original sentence, the sentence with corrected grammar.
  - LLM will compare these chunks to identify areas where corrections are needed or beneficial.
- **Output Format:** JSON containing boolean indicating if the grammar corrections are correct and not harmful to the original meaning of the sentence.

# 3: Assemble the corrected paper from corrected sentences
 - input: The corrected sentences.
 - output: a corrected paper as a string.



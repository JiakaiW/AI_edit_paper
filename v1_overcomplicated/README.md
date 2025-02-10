# Design Document: Minimalistic Python Script Using LLM for LaTeX Paper Preprocessing

This design outlines a Python script intended to preprocess a LaTeX paper using an LLM (Large Language Model) in a minimalistic manner,
ensuring efficiency and clarity for automated implementation.

# 1. Main Preprocessing Function
- **Function Name:** `preprocessing_main()`
- **Return Type:** `List[json {"coarse_sentence": "...", "fine_sentence": "..."}]`
- **Purpose:** It recursively calls the preprocessing steps to preprocess the whole paper. In each step, it determines the starting point of the next grammar correction step by calculating the length of the coarse sentence, and then consumes the next chunk of the paper. It returns a list of json objects, each containing a coarse sentence and the corresponding fine sentence. This return object is the preprocessed version of the whole paper.

## Preprocessing Steps

The preprocessing workflow consists of the following steps, each designed for efficiency and clarity:

### a) Coarse Sentence Extraction
- **Function Name:** `extract_coarse_sentence_from_queue`
- **Purpose:** Breaks down the paper into manageable coarse chunks (coarse sentences). Different coarse chunks consecutively make up the whole paper.

### b) Fine Sentence Polish
- **Function Name:** `extract_fine_sentence_from_coarse_sentence`
- **Purpose:** Enhances coarse sentences by removing or replacing complex LaTeX commands with dummy text, ensuring fine sentences retain original structure. Removes the LaTeX commands such as \begin{figure} and \end{figure} and \section{...} and \subsection{...}.

### c) Redundancy Check
- **Function Name:** `double_check_preprocessing`
- **Purpose:** Verifies the fine sentences align with their respective coarse sentences to ensure accuracy and prevent errors.

### d) Sentence Length Calculation
- **Function Name:** `calculate_coarse_sentence_length`
- **Purpose:** Calculates lengths of both coarse and fine sentences for optimal processing in subsequent steps. This step is done using python functions not LLM. This length is used to determine the starting point of the next grammar correction step.


# 2. Main Grammar Correction Function
- **Function Name:** `grammar_correction_main()`
- **Purpose:** After preprocessing which turns the whole paper into a list of json objects, it recursively calls the grammar correction steps to correct the whole paper. It returns a list of json objects, each containing a coarse sentence and the corresponding fine sentence. This return object is the preprocessed version of the whole paper.

## Grammar Correction Steps

### a) Single Sentence Grammar Correction Proposal
- **Function Name:** `grammar_correct_proposal`
- **Objective:** Correct grammatical errors in a single sentence using an LLM.
- **Process:**
  - Input: A single sentence (a segment of the paper).
  - LLM will analyze and correct grammatical errors, producing a JSON-formatted output with error corrections.
- **Output Format:** JSON containing corrections for identified grammatical issues.

### b) Incorporating Changes into the original coarse chunk
- **Function Name:** `incorporate_corrections`
- **Objective:** Provide the LLM with the original coarse chunk, the original fine chunk, and the fine chunk with corrected grammar. The LLM will then compare these chunks to identify areas where corrections are needed or beneficial, and integrate the changes into the original coarse chunk.
- **Output Format:** Corrected coarse chunk.

### c) Quality Assurance of Grammar Corrections
- **Objective:** Ensure the grammar corrections are correct and not harmful to the original meaning of the sentence.
- **Process:**
  - Input: The original coarse chunk, the coarse chunk with corrected grammar.
  - LLM will compare these chunks to identify areas where corrections are needed or beneficial.
- **Output Format:** JSON containing boolean indicating if the grammar corrections are correct and not harmful to the original meaning of the sentence.

# 3: Assemble the corrected paper from corrected coarse chunks
 - input: The corrected coarse chunks.
 - output: a corrected paper as a string.



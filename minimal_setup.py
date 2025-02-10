from ollama import chat
from langchain_ollama import ChatOllama

import re
from pydantic import BaseModel, Field
import json
import os
from datetime import datetime

class GrammarCorrection(BaseModel):
    corrected_segment: str = Field(description="The corrected version of the original segment")
    confidence: float = Field(description="The confidence score of the correction you made")
    explanation: str = Field(description="The explanation of the correction you made")


def check_grammar(sentences, llm_model = ChatOllama(model="llama3.1", temperature=0)):
    structured_llm = llm_model.with_structured_output(GrammarCorrection, method="json_schema")

    # Create a prompt for grammar correction
    for sent in sentences:
        prompt = rf"""You are a grammar checker for the LaTex draft of an academic paper. Correct the following segment for obvious grammar mistakes. 
        You are not supposed to add quotation marks or modify other LaTex commands. Only return the corrected segment without explanations or additional text.
        Segment: "{sent}"
        Return your answer in a new line, beginning with "Corrected:" """
        
        num_attempts = 0
        max_attempts = 5
        highest_confidence = 0
        correction_with_high_confidence: GrammarCorrection = None
        while num_attempts < max_attempts:
            correction = structured_llm.invoke(prompt)

            if correction.confidence > highest_confidence:
                highest_confidence = correction.confidence
                correction_with_high_confidence = correction
            if highest_confidence > 0.6:
                break
            num_attempts += 1

        yield correction_with_high_confidence.corrected_segment,correction_with_high_confidence.explanation


# def extract_sentences(latex_file):
#     with open(latex_file, 'r', encoding='utf-8') as f:
#         text = f.read()

#     # Split into sentences, remove the space at the begining of each senteces, start from the 4th sentence
#     sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)][2:]
#     #Let's modify stuff like $\ket{{f}}$, $\ket{{e}}$, $\ket{{g}}$ into f, e, g. Let's match "$\ket{ and   }"
#     sentences = [re.sub(r"\$\\ket\{([a-zA-Z]+)\}\$", r"\1", s) for s in sentences]
#     return sentences

def extract_sentences(latex_file):
    with open(latex_file, 'r', encoding='utf-8') as f:
        text = f.read()

    segments = []        # list to hold complete segments
    current_segment = [] # current segment being built as a list of characters
    env_stack = []       # stack to track open LaTeX environments
    brace_count = 0      # counter for unmatched "{" versus "}"

    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        current_segment.append(ch)

        # Check if a LaTeX environment starts here, e.g. "\begin{figure*}"
        if text.startswith(r'\begin{', i):
            m = re.match(r'\\begin\{([^}]+)\}', text[i:])
            if m:
                env_stack.append(m.group(1))
                # Advance index past the matched "\begin{...}" text.
                i += m.end() - 1  # subtract 1 because we already processed the current char
                # Append the rest of the matched text to current_segment.
                # (Alternatively, since we've already appended ch, and the loop will
                # pick up the following characters, you may choose not to append explicitly.)
        # Check for an environment closing, e.g. "\end{figure*}"
        elif text.startswith(r'\end{', i):
            m = re.match(r'\\end\{([^}]+)\}', text[i:])
            if m:
                env_name = m.group(1)
                # Only pop if the closing matches the last open environment.
                if env_stack and env_stack[-1] == env_name:
                    env_stack.pop()
                i += m.end() - 1
        # Update the brace counter.
        elif ch == '{':
            brace_count += 1
        elif ch == '}':
            if brace_count > 0:
                brace_count -= 1

        # Now check if the current character is sentence-ending punctuation.
        if ch in '.!?' and not env_stack and brace_count == 0:
            # Check that the next character is whitespace or end-of-text.
            if i + 1 >= length or text[i+1].isspace():
                # We decide this is a safe segmentation point.
                segment = ''.join(current_segment).strip()
                if segment:
                    segments.append(segment)
                current_segment = []
                # Optionally skip additional whitespace.
                while i + 1 < length and text[i+1].isspace():
                    i += 1
        i += 1

    # Add any trailing text as a segment.
    leftover = ''.join(current_segment).strip()
    if leftover:
        segments.append(leftover)

    # Post-process each segment: for example, replace LaTeX commands like $\ket{f}$ with just "f"
    processed_segments = []
    for s in segments:
        # This regex looks for $ \ket{<letters>} $ and replaces it with the letters.
        # s = re.sub(r"\$\\ket\{([a-zA-Z]+)\}\$", r"\1", s)
        s = re.sub(r'\\', r'\\\\', s)

        processed_segments.append(s)

    return processed_segments


progress_dir = "progress"
if not os.path.exists(progress_dir):
    os.makedirs(progress_dir)
# Create timestamped progress file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
progress_file = os.path.join(progress_dir, f"correction_log_{timestamp}.txt")
with open(progress_file, 'w', encoding='utf-8') as f:
    f.write("Grammar Correction Log\n")
    f.write("====================\n\n")
    f.write(f"Started at: {timestamp}\n\n")


latex_file = "main.tex"
sentences = extract_sentences(latex_file)

# Append to progress file
for original, (corrected, explanation) in zip(sentences, check_grammar(sentences)):
    original = original.replace("\\\\", "\\")
    if original != corrected:
        print(f"Checking original:\n{original}\nhas correection\ncorrected: {corrected}")
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write("CHANGED:\n")
            f.write(f"Original :\n{original}\n")
            f.write(f"Corrected: \n{corrected}\n")
            # f.write(f"Explanation: \n{explanation}\n")
            f.write("\n")
    else:
        print(f"checking:\n{original}\nno correction")



import json
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import argparse
import os

# --- Configuration ---
# Default file paths (can be overridden by command-line arguments)
DEFAULT_REF_FILE = "/home/ubuntu/workspace/codereviewer_llm/datasets/code_refinement/ref-test.jsonl"
DEFAULT_MODEL_OUTPUT_FILE = "/home/ubuntu/workspace/codereviewer_llm/test_results_202505271300.jsonl"
DEFAULT_RESULTS_OUTPUT_FILE = "evaluation_results.json"

def clean_model_output(raw_output_str: str) -> str:
    """
    Cleans the model's output string by removing markdown-style code block fences.
    Handles "```python\ncode\n```", "```\ncode\n```", and just "```code```".
    Also strips leading/trailing whitespace from the extracted code.
    """
    # Regex to find ``` optionally followed by a language, then newline, then content, then ```
    # re.DOTALL (or s flag) makes . match newlines as well
    match = re.search(r"```(?:[a-zA-Z]+\n|\n)?(.*?)\n```", raw_output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback for cases where the output might not be perfectly formatted
    # or if it's already clean (though the prompt implies it's not)
    return raw_output_str.strip()

def calculate_metrics(ref_file: str, model_output_file: str, results_output_file: str):
    """
    Calculates BLEU and Exact Match scores.
    """
    base_lines = []
    references = []
    model_outputs_raw = []

    # 1. Load reference data
    try:
        with open(ref_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if 'new' not in data:
                        print(f"Warning: 'new' field missing in reference file {ref_file} at line {line_num}. Skipping.")
                        continue
                    references.append(data['new'])
                    if 'old' not in data:
                        print(f"Warning: 'old' field missing in reference file {ref_file} at line {line_num}. Skipping.")
                        continue
                    base_lines.append(data['old'])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON in reference file {ref_file} at line {line_num}. Skipping: {line.strip()}")
                except KeyError:
                    print(f"Warning: 'new' key missing in JSON object in reference file {ref_file} at line {line_num}. Skipping.")
    except FileNotFoundError:
        print(f"Error: Reference file not found at {ref_file}")
        return
    except Exception as e:
        print(f"An error occurred while reading {ref_file}: {e}")
        return

    # 2. Load model output data
    try:
        with open(model_output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if 'output' not in data:
                        print(f"Warning: 'output' field missing in model output file {model_output_file} at line {line_num}. Skipping.")
                        continue
                    model_outputs_raw.append(data['output'])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON in model output file {model_output_file} at line {line_num}. Skipping: {line.strip()}")
                except KeyError:
                    print(f"Warning: 'output' key missing in JSON object in model output file {model_output_file} at line {line_num}. Skipping.")
    except FileNotFoundError:
        print(f"Error: Model output file not found at {model_output_file}")
        return
    except Exception as e:
        print(f"An error occurred while reading {model_output_file}: {e}")
        return

    if not references or not model_outputs_raw:
        print("Error: No data loaded from one or both files. Cannot proceed.")
        return

    num_references = len(references)
    num_model_outputs = len(model_outputs_raw)

    if num_references != num_model_outputs:
        print(f"Warning: Mismatch in the number of entries. "
              f"References: {num_references}, Model Outputs: {num_model_outputs}.")
        print("Proceeding with the minimum number of entries.")
        min_len = min(num_references, num_model_outputs)
        references = references[:min_len]
        model_outputs_raw = model_outputs_raw[:min_len]

    if not references: # After potential truncation
        print("Error: No common entries to compare after aligning lengths. Exiting.")
        return

    total_samples = len(references)
    exact_match_count = 0
    total_bleu_score = 0.0
    # Using smoothing function for BLEU as code snippets can be short
    # method1 is a common choice for sentence-level BLEU
    smoother = SmoothingFunction().method1

    individual_results = [] # For potential detailed logging if needed later
    
    for i in range(total_samples):
        ref_code = references[i]
        model_raw_output = model_outputs_raw[i]

        # Clean the model output (remove backticks etc.)
        model_cleaned_code = clean_model_output(model_raw_output)

        # For EM, strip leading/trailing whitespace from the entire block
        # This ensures comparison is fair if one has an extra newline at start/end
        # but content is otherwise identical.
        ref_code_stripped = ref_code.strip()
        model_cleaned_code_stripped = model_cleaned_code.strip()
        
        # 3. Calculate Exact Match (EM)
        is_em = (ref_code_stripped == model_cleaned_code_stripped)
        if is_em:
            exact_match_count += 1

        # 4. Calculate BLEU score
        # Tokenize by whitespace for BLEU.
        # NLTK's sentence_bleu expects a list of reference token lists
        # and a hypothesis token list.
        # ref_tokens = ref_code_stripped.split()
        # model_tokens = model_cleaned_code_stripped.split()
        ref_tokens = word_tokenize(ref_code_stripped)
        model_tokens = word_tokenize(model_cleaned_code_stripped)

        # Handle empty model output gracefully for BLEU (NLTK sentence_bleu returns 0 if hypothesis is empty)
        bleu_score = sentence_bleu([ref_tokens], model_tokens, smoothing_function=smoother)
        total_bleu_score += bleu_score
        
        individual_results.append({
            "id": i,
            "reference_stripped": ref_code_stripped,
            "model_cleaned_stripped": model_cleaned_code_stripped,
            "is_em": is_em,
            "bleu": bleu_score
        })


    # Calculate final metrics
    em_percentage = (exact_match_count / total_samples) * 100 if total_samples > 0 else 0
    average_bleu_score = (total_bleu_score / total_samples) if total_samples > 0 else 0 # BLEU is 0-1

    results_summary = {
        "total_samples_processed": total_samples,
        "exact_match_percentage": round(em_percentage, 4),
        "average_bleu_score": round(average_bleu_score, 6), # BLEU is typically 0-1, sometimes scaled to 0-100
        "average_bleu_score_scaled_100": round(average_bleu_score * 100, 4) # Scaled to 100 for easier reading
    }

    # 5. Output results to a JSON file
    try:
        with open(results_output_file, 'w', encoding='utf-8') as f_out:
            json.dump(results_summary, f_out, indent=4)
        print(f"\nEvaluation results successfully written to: {results_output_file}")
    except IOError:
        print(f"Error: Could not write results to {results_output_file}")
    except Exception as e:
        print(f"An error occurred while writing the results file: {e}")

    print("\n--- Summary ---")
    print(json.dumps(results_summary, indent=4))
    
    # Optional: Save detailed per-sample results
    # with open("detailed_evaluation_results.jsonl", 'w', encoding='utf-8') as f_detailed:
    #     for res in individual_results:
    #         f_detailed.write(json.dumps(res) + "\n")
    # print(f"Detailed per-sample results written to: detailed_evaluation_results.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BLEU and Exact Match for model code generation.")
    parser.add_argument(
        "--ref_file",
        type=str,
        default=DEFAULT_REF_FILE,
        help=f"Path to the reference JSONL file. Default: {DEFAULT_REF_FILE}"
    )
    parser.add_argument(
        "--model_output_file",
        type=str,
        default=DEFAULT_MODEL_OUTPUT_FILE,
        help=f"Path to the model output JSONL file. Default: {DEFAULT_MODEL_OUTPUT_FILE}"
    )
    parser.add_argument(
        "--results_output_file",
        type=str,
        default=DEFAULT_RESULTS_OUTPUT_FILE,
        help=f"Path to save the evaluation results JSON file. Default: {DEFAULT_RESULTS_OUTPUT_FILE}"
    )

    args = parser.parse_args()

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(args.results_output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    print(f"Using Reference File: {args.ref_file}")
    print(f"Using Model Output File: {args.model_output_file}")
    print(f"Saving Results To: {args.results_output_file}")

    calculate_metrics(args.ref_file, args.model_output_file, args.results_output_file)

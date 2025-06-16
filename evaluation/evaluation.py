import pandas as pd
import json
import re
import argparse
from typing import List, Tuple, Dict, Any

def load_ground_file(file_name: str) -> List[Dict[str, Any]]:
    with open(file_name, "r") as f:
        return json.load(f)


def load_pred_file(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)


def extract_values(text_list: List[str]) -> List[str]:
    """
    Extract numeric values or ranges from a list of strings, removing ± uncertainty parts.
    """
    values = []
    for text in text_list:
        cleaned = re.sub(r"\(±[^)]+\)", "", text)
        match = re.match(r"\s*([\d.\-–]+)", cleaned)
        if match:
            values.append(match.group(1).strip())
    return values


def split_and_extract(predicted_parameters: List[str]) -> List[str]:
    """
    Split list of predicted strings by semicolon, strip whitespace, and extract numeric values.
    """
    split_outputs = []
    for val in predicted_parameters:
        parts = [item.strip() for item in str(val).split(';') if item.strip()]
        split_outputs.extend(parts)
    return extract_values(split_outputs)

def compute_scores_exp1(predicted_list: List[str], ground_list: List[str]) -> Tuple[int, int, int]:
    matched_indices = set()
    true_positives = 0

    for pred in predicted_list:
        for i, gold in enumerate(ground_list):
            if i not in matched_indices and pred == gold:
                true_positives += 1
                matched_indices.add(i)
                break

    false_positives = len(predicted_list) - true_positives
    false_negatives = len(ground_list) - true_positives
    return true_positives, false_positives, false_negatives


def compute_scores_exp3(predicted_list: List[str], ground_list: List[str], abstract: str) -> Tuple[int, int, int, int]:
    matched_indices = set()
    true_positives = 0
    hallucination = 0

    for pred in predicted_list:
        for i, gold in enumerate(ground_list):
            if i not in matched_indices and pred == gold:
                true_positives += 1
                matched_indices.add(i)
                break
        if pred not in abstract and pred not in ground_list:
            hallucination += 1

    false_positives = len(predicted_list) - true_positives
    false_negatives = len(ground_list) - true_positives
    return true_positives, false_positives, false_negatives, hallucination


def compute_scores_exp2(predicted_list: List[str]) -> Tuple[int, int]:
    hallucination = 0
    non = 0
    if not predicted_list or predicted_list[0] == 'NONE':
        non = 1
    else:
        hallucination = len(predicted_list)
    return hallucination, non


def evaluation_exp1(scores: Dict[str, int]) -> Tuple[float, float, float]:
    tp, fp, fn = scores['true_positives'], scores['false_positives'], scores['false_negatives']
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


def evaluation_exp3(scores: Dict[str, int]) -> Tuple[float, float, float, float]:
    tp, fp, fn, hallucinated = scores['true_positives'], scores['false_positives'], scores['false_negatives'], scores['hallucination']
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    hallucination_rate = hallucinated / (tp + fp + hallucinated) if (tp + fp + hallucinated) else 0
    return precision, recall, f1, hallucination_rate


def evaluation_exp2(scores: Dict[str, int], total_docs: int = 390) -> Tuple[float, float]:
    non_rate = scores['non'] / total_docs
    hallucination_rate = scores['hallucination'] / total_docs
    return non_rate, hallucination_rate


def exp1(predicted: pd.DataFrame, ground_truth: List[Dict[str, Any]]) -> None:
    scores = {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}

    for item in ground_truth:
        predicted_ = predicted.loc[predicted['paper_id'] == int(item['doc_id'])]
        ground_parameters = extract_values(item['answers'])
        predicted_parameters = predicted_['outputs'].dropna().tolist()
        predicted_parameters = split_and_extract(predicted_parameters)

        tp, fp, fn = compute_scores_exp1(predicted_parameters, ground_parameters)
        scores['true_positives'] += tp
        scores['false_positives'] += fp
        scores['false_negatives'] += fn

    precision, recall, f1 = evaluation_exp1(scores)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


def exp2(predicted: pd.DataFrame, _: Any) -> None:
    with open('docs.json') as f:
        _ = json.load(f)

    scores = {'non': 0, 'hallucination': 0}

    for _, row in predicted.iterrows():
        predicted_parameters = row['outputs']
        predicted_list = split_and_extract([predicted_parameters]) if pd.notna(predicted_parameters) and 'NONE' not in str(predicted_parameters) else ['NONE']
        hl, non = compute_scores_exp2(predicted_list)
        scores['hallucination'] += hl
        scores['non'] += non

    non_rate, halluc_rate = evaluation_exp2(scores)
    print(f"Non Rate:           {1 - non_rate:.4f} ")
    print(f"Hallucination Rate: {halluc_rate:.4f} ")


def exp3(predicted: pd.DataFrame, ground_truth: List[Dict[str, Any]]) -> None:
    with open('docs.json') as f:
        ground_truth_docs = json.load(f)

    scores = {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0, 'hallucination': 0}

    for item in ground_truth:
        predicted_ = predicted.loc[predicted['paper_id'] == int(item['doc_id'])]
        ground_parameters = extract_values(item['answers'])
        predicted_parameters = predicted_['outputs'].dropna().tolist()
        predicted_parameters = split_and_extract(predicted_parameters)

        abstract_text = ground_truth_docs[str(item['doc_id'])]['abstract']
        tp, fp, fn, hl = compute_scores_exp3(predicted_parameters, ground_parameters, abstract_text)

        scores['true_positives'] += tp
        scores['false_positives'] += fp
        scores['false_negatives'] += fn
        scores['hallucination'] += hl

    precision, recall, f1, halluc_rate = evaluation_exp3(scores)
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1 Score:           {f1:.4f}")
    print(f"Hallucination Rate: {halluc_rate:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted extraction results.")
    parser.add_argument(
        "-e", "--experiment", type=str, choices=['exp1', 'exp2', 'exp3'], required=True,
        help="Experiment to run: exp1 (relevance), exp2 (none/hallucination), or exp3 (full with hallucination rate)"
    )

    parser.add_argument(
        "-p", "--predicted-file", type=str, required=True,
        help="Path to the CSV file containing predicted values"
    )

    parser.add_argument(
        "-g", "--ground-file", type=str, required=True,
        help="Path to the JSON file containing ground truth annotations"
    )
    args = parser.parse_args()

    predicted = load_pred_file(args.predicted_file)
    ground = load_ground_file(args.ground_file)

    if args.experiment == 'exp1':
        exp1(predicted, ground)
    elif args.experiment == 'exp2':
        exp2(predicted, ground)
    elif args.experiment == 'exp3':
        exp3(predicted, ground)


if __name__ == "__main__":
    main()

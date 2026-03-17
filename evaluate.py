"""
Evaluation script: Exact Match and F1 metrics for QA.
Based on SQuAD evaluation methodology.
"""

import re
import string
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """Check if normalized prediction matches any ground truth answer."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    """Compute token-level F1 between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)

    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = sum(min(prediction_tokens.count(w), ground_truth_tokens.count(w)) for w in common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate(predictions_file, references_file):
    """Evaluate predictions against references."""
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]

    with open(references_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]

    assert len(predictions) == len(references), \
        f"Number of predictions ({len(predictions)}) != references ({len(references)})"

    total_em = 0
    total_f1 = 0

    for pred, ref in zip(predictions, references):
        # Handle multiple reference answers (separated by |)
        ref_answers = [r.strip() for r in ref.split('|')]

        # Take the best score across all valid reference answers
        best_em = max(exact_match_score(pred, r) for r in ref_answers)
        best_f1 = max(f1_score(pred, r) for r in ref_answers)

        total_em += best_em
        total_f1 += best_f1

    n = len(predictions)
    print(f"Exact Match: {total_em / n * 100:.2f}%")
    print(f"F1 Score:    {total_f1 / n * 100:.2f}%")
    print(f"Total questions: {n}")

    return total_em / n, total_f1 / n


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate.py <predictions_file> <references_file>")
        sys.exit(1)

    evaluate(sys.argv[1], sys.argv[2])

"""
Evaluation script V2: Weighted evaluation to encourage unified rule discovery.
Pieri cases are weighted more heavily to guide evolution toward correct patterns.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from shinka.core import run_shinka_eval

# Type aliases
Partition = List[int]
Cell = Tuple[int, int]
CellSet = Set[Cell]


def load_test_data(data_path: str, num_samples: int = None) -> List[Dict]:
    """Load test data with (lambda, mu, nu) triples and ground truth A, B, C."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if num_samples is not None and num_samples < len(data):
        random.seed(42)
        data = random.sample(data, num_samples)
    
    return data


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Compute Jaccard similarity: |A ∩ B| / |A ∪ B|"""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def f1_score(pred_set: Set, true_set: Set) -> float:
    """Compute F1 score for set prediction."""
    if len(pred_set) == 0 and len(true_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(true_set) == 0:
        return 0.0
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_single_case(
    prediction: Dict[str, CellSet], 
    ground_truth: Dict[str, CellSet],
    is_pieri: bool = False
) -> Dict[str, float]:
    """
    Evaluate a single test case with optional Pieri weighting.
    
    Args:
        prediction: Predicted A, B, C sets
        ground_truth: True A, B, C sets
        is_pieri: Whether this is a Pieri case (single-row mu)
        
    Returns:
        Dictionary with metrics and weight
    """
    # Convert to sets if needed
    A_pred = set(map(tuple, prediction['A'])) if isinstance(prediction['A'], list) else prediction['A']
    B_pred = set(map(tuple, prediction['B'])) if isinstance(prediction['B'], list) else prediction['B']
    C_pred = set(map(tuple, prediction['C'])) if isinstance(prediction['C'], list) else prediction['C']
    
    A_true = set(map(tuple, ground_truth['A'])) if isinstance(ground_truth['A'], list) else ground_truth['A']
    B_true = set(map(tuple, ground_truth['B'])) if isinstance(ground_truth['B'], list) else ground_truth['B']
    C_true = set(map(tuple, ground_truth['C'])) if isinstance(ground_truth['C'], list) else ground_truth['C']
    
    # Exact match
    exact_match = (A_pred == A_true) and (B_pred == B_true) and (C_pred == C_true)
    
    # Individual set metrics
    jaccard_A = jaccard_similarity(A_pred, A_true)
    jaccard_B = jaccard_similarity(B_pred, B_true)
    jaccard_C = jaccard_similarity(C_pred, C_true)
    
    f1_A = f1_score(A_pred, A_true)
    f1_B = f1_score(B_pred, B_true)
    f1_C = f1_score(C_pred, C_true)
    
    # Average scores
    avg_jaccard = (jaccard_A + jaccard_B + jaccard_C) / 3.0
    avg_f1 = (f1_A + f1_B + f1_C) / 3.0
    
    # Case weight: Pieri cases are more important
    # They provide ground truth that should be learned
    case_weight = 2.5 if is_pieri else 1.0
    
    return {
        'exact_match': 1.0 if exact_match else 0.0,
        'jaccard_A': jaccard_A,
        'jaccard_B': jaccard_B,
        'jaccard_C': jaccard_C,
        'avg_jaccard': avg_jaccard,
        'f1_A': f1_A,
        'f1_B': f1_B,
        'f1_C': f1_C,
        'avg_f1': avg_f1,
        'case_weight': case_weight,
        'is_pieri': is_pieri,
    }


def get_experiment_kwargs(run_idx: int, test_data: List[Dict]) -> dict:
    """Get kwargs for a specific test case."""
    if run_idx >= len(test_data):
        run_idx = run_idx % len(test_data)
    
    case = test_data[run_idx]
    return {
        'lambda_part': case['lambda'],
        'mu_part': case['mu'],
        'nu_part': case['nu'],
    }


def aggregate_metrics(results: List[Tuple[Dict, Dict]]) -> dict:
    """
    Aggregate metrics with weighted averaging for Pieri cases.
    
    This encourages the evolved rule to master Pieri cases first,
    then generalize to other cases.
    """
    all_metrics = [metrics for _, metrics in results]
    
    # Separate Pieri and non-Pieri cases
    pieri_metrics = [m for m in all_metrics if m['is_pieri']]
    general_metrics = [m for m in all_metrics if not m['is_pieri']]
    
    # Weighted aggregation
    def weighted_avg(metrics_list, key):
        if not metrics_list:
            return 0.0
        total_weight = sum(m['case_weight'] for m in metrics_list)
        weighted_sum = sum(m[key] * m['case_weight'] for m in metrics_list)
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Overall weighted metrics
    exact_match_rate = weighted_avg(all_metrics, 'exact_match')
    avg_f1 = weighted_avg(all_metrics, 'avg_f1')
    avg_jaccard = weighted_avg(all_metrics, 'avg_jaccard')
    
    # Separate metrics for analysis
    pieri_exact = weighted_avg(pieri_metrics, 'exact_match') if pieri_metrics else 0.0
    general_exact = weighted_avg(general_metrics, 'exact_match') if general_metrics else 0.0
    
    pieri_f1 = weighted_avg(pieri_metrics, 'avg_f1') if pieri_metrics else 0.0
    general_f1 = weighted_avg(general_metrics, 'avg_f1') if general_metrics else 0.0
    
    # Combined fitness score with Pieri emphasis
    # Pieri correctness is critical - it's the known ground truth
    combined_score = (
        0.5 * exact_match_rate +     # Overall exact matches
        0.3 * avg_f1 +                 # Partial credit
        0.2 * avg_jaccard +            # Secondary metric
        0.1 * pieri_exact              # Bonus for Pieri mastery
    )
    # Note: This can exceed 1.0, which is fine for optimization
    
    # Public metrics (visible to LLM)
    public_metrics = {
        'overall_exact_match': f"{exact_match_rate:.4f}",
        'pieri_exact_match': f"{pieri_exact:.4f}",
        'general_exact_match': f"{general_exact:.4f}",
        'overall_f1': f"{avg_f1:.4f}",
        'pieri_f1': f"{pieri_f1:.4f}",
        'general_f1': f"{general_f1:.4f}",
        'pieri_vs_general_gap': f"{abs(pieri_exact - general_exact):.4f}",
    }
    
    # Private metrics
    private_metrics = {
        'num_pieri': len(pieri_metrics),
        'num_general': len(general_metrics),
        'pieri_ratio': len(pieri_metrics) / len(all_metrics) if all_metrics else 0,
        'jaccard_breakdown': {
            'overall': avg_jaccard,
            'pieri': weighted_avg(pieri_metrics, 'avg_jaccard') if pieri_metrics else 0,
            'general': weighted_avg(general_metrics, 'avg_jaccard') if general_metrics else 0,
        }
    }
    
    # Text feedback for LLM
    pieri_general_gap = abs(pieri_exact - general_exact)
    
    feedback_lines = [
        "Performance Summary:",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Overall Exact Match:  {exact_match_rate*100:.1f}%",
        f"  → Pieri cases:      {pieri_exact*100:.1f}% ({len(pieri_metrics)} cases)",
        f"  → General cases:    {general_exact*100:.1f}% ({len(general_metrics)} cases)",
        f"",
        f"F1 Scores:",
        f"  → Overall:          {avg_f1:.4f}",
        f"  → Pieri:            {pieri_f1:.4f}",
        f"  → General:          {general_f1:.4f}",
        f"",
    ]
    
    # Guidance based on performance
    if pieri_exact < 0.95:
        feedback_lines.append(
            "⚠️  PRIORITY: Master Pieri cases first (single-row mu)."
        )
        feedback_lines.append(
            "   These are known ground truth - learn this pattern!"
        )
    elif pieri_general_gap > 0.3:
        feedback_lines.append(
            "⚠️  Good Pieri performance, but generalization gap is large."
        )
        feedback_lines.append(
            "   Try to extend the Pieri pattern to multi-row mu."
        )
    elif general_exact > 0.7:
        feedback_lines.append(
            "✓  Strong performance! Fine-tune for remaining edge cases."
        )
    else:
        feedback_lines.append(
            "→  Focus on finding common patterns between Pieri and general cases."
        )
    
    text_feedback = "\n".join(feedback_lines)
    
    return {
        'combined_score': combined_score,
        'public': public_metrics,
        'private': private_metrics,
        'text_feedback': text_feedback,
        'extra_data': {
            'all_metrics': all_metrics,
            'pieri_metrics': pieri_metrics,
            'general_metrics': general_metrics,
        }
    }


def validate_fn(program_path: str) -> Tuple[bool, str]:
    """Validate program structure before evaluation."""
    try:
        with open(program_path, 'r') as f:
            code = f.read()
        
        compile(code, program_path, 'exec')
        
        if 'def compute_ABC_sets' not in code:
            return False, "Missing required function: compute_ABC_sets"
        
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def main(program_path: str, results_dir: str):
    """Main evaluation function."""
    data_path = Path(__file__).parent / "test_data_qt_LR.json"
    test_data = load_test_data(str(data_path), num_samples=5000)
    
    def get_kwargs_closure(run_idx: int) -> dict:
        return get_experiment_kwargs(run_idx, test_data)
    
    def custom_eval_fn(run_idx: int, result: Dict) -> Tuple[Dict, Dict]:
        """Evaluate one run with Pieri awareness."""
        case = test_data[run_idx]
        ground_truth = {
            'A': case['A'],
            'B': case['B'],
            'C': case['C'],
        }
        
        is_pieri = case.get('metadata', {}).get('is_pieri', False)
        metrics = evaluate_single_case(result, ground_truth, is_pieri)
        return (result, metrics)
    
    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=len(test_data),
        get_experiment_kwargs=get_kwargs_closure,
        aggregate_metrics_fn=aggregate_metrics,
        validate_fn=validate_fn,
        custom_eval_fn=custom_eval_fn,
    )
    
    return metrics, correct, err


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("program_path", type=str)
    parser.add_argument("results_dir", type=str)
    
    args = parser.parse_args()
    main(args.program_path, args.results_dir)

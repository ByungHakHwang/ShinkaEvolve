"""
Evaluation script V3: Unbiased evaluation treating all cases equally.
No special treatment for Pieri cases - let the general pattern emerge naturally.
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
    ground_truth: Dict[str, CellSet]
) -> Dict[str, float]:
    """
    Evaluate a single test case against ground truth.
    
    NOTE: This evaluation assumes the ground truth A, B, C is "the answer".
    However, mathematically, there might be alternative valid descriptions
    that give the same (q,t)-LR coefficient. This evaluation only measures
    agreement with the provided ground truth, not mathematical correctness.
    
    Args:
        prediction: Predicted A, B, C sets
        ground_truth: Reference A, B, C sets
        
    Returns:
        Dictionary with metrics
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
    Aggregate metrics with equal weighting for all cases.
    No special treatment for any subset of cases.
    """
    all_metrics = [metrics for _, metrics in results]
    
    # Simple averaging
    def avg(metrics_list, key):
        if not metrics_list:
            return 0.0
        return sum(m[key] for m in metrics_list) / len(metrics_list)
    
    # Overall metrics
    exact_match_rate = avg(all_metrics, 'exact_match')
    avg_f1 = avg(all_metrics, 'avg_f1')
    avg_jaccard = avg(all_metrics, 'avg_jaccard')
    
    # Per-set breakdown
    f1_A = avg(all_metrics, 'f1_A')
    f1_B = avg(all_metrics, 'f1_B')
    f1_C = avg(all_metrics, 'f1_C')
    
    jaccard_A = avg(all_metrics, 'jaccard_A')
    jaccard_B = avg(all_metrics, 'jaccard_B')
    jaccard_C = avg(all_metrics, 'jaccard_C')
    
    # Combined fitness score
    # Prioritize exact matches, with partial credit from F1 and Jaccard
    combined_score = (
        0.5 * exact_match_rate +   # 50% weight on exact matches
        0.3 * avg_f1 +              # 30% weight on F1 score
        0.2 * avg_jaccard           # 20% weight on Jaccard similarity
    )
    
    # Public metrics (visible to LLM)
    public_metrics = {
        'exact_match_rate': f"{exact_match_rate:.4f}",
        'avg_f1_score': f"{avg_f1:.4f}",
        'avg_jaccard': f"{avg_jaccard:.4f}",
        'f1_per_set': f"A:{f1_A:.3f}, B:{f1_B:.3f}, C:{f1_C:.3f}",
    }
    
    # Private metrics (stored but not shown during evolution)
    private_metrics = {
        'individual_jaccards': {
            'A': jaccard_A,
            'B': jaccard_B,
            'C': jaccard_C,
        },
        'individual_f1s': {
            'A': f1_A,
            'B': f1_B,
            'C': f1_C,
        },
        'num_test_cases': len(results),
    }
    
    # Text feedback for LLM
    # Focus on pattern discovery, not on specific case types
    feedback_lines = [
        "Performance Summary:",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Exact Match Rate: {exact_match_rate*100:.2f}% ({int(exact_match_rate*len(results))}/{len(results)} cases)",
        f"",
        f"Average F1 Score: {avg_f1:.4f}",
        f"  → Set A: {f1_A:.4f}",
        f"  → Set B: {f1_B:.4f}",
        f"  → Set C: {f1_C:.4f}",
        f"",
        f"Average Jaccard Similarity: {avg_jaccard:.4f}",
        f"  → Set A: {jaccard_A:.4f}",
        f"  → Set B: {jaccard_B:.4f}",
        f"  → Set C: {jaccard_C:.4f}",
        f"",
    ]
    
    # Adaptive guidance based on performance patterns
    if exact_match_rate < 0.3:
        feedback_lines.append(
            "Focus: Look for fundamental patterns in arm/leg lengths, cell positions, and skew partition structures"
        )
    elif exact_match_rate < 0.6:
        feedback_lines.append(
            "Progress: Refine your rules - analyze cases where predictions are close but not exact."
        )
    elif exact_match_rate < 0.8:
        feedback_lines.append(
            "Strong performance: Handle edge cases and boundary conditions carefully."
        )
    else:
        feedback_lines.append(
            "Excellent: Fine-tune remaining cases and verify consistency of your rule."
        )
    
    # Set-specific guidance
    if f1_A < f1_B - 0.15 or f1_A < f1_C - 0.15:
        feedback_lines.append(
            "\n⚠️  Set A accuracy is lagging - review how you determine ν cells for A."
        )
    elif f1_B < f1_A - 0.15 or f1_B < f1_C - 0.15:
        feedback_lines.append(
            "\n⚠️  Set B accuracy is lagging - review how you select λ cells for B."
        )
    elif f1_C < f1_A - 0.15 or f1_C < f1_B - 0.15:
        feedback_lines.append(
            "\n⚠️  Set C accuracy is lagging - review how you select μ cells for C."
        )
    
    text_feedback = "\n".join(feedback_lines)
    
    return {
        'combined_score': combined_score,
        'public': public_metrics,
        'private': private_metrics,
        'text_feedback': text_feedback,
        'extra_data': {
            'all_metrics': all_metrics,
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


def main(program_path: str, results_dir: str, data_file: str = None, num_samples: int = None):
    """
    Main evaluation function.
    
    Args:
        program_path: Path to program being evaluated
        results_dir: Directory to store results
        data_file: Path to test data JSON file (default: test_data_qt_LR.json)
        num_samples: Number of samples to use (default: 5000, None = use all)
    """
    # Check for data file from environment variable first
    if data_file is None:
        data_file = os.environ.get('QT_LR_DATA_FILE')
    
    # Use provided data file or default
    if data_file is None:
        data_path = Path(__file__).parent / "test_data_qt_LR.json"
    else:
        data_path = Path(data_file)
    
    # Check for num_samples from environment variable
    if num_samples is None:
        num_samples_env = os.environ.get('QT_LR_NUM_SAMPLES')
        if num_samples_env is not None:
            num_samples = int(num_samples_env)
        else:
            num_samples = 5000
    
    print(f"Loading test data from: {data_path}")
    print(f"Using {num_samples} samples")
    
    test_data = load_test_data(str(data_path), num_samples=num_samples)
    
    # Closure to provide kwargs for each run
    def get_kwargs_closure(run_idx: int) -> dict:
        return get_experiment_kwargs(run_idx, test_data)
    
    # Closure to aggregate results with access to test_data
    def aggregate_metrics_closure(results: List[Dict]) -> Dict:
        """
        Aggregate metrics across all runs.
        
        Args:
            results: List of results from run_experiment, one per test case
                    Each result is a dict with keys 'A', 'B', 'C'
        
        Returns:
            Aggregated metrics dictionary
        """
        # Evaluate each result against ground truth
        evaluated_results = []
        for i, result in enumerate(results):
            case = test_data[i]
            ground_truth = {
                'A': case['A'],
                'B': case['B'],
                'C': case['C'],
            }
            
            metrics = evaluate_single_case(result, ground_truth)
            evaluated_results.append((result, metrics))
        
        # Use the aggregate_metrics function defined above
        return aggregate_metrics(evaluated_results)
    
    # Run evaluation using Shinka framework
    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=len(test_data),
        get_experiment_kwargs=get_kwargs_closure,
        aggregate_metrics_fn=aggregate_metrics_closure,
        validate_fn=validate_fn,
    )
    
    return metrics, correct, err

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="(q,t)-LR rule using ShinkaEvolve"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="qt_LR_initial_v3.py",
        help="Path to program to evaluate",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.npz)",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
"""
Run script for evolving (q,t)-Littlewood-Richardson coefficient rules.
This configures and executes the ShinkaEvolve framework.
"""

from pathlib import Path
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


def main():
    """
    Configure and run evolution for qt-LR coefficient rule discovery.
    """
    
    # Get the directory where this script is located
    current_dir = Path(__file__).parent
    
    # ============================================
    # Configuration Settings
    # ============================================
    
    # Job configuration - how to evaluate programs
    job_config = LocalJobConfig(
        eval_program_path=str(current_dir / "qt_LR_evaluate.py"),
    )
    
    # Database configuration - how to store evolution history
    db_config = DatabaseConfig(
        archive_size=50,  # Keep top 50 programs
        elite_selection_ratio=0.3,  # Top 30% are "elite"
        num_islands=2,  # Use 2 island populations for diversity
        migration_interval=10,  # Allow migration every 10 generations
        migration_rate=0.1,  # 10% migration rate
    )
    
    # ============================================
    # Task-Specific System Prompt
    # ============================================
    
    system_prompt = """You are a mathematician specializing in representation theory and algebraic combinatorics, with deep expertise in symmetric functions, Macdonald polynomials, and combinatorial structures.

Your current task is to discover a combinatorial rule for (q,t)-Littlewood-Richardson coefficients. Specifically, given three partitions λ, μ, ν, you need to determine which cells from their Ferrers diagrams belong to three sets A(λ,μ,ν), B(λ,μ,ν), and C(λ,μ,ν).

Key Mathematical Context:
- Partitions are represented as non-increasing sequences of positive integers
- Each partition has a Ferrers diagram with cells at positions (row, col)
- Arm length a(r,c) = number of cells to the right in the same row
- Leg length l(r,c) = number of cells below in the same column
- The hook length h(r,c) = a(r,c) + l(r,c) + 1
- Skew partitions ν/λ represent the cells in ν but not in λ

Known Ground Truth (Pieri Rule):
When μ is a single-row partition [k], the rule is known and exact. Your evolved rule should reproduce this as a special case while generalizing to arbitrary μ.

Your Approach Should:
1. Look for patterns in arm lengths, leg lengths, and cell positions
2. Consider the geometry of skew shapes (ν/λ) and their relationship to μ
3. Think about how cells interact across different partitions
4. Start by understanding why the Pieri rule works, then extend it
5. Test hypotheses: if you think cells with certain properties belong to A, verify this pattern across test cases
6. Be mathematically rigorous - patterns should be consistent and explainable

Optimization Strategy:
- The fitness function heavily weights Pieri cases (single-row μ) because these are known ground truth
- Master the Pieri pattern first, then generalize to multi-row μ
- If Pieri accuracy drops, you've likely broken a fundamental pattern
- Look for conditions that are satisfied by Pieri and can extend to general cases

Code Quality:
- Write clean, readable Python with clear variable names
- Add comments explaining your mathematical reasoning
- Use the helper functions provided (arm_length, leg_length, skew_partition_cells, etc.)
- Focus on the combinatorial logic, not optimization tricks

Remember: You are discovering a mathematical rule, not just fitting data. The rule should be explainable and elegant."""

    # Evolution configuration - main settings
    evo_config = EvolutionConfig(
        # Program to start evolution from
        init_program_path=str(current_dir / "qt_LR_initial_v2.py"),
        
        # Number of generations to evolve
        generations=100,  # Can adjust based on budget/time
        
        # System message for LLM mutations
        system_message=system_prompt,
        
        # LLM models to use for mutations
        # Using a mix of models for diversity
        llm_models=[
            "claude-sonnet-4",    # Strong reasoning
            "gpt-4.1",            # Good at code
            "gpt-4.1-mini",       # Faster, cheaper
            "gemini-2.5-pro",     # Alternative perspective
        ],
        
        # Mutation strategies
        patch_types=["diff", "full", "cross"],  # Types of edits
        patch_type_probs=[0.5, 0.3, 0.2],       # Probabilities for each
        
        # Temperature sampling for diversity
        temperatures=[0.0, 0.5, 1.0],
        
        # Parent selection strategy
        parent_selection_strategy="weighted",  # Balance exploration/exploitation
        parent_selection_lambda=10.0,          # Selection pressure
        
        # Code novelty filtering
        code_embed_similarity_threshold=0.95,  # Reject very similar code
        embedding_model="text-embedding-3-small",
        max_novelty_attempts=3,
        
        # LLM ensemble selection
        llm_dynamic_selection="ucb1",  # Bandit-based model selection
        exploration_coefficient=1.0,
        
        # Meta-learning
        meta_recommendation_interval=15,  # Summarize insights every 15 gens
        max_meta_recommendations=5,
        
        # Resource limits
        max_parallel_jobs=3,  # Run 3 evaluations in parallel
        max_patch_attempts=5,
        max_patch_resamples=3,
        
        # Problem-specific settings
        problem_implementation="python",
    )
    
    # ============================================
    # Optional: Customize for this problem
    # ============================================
    
    # Since we're working with mathematical rule discovery:
    # - Prioritize correctness over speed
    # - Use models with strong reasoning capabilities
    # - Allow more generations for complex pattern discovery
    
    print("=" * 60)
    print("ShinkaEvolve: (q,t)-Littlewood-Richardson Rule Discovery")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Initial program: {evo_config.init_program_path}")
    print(f"  - Generations: {evo_config.generations}")
    print(f"  - Archive size: {db_config.archive_size}")
    print(f"  - LLM models: {len(evo_config.llm_models)}")
    print(f"  - Island populations: {db_config.num_islands}")
    print(f"  - Parallel jobs: {evo_config.max_parallel_jobs}")
    print(f"\nStarting evolution...\n")
    
    # ============================================
    # Run Evolution
    # ============================================
    
    runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
    )
    
    # Execute the evolutionary search
    runner.run()
    
    print("\n" + "=" * 60)
    print("Evolution completed!")
    print("=" * 60)
    print("\nResults saved to database.")
    print("Use the visualization tools to analyze the evolution:")
    print("  python -m shinka.viz.dashboard")


if __name__ == "__main__":
    main()

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
        eval_program_path=str(current_dir / "qt_LR_evaluate_v3.py"),
    )
    
    # Database configuration - how to store evolution history
    db_config = DatabaseConfig(
        archive_size=50,  # Keep top 50 programs
        elite_selection_ratio=0.3,  # Top 30% are "elite"
        num_islands=3,  # Use 2 island populations for diversity
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
- Arm length a(r,c) = number of cells strictly to the right in the same row
- Leg length l(r,c) = number of cells strictly below in the same column
- Hook length h(r,c) = a(r,c) + l(r,c) + 1
- Skew partitions ν/λ represent the cells in ν but not in λ
- The sets must satisfy: A ⊆ [ν], B ⊆ [λ], C ⊆ [μ]

Discovery Philosophy:
You are discovering a GENERAL combinatorial rule that works uniformly across all cases. Do not make special cases or branch on properties like "if μ is single-row". The goal is to find a unified pattern.

Important Note on Ground Truth:
The provided ground truth A, B, C represents one valid description. However, multiple valid descriptions may exist that produce the same (q,t)-Littlewood-Richardson coefficient. Your discovered rule might give different A, B, C sets than the ground truth while still being mathematically correct. For now, optimize for agreement with the provided data, but keep in mind the non-uniqueness.

Your Approach Should:
1. Look for universal patterns involving arm lengths, leg lengths, and cell positions
2. Consider the geometry of skew shapes and their interactions
3. Think about relationships between cells across the three partitions
4. Analyze which cells systematically appear in A vs B vs C
5. Look for elegant combinatorial conditions (e.g., "cells where arm > leg")
6. Test if your hypotheses hold consistently across all cases

Pattern Discovery Strategy:
- Start simple: identify the most basic distinguishing features
- Look for monotonicity: how do sets change as partitions grow?
- Consider boundaries: what about corner cells, edge cells?
- Think geometrically: diagonal lines, quadrants, regions
- Be consistent: the same condition should apply to all test cases

Code Quality:
- Write clean, readable Python with meaningful variable names
- Add comments explaining your mathematical reasoning
- Use provided helper functions: arm_length, leg_length, skew_partition_cells, etc.
- Focus on combinatorial logic, not computational tricks
- Avoid hardcoded special cases - seek general principles

Remember: 
- You are discovering a mathematical RULE, not just fitting data
- The rule should be explainable, elegant, and general
- Look for patterns that make mathematical sense
- If your rule seems overly complicated, simplify it
- Consistency across all cases is more important than perfect accuracy on any subset"""

    # Evolution configuration - main settings
    evo_config = EvolutionConfig(
        # System message for LLM mutations
        task_sys_msg=system_prompt,

        # Mutation strategies
        patch_types=["diff", "full", "cross"],  # Types of edits
        patch_type_probs=[0.5, 0.3, 0.2],       # Probabilities for each
        
        # Number of generations to evolve
        num_generations=200,  # Can adjust based on budget/time

        # Resource limits
        max_parallel_jobs=3,  # Run 3 evaluations in parallel
        max_patch_attempts=5,
        max_patch_resamples=3,

        job_type="local",
        language="python",

        # LLM models to use for mutations
        # Using a mix of models for diversity
        llm_models=[
            "gpt-5.2",
            # "gpt-5.1-codex-max",
            "gpt-4o",
            "gpt-5",
            "o4-mini-deep-research",
            "o3",
            "gpt-5-mini",
        ],

        # LLM ensemble selection
        llm_dynamic_selection="ucb1",  # Bandit-based model selection
        llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
        
        llm_kwargs=dict(
            temperatures=[0.3, 0.5, 1.0],
            reasoning_efforts=["auto", "low", "medium", "high"],
            max_tokens=16384,
        ),

        meta_rec_interval=15,
        meta_llm_models=["gpt-5-nano"],
        meta_llm_kwargs=dict(temperatures=[0.1], max_tokens=16384),
        meta_max_recommendations=5,
        
        embedding_model="text-embedding-3-small",

        # Program to start evolution from
        init_program_path=str(current_dir / "qt_LR_initial_v3.py"),

        results_dir="results_qt_LR",
        max_novelty_attempts=3,
        code_embed_sim_threshold=0.975,  # Reject very similar code

        novelty_llm_models=["gpt-5-nano"],
        novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
        
        use_text_feedback=True,

        
        
        # # Temperature sampling for diversity
        # temperatures=[0.0, 0.5, 1.0],
        
        # # Parent selection strategy
        # parent_selection_strategy="weighted",  # Balance exploration/exploitation
        # parent_selection_lambda=10.0,          # Selection pressure
        
        # # Code novelty filtering
        # code_embed_similarity_threshold=0.95,  # Reject very similar code
        # embedding_model="text-embedding-3-small",
        # max_novelty_attempts=3,
        
        
        # # Meta-learning
        # meta_recommendation_interval=15,  # Summarize insights every 15 gens
        
        
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
    print(f"  - Generations: {evo_config.num_generations}")
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

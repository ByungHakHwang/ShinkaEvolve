"""
Initial program for (q,t)-Littlewood-Richardson coefficient rule discovery.
Version 2: Using helper functions instead of classes for better LLM flexibility.
"""

from typing import List, Tuple, Set

# Type aliases
Partition = List[int]
Cell = Tuple[int, int]
CellSet = Set[Cell]


# ============================================
# HELPER FUNCTIONS (Outside EVOLVE-BLOCK)
# These are available to LLM but not forced
# ============================================

def partition_to_cells(partition: Partition) -> CellSet:
    """Convert partition to set of cells (row, col) with 0-indexing."""
    cells = set()
    for r, row_length in enumerate(partition):
        for c in range(row_length):
            cells.add((r, c))
    return cells


def get_partition_size(partition: Partition) -> int:
    """Return total number of cells in partition."""
    return sum(partition)


def get_partition_length(partition: Partition) -> int:
    """Return number of rows in partition."""
    return len(partition)


def partition_conjugate(partition: Partition) -> Partition:
    """
    Compute conjugate partition (transpose Ferrers diagram).
    Example: [4, 3, 1] -> [3, 2, 2, 1]
    """
    if not partition:
        return []
    conj = []
    max_val = partition[0]
    for c in range(max_val):
        col_height = sum(1 for row_len in partition if row_len > c)
        conj.append(col_height)
    return conj


def arm_length(partition: Partition, cell: Cell) -> int:
    """
    Arm length: number of cells strictly to the right in same row.
    a(r,c) = lambda[r] - c - 1
    """
    r, c = cell
    if r >= len(partition) or c >= partition[r]:
        return 0
    return partition[r] - c - 1


def leg_length(partition: Partition, cell: Cell) -> int:
    """
    Leg length: number of cells strictly below in same column.
    l(r,c) = number of rows below r where row_length > c
    """
    r, c = cell
    if r >= len(partition):
        return 0
    return sum(1 for i in range(r + 1, len(partition)) if partition[i] > c)


def arm_leg_length(partition: Partition, cell: Cell) -> Tuple[int, int]:
    """Return (arm, leg) lengths for a cell."""
    return arm_length(partition, cell), leg_length(partition, cell)


def is_partition_including(outer: Partition, inner: Partition) -> bool:
    """
    Check if outer partition contains inner partition.
    True if outer[i] >= inner[i] for all i.
    """
    if len(outer) < len(inner):
        return False
    for i in range(len(inner)):
        if outer[i] < inner[i]:
            return False
    return True


def skew_partition_cells(outer: Partition, inner: Partition) -> CellSet:
    """
    Return cells in skew shape (outer / inner).
    Cells in outer but not in inner.
    """
    if not is_partition_including(outer, inner):
        return set()
    
    skew_cells = set()
    for r in range(len(outer)):
        start_col = inner[r] if r < len(inner) else 0
        for c in range(start_col, outer[r]):
            skew_cells.add((r, c))
    return skew_cells


def skew_partition_row_indices(outer: Partition, inner: Partition) -> List[int]:
    """
    Return row indices that are non-empty in skew diagram.
    Row r is included if outer[r] > inner[r].
    """
    indices = []
    for r in range(len(outer)):
        inner_r = inner[r] if r < len(inner) else 0
        if outer[r] > inner_r:
            indices.append(r)
    return indices


def skew_partition_column_indices(outer: Partition, inner: Partition) -> List[int]:
    """
    Return column indices that are non-empty in skew diagram.
    Computed via conjugate skew partition's row indices.
    """
    outer_conj = partition_conjugate(outer)
    inner_conj = partition_conjugate(inner)
    return skew_partition_row_indices(outer_conj, inner_conj)


# ============================================
# PIERI / DUAL PIERI RULE - Known Solution
# Provided as reference/validation
# ============================================

def compute_pieri_ABC_sets(
    lambda_part: Partition,
    mu_part: Partition,
    nu_part: Partition
) -> Tuple[CellSet, CellSet, CellSet]:
    """
    Known Pieri rule for the case when mu is a single-row partition.
    
    This is the mathematically proven correct solution for this special case.
    The general rule being evolved should reproduce this as a special case.
    
    Args:
        lambda_part: First partition
        mu_part: Single-row partition [k]
        nu_part: Result partition
        
    Returns:
        (A, B, C) sets according to Pieri rule
    """

    A_set = set()
    B_set = set()
    C_set = set()
    
    nu_cells = partition_to_cells(nu_part)
    lambda_cells = partition_to_cells(lambda_part)
    mu_cells = partition_to_cells(mu_part)
    
    nu_skew_lambda_cells = nu_cells - lambda_cells
    nu_skew_mu_cells = nu_cells - mu_cells
    nu_skew_lambda_row_indices = set(r for (r, _) in nu_skew_lambda_cells)
    nu_skew_lambda_column_indices = set(c for (_, c) in nu_skew_lambda_cells)
    nu_skew_mu_row_indices = set(r for (r, _) in nu_skew_mu_cells)
    nu_skew_mu_column_indices = set(c for (_, c) in nu_skew_mu_cells)

    for (r,c) in nu_cells:
        if c in nu_skew_lambda_column_indices:
            A_set.add((r,c))
    
    for (r,c) in lambda_cells:
        if c in nu_skew_lambda_column_indices:
            B_set.add((r,c))
    
    for (r,c) in mu_cells:
        C_set.add((r,c))
    
    return A_set, B_set, C_set

def compute_dual_pieri_ABC_sets(
    lambda_part: Partition,
    mu_part: Partition,
    nu_part: Partition
) -> Tuple[CellSet, CellSet, CellSet]:
    """
    Known Pieri rule for the case when mu is a single-column partition.
    
    This is the mathematically proven correct solution for this special case.
    The general rule being evolved should reproduce this as a special case, at least should give a similar production.
    
    Args:
        lambda_part: First partition
        mu_part: Single-column partition [1,1,...,1]
        nu_part: Result partition
        
    Returns:
        (A, B, C) sets according to dual Pieri rule
    """

    A_set = set()
    B_set = set()
    C_set = set()
    
    nu_cells = partition_to_cells(nu_part)
    lambda_cells = partition_to_cells(lambda_part)
    mu_cells = partition_to_cells(mu_part)
    
    nu_skew_lambda_cells = nu_cells - lambda_cells
    nu_skew_mu_cells = nu_cells - mu_cells
    nu_skew_lambda_row_indices = set(r for (r, _) in nu_skew_lambda_cells)
    nu_skew_lambda_column_indices = set(c for (_, c) in nu_skew_lambda_cells)
    nu_skew_mu_row_indices = set(r for (r, _) in nu_skew_mu_cells)
    nu_skew_mu_column_indices = set(c for (_, c) in nu_skew_mu_cells)

    for (r,c) in nu_cells:
        if (c in nu_skew_lambda_column_indices) and (r not in nu_skew_lambda_row_indices):
            A_set.add((r,c))

    for (r,c) in lambda_cells:
        if (c in nu_skew_lambda_column_indices) and (r not in nu_skew_lambda_row_indices):
            B_set.add((r,c))
    
    return A_set, B_set, C_set


# ============================================
# MAIN FUNCTION TO EVOLVE
# ============================================

# EVOLVE-BLOCK-START
def compute_ABC_sets(
    lambda_part: Partition, 
    mu_part: Partition, 
    nu_part: Partition
) -> Tuple[CellSet, CellSet, CellSet]:
    """
    Compute the sets A(λ,μ,ν), B(λ,μ,ν), C(λ,μ,ν) for given partitions.
    
    This is the function that ShinkaEvolve will optimize to discover
    the general rule that includes Pieri as a special case.
    
    DESIGN PHILOSOPHY:
    - Start with a general approach that works for all cases
    - Can use helper functions defined above
    - Should naturally handle Pieri case (single-row mu) and dual Pieri case (single-column mu) correctly
    - Fitness will heavily weight Pieri / dual Pieri case accuracy to guide evolution
    
    Args:
        lambda_part: First partition λ
        mu_part: Second partition μ  
        nu_part: Result partition ν
        
    Returns:
        Tuple of (A, B, C) where each is a set of cells (row, col)
        - A is a set of cells (row, col) in nu_part
        - B is a set of cells (row, col) in lambda_part
        - C is a set of cells (row, col) in mu_part
    """

    # Initialize sets and define basic sets
    A_set = set()
    B_set = set()
    C_set = set()
    
    nu_cells = partition_to_cells(nu_part)
    lambda_cells = partition_to_cells(lambda_part)
    mu_cells = partition_to_cells(mu_part)
    
    nu_skew_lambda_cells = nu_cells - lambda_cells
    nu_skew_mu_cells = nu_cells - mu_cells
    nu_skew_lambda_row_indices = set(r for (r, _) in nu_skew_lambda_cells)
    nu_skew_lambda_column_indices = set(c for (_, c) in nu_skew_lambda_cells)
    nu_skew_mu_row_indices = set(r for (r, _) in nu_skew_mu_cells)
    nu_skew_mu_column_indices = set(c for (_, c) in nu_skew_mu_cells)
    
    # ==========================================
    # INITIAL NAIVE HEURISTIC
    # This will be evolved by ShinkaEvolve
    # ==========================================
    
    # Heuristic 1: Cells in nu/lambda go to A
    for cell in nu_cells:
        r, c = cell
        if r not in nu_skew_lambda_row_indices:
            A_set.add(cell)
        elif c in nu_skew_lambda_column_indices:
            A_set.add(cell)

    # Heuristic 2: Some lambda cells go to B
    for cell in lambda_cells:
        r, c = cell
        arm = arm_length(lambda_part, cell)
        leg = leg_length(lambda_part, cell)
        
        if arm == arm_length(nu_part, cell) and leg < leg_length(nu_part, cell):
            B_set.add(cell)
    
    # Heuristic 3: Some mu cells go to C
    for cell in mu_cells:
        r, c = cell
        arm = arm_length(mu_part, cell)
        leg = leg_length(mu_part, cell)
        
        if arm + leg < len(lambda_part):
            C_set.add(cell)
    
    # NOTE FOR LLM:
    # The above heuristics are intentionally simple and likely incorrect.
    # During evolution, you should:
    # 1. Analyze which cells actually belong in A, B, C
    # 2. Look for patterns in arm lengths, leg lengths, positions
    # 3. Consider the structure of skew partitions
    # 4. Focus on row and column indices of skew partitions
    # 5. Pay special attention to Pieri / dual Pieri cases (single-row mu / single-column mu)
    # 6. Do not forget that A, B, C is sets of cells in nu, lambda, mu, respectively.
    # 7. Try to find a unified rule that handles all cases
    
    return A_set, B_set, C_set
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """
    Main entry point called by the evaluation framework.
    """
    lambda_part = kwargs.get('lambda_part')
    mu_part = kwargs.get('mu_part')
    nu_part = kwargs.get('nu_part')
    
    # Use the general function (which LLM will evolve)
    A_pred, B_pred, C_pred = compute_ABC_sets(lambda_part, mu_part, nu_part)
    
    return {
        'A': A_pred,
        'B': B_pred,
        'C': C_pred
    }

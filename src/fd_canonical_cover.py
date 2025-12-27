from typing import List, Tuple, Iterator
from itertools import combinations
import pandas as pd


# Check whether X → A is a functional dependency
def is_fd(df: pd.DataFrame, X: List[str], A: str) -> bool:
    groups = df.groupby(X)[A].nunique()
    return (groups <= 1).all()

# Generate candidate LHS attribute sets up to size max_size
def level_wise_candidates(columns: List[str], max_size: int):
    # Generate combinations of size 1, 2, ..., max_size
    for size in range(1, max_size + 1):
        for combo in combinations(columns, size):
            yield list(combo)

# Discover all functional dependencies X → A (not minimal)
def discover_fds(df: pd.DataFrame, max_size: int = 3):
    columns = list(df.columns)
    # For each candidate LHS X
    for X in level_wise_candidates(columns, max_size):
        # For each possible RHS A
        for A in columns:
            if A not in X and is_fd(df, X, A):
                yield (X, A)


# Load dataset
df = pd.read_csv("C:/Users/uhunm/Downloads/Airline Dataset Updated - v2.csv")

from itertools import combinations
from typing import List

# Check whether X → A is minimal
def is_minimal_fd(df: pd.DataFrame, X: List[str], A: str) -> bool:
    # Try all proper subsets of X
    for size in range(1, len(X)):
        for subset in combinations(X, size):
            groups = df.groupby(list(subset))[A].nunique()
            # If a subset already determines A, X is not minimal
            if (groups <= 1).all():
                return False
    return True

# Discover minimal FDs using a base FD discovery function
def discover_minimal_fds(df, discover_fds_fn, max_size=3):
    for X, A in discover_fds_fn(df, max_size):
        if is_minimal_fd(df, X, A):
            yield (X, A)

# Compute attribute closure of X under a set of FDs
def attribute_closure(X, fds):
    closure = set(X)
    changed = True

    # Keep adding attributes until no more can be added
    while changed:
        changed = False
        for lhs, rhs in fds:
            # If lhs ⊆ closure, then rhs must be added
            if set(lhs).issubset(closure) and rhs not in closure:
                closure.add(rhs)
                changed = True

    return closure


# Check if an FD is redundant in a set of FDs
def is_redundant_fd(fd, fds):
    X, A = fd
    # Remove the FD and compute closure
    reduced = [f for f in fds if f != fd]
    # If A is still implied, the FD is redundant
    return A in attribute_closure(X, reduced)

# Compute canonical cover (minimal cover) of FDs
def canonical_cover(fds: List[Tuple[List[str], str]]):
    # Convert to sorted tuples for consistency
    current_fds = {(tuple(sorted(X)), A) for X, A in fds}

    changed = True
    while changed:
        changed = False
        minimized = set()

        # Step 1: Minimize LHS of each FD
        for X, A in current_fds:
            X = list(X)
            for attr in X[:]:
                trial = [x for x in X if x != attr]
                # If removing attr still implies A, remove it
                if trial and A in attribute_closure(trial, list(current_fds)):
                    X = trial
                    changed = True
            minimized.add((tuple(sorted(X)), A))

        current_fds = minimized
        non_redundant = set()

        # Step 2: Remove redundant FDs
        for fd in current_fds:
            if not is_redundant_fd(fd, list(current_fds)):
                non_redundant.add(fd)
            else:
                changed = True

        current_fds = non_redundant

    # Convert back to list format
    return [(list(X), A) for X, A in sorted(current_fds)]

# Run FD discovery + canonical cover
fds = list(discover_minimal_fds(df, discover_fds))
canonical_fds = canonical_cover(fds)

# Print canonical FDs
for X, A in canonical_fds:
    print(f"{X} -> {A}")

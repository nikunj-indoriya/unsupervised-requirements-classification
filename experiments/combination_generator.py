import itertools


def generate_all_combinations(class_indices):
    """
    Generate all combinations from k=2 to k=n
    """
    all_combinations = []

    n = len(class_indices)

    for k in range(2, n + 1):
        combos = list(itertools.combinations(class_indices, k))
        all_combinations.extend(combos)

    return all_combinations


def generate_functional_vs_nonfunctional(class_names, label_encoder):
    """
    Special case: Functional vs Non-Functional
    """
    functional_idx = label_encoder["F"]

    non_functional = [
        label_encoder[c] for c in class_names if c != "F"
    ]

    return functional_idx, non_functional

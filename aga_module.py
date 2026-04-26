"""
aga_module.py
=============
Adaptive Genetic Algorithm for per-user criterion weight optimisation.

Chromosome  : W = {w1, w2, w3, w4, w5}  (real-valued, sum = 1)
Fitness     : MAE between predicted utility scores and actual ratings (lower = better)
Selection   : Tournament selection (T=3)
Crossover   : Arithmetic crossover (Pc = 0.80)
Mutation    : Gaussian mutation — N(0, sigma²)
Adaptive    : Diversity Div(t) < threshold → raise Pm | else → lower Pm
Elitism     : Best chromosome always carried to next generation
Termination : max_generations OR no improvement > delta for patience generations
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

AGA_CONFIG = {
    "population_size":      50,
    "max_generations":      100,
    "crossover_rate":       0.80,
    "mutation_base":        0.10,
    "mutation_low":         0.05,     # when diversity is HIGH  → exploit
    "mutation_high":        0.20,     # when diversity is LOW   → explore
    "mutation_sigma":       0.05,     # std dev of Gaussian perturbation
    "tournament_size":      3,
    "diversity_check_freq": 10,       # check diversity every N generations
    "diversity_threshold":  0.10,     # below this → raise mutation rate
    "early_stop_patience":  20,       # no improvement for N gen → stop
    "early_stop_delta":     0.001,    # minimum meaningful MAE improvement
    "chromosome_length":    5,        # one weight per criterion
    "seed":                 42,
}

CRITERIA  = ["storyline", "acting", "visuals", "emotional_impact", "enjoyment"]
NORM_COLS = [f"{c}_norm" for c in CRITERIA]


# ─────────────────────────────────────────────────────────────────────────────
# 1. POPULATION INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def initialise_population(pop_size: int,
                           n_genes: int,
                           rng: np.random.Generator) -> np.ndarray:
    """
    Create initial population of pop_size chromosomes.
    Each chromosome is a random weight vector normalised to sum = 1.

    Returns ndarray of shape (pop_size, n_genes).
    """
    pop = rng.uniform(0, 1, size=(pop_size, n_genes))
    # Normalise each row to sum to 1
    row_sums = pop.sum(axis=1, keepdims=True)
    pop = pop / row_sums
    return pop


def normalise_chromosome(w: np.ndarray) -> np.ndarray:
    """
    Ensure all weights are non-negative and sum to 1.
    If all weights collapse to zero, return equal weights.
    """
    w = np.maximum(w, 0)
    total = w.sum()
    if total == 0:
        return np.ones(len(w)) / len(w)
    return w / total


# ─────────────────────────────────────────────────────────────────────────────
# 2. FITNESS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_utility_score(ratings_row: pd.Series,
                           weights: np.ndarray) -> float:
    """
    Compute U(u, i) = Σc [ wc · r'(u, i, c) ] for one rating row.
    """
    score = 0.0
    for i, col in enumerate(NORM_COLS):
        if col in ratings_row.index:
            score += weights[i] * float(ratings_row[col])
    return float(np.clip(score, 0.0, 1.0))


def compute_fitness(weights: np.ndarray,
                    user_train: pd.DataFrame) -> float:
    """
    Compute MAE fitness for one chromosome against a user's training ratings.

    f(W, u) = (1 / |T_u|) × Σ_{(u,i) ∈ T_u} |U(u,i,W) − r_actual(u,i)|

    r_actual is overall_norm (normalised overall rating).

    Lower MAE = better fitness.
    """
    if len(user_train) == 0:
        return 1.0  # worst possible fitness

    errors = []
    for _, row in user_train.iterrows():
        predicted = compute_utility_score(row, weights)
        actual    = float(row["overall_norm"])
        errors.append(abs(predicted - actual))

    return float(np.mean(errors))


def evaluate_population(population: np.ndarray,
                         user_train: pd.DataFrame) -> np.ndarray:
    """
    Evaluate fitness for every chromosome in the population.

    Returns ndarray of shape (pop_size,) — fitness values.
    """
    return np.array([
        compute_fitness(population[i], user_train)
        for i in range(len(population))
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOURNAMENT SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def tournament_selection(population: np.ndarray,
                          fitness: np.ndarray,
                          tournament_size: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Select one parent via tournament selection.
    Draw tournament_size random chromosomes, return the one with lowest MAE.
    """
    indices  = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]
    return population[best_idx].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. ARITHMETIC CROSSOVER
# ─────────────────────────────────────────────────────────────────────────────

def arithmetic_crossover(parent1: np.ndarray,
                          parent2: np.ndarray,
                          crossover_rate: float,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Arithmetic crossover with probability crossover_rate.

    C1 = α · P1 + (1 - α) · P2
    C2 = (1 - α) · P1 + α · P2

    Both children are renormalised to sum = 1.
    If crossover is not applied, children are copies of parents.
    """
    if rng.random() < crossover_rate:
        alpha   = rng.random()
        child1  = alpha * parent1 + (1 - alpha) * parent2
        child2  = (1 - alpha) * parent1 + alpha * parent2
        child1  = normalise_chromosome(child1)
        child2  = normalise_chromosome(child2)
    else:
        child1 = parent1.copy()
        child2 = parent2.copy()

    return child1, child2


# ─────────────────────────────────────────────────────────────────────────────
# 5. GAUSSIAN MUTATION
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_mutation(chromosome: np.ndarray,
                       mutation_rate: float,
                       sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Apply Gaussian mutation to each gene with probability mutation_rate.

    For each selected gene: w_i = w_i + N(0, sigma²)
    Chromosome is renormalised after mutation.
    """
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            mutated[i] += rng.normal(0, sigma)
    return normalise_chromosome(mutated)


# ─────────────────────────────────────────────────────────────────────────────
# 6. POPULATION DIVERSITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_diversity(population: np.ndarray) -> float:
    """
    Compute average pairwise Euclidean distance across all chromosome pairs.

    Div(t) = (2 / (P · (P-1))) × Σ_{i<j} ||W_i - W_j||_2
    """
    n = len(population)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist   = np.linalg.norm(population[i] - population[j])
            total += dist
            count += 1

    return float(total / count) if count > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. ADAPTIVE MUTATION RATE
# ─────────────────────────────────────────────────────────────────────────────

def adapt_mutation_rate(diversity: float,
                         config: dict) -> float:
    """
    Adjust mutation rate based on current population diversity.

    diversity < threshold  →  raise to mutation_high  (inject variation)
    diversity >= threshold →  lower to mutation_low   (allow exploitation)
    """
    if diversity < config["diversity_threshold"]:
        return config["mutation_high"]
    else:
        return config["mutation_low"]


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN AGA LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_aga(user_id: int,
             user_train: pd.DataFrame,
             config: dict = None) -> dict:
    """
    Run the Adaptive Genetic Algorithm for one user.

    Returns dict:
        {
          "user_id":       int,
          "best_weights":  ndarray (5,),
          "best_mae":      float,
          "generations":   int,
          "converged":     bool,
          "history":       list of best MAE per generation
        }
    """
    if config is None:
        config = AGA_CONFIG

    rng = np.random.default_rng(config["seed"])

    # Check if user has enough training data
    if len(user_train) < 3:
        print(f"[aga_module] User {user_id}: insufficient training data "
              f"({len(user_train)} ratings) — returning equal weights")
        return {
            "user_id":      user_id,
            "best_weights": np.ones(config["chromosome_length"]) / config["chromosome_length"],
            "best_mae":     1.0,
            "generations":  0,
            "converged":    False,
            "history":      []
        }

    pop_size   = config["population_size"]
    n_genes    = config["chromosome_length"]
    max_gen    = config["max_generations"]
    pc         = config["crossover_rate"]
    pm         = config["mutation_base"]
    sigma      = config["mutation_sigma"]
    t_size     = config["tournament_size"]
    patience   = config["early_stop_patience"]
    delta      = config["early_stop_delta"]
    div_freq   = config["diversity_check_freq"]

    # ── Initialise population ───────────────────────────────────────────────
    population = initialise_population(pop_size, n_genes, rng)
    fitness    = evaluate_population(population, user_train)

    best_idx     = np.argmin(fitness)
    best_weights = population[best_idx].copy()
    best_mae     = float(fitness[best_idx])
    history      = [best_mae]
    no_improve   = 0
    converged    = False

    # ── Evolutionary loop ───────────────────────────────────────────────────
    for gen in range(1, max_gen + 1):

        # Adaptive mutation rate check every div_freq generations
        if gen % div_freq == 0:
            div = compute_diversity(population)
            pm  = adapt_mutation_rate(div, config)

        # Build next generation
        next_pop = []

        # Elitism: carry best chromosome forward
        next_pop.append(best_weights.copy())

        # Fill remaining slots with offspring
        while len(next_pop) < pop_size:
            p1 = tournament_selection(population, fitness, t_size, rng)
            p2 = tournament_selection(population, fitness, t_size, rng)

            c1, c2 = arithmetic_crossover(p1, p2, pc, rng)
            c1     = gaussian_mutation(c1, pm, sigma, rng)
            c2     = gaussian_mutation(c2, pm, sigma, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size:
                next_pop.append(c2)

        population = np.array(next_pop)
        fitness    = evaluate_population(population, user_train)

        # Track best
        gen_best_idx = np.argmin(fitness)
        gen_best_mae = float(fitness[gen_best_idx])

        if best_mae - gen_best_mae > delta:
            best_mae     = gen_best_mae
            best_weights = population[gen_best_idx].copy()
            no_improve   = 0
        else:
            no_improve += 1

        history.append(best_mae)

        # Early stopping check
        if no_improve >= patience:
            converged = True
            print(f"[aga_module] User {user_id}: early stop at gen {gen} "
                  f"(no improvement for {patience} gen) | MAE={best_mae:.4f}")
            break

    if not converged:
        print(f"[aga_module] User {user_id}: max generations reached | "
              f"MAE={best_mae:.4f}")

    return {
        "user_id":      user_id,
        "best_weights": best_weights,
        "best_mae":     round(best_mae, 6),
        "generations":  len(history) - 1,
        "converged":    converged,
        "history":      history
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. FORMAT RESULT FOR STORAGE / DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def format_weights(result: dict) -> dict:
    """
    Format AGA result into a clean dict suitable for database storage
    or API response.
    """
    weights = result["best_weights"]
    return {
        "user_id":        result["user_id"],
        "w1_storyline":   round(float(weights[0]), 4),
        "w2_acting":      round(float(weights[1]), 4),
        "w3_visuals":     round(float(weights[2]), 4),
        "w4_emotional":   round(float(weights[3]), 4),
        "w5_enjoyment":   round(float(weights[4]), 4),
        "best_mae":       result["best_mae"],
        "generations":    result["generations"],
        "converged":      result["converged"],
    }

def format_weights(result: dict) -> dict:
    """
    Format AGA result into a clean dict suitable for database storage
    or API response.
    """
    weights = result["best_weights"]
    return {
        "user_id":        result["user_id"],
        "w1_storyline":   round(float(weights[0]), 4),
        "w2_acting":      round(float(weights[1]), 4),
        "w3_visuals":     round(float(weights[2]), 4),
        "w4_emotional":   round(float(weights[3]), 4),
        "w5_enjoyment":   round(float(weights[4]), 4),
        "best_mae":       result["best_mae"],
        "generations":    result["generations"],
        "converged":      result["converged"],
    }
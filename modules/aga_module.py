import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# SECTION 3.5.3 — Adaptive Genetic Algorithm
# Optimises the 5 criterion weights (w1–w5)
# for a specific user based on their rating history
# ─────────────────────────────────────────────

CRITERIA = [
    "storyline_norm",
    "acting_norm",
    "visuals_norm",
    "emotional_impact_norm",
    "enjoyment_norm"
]

# ─────────────────────────────────────────────
# AGA Parameters (Chapter 3.5.3 Table 3.4)
# ─────────────────────────────────────────────

AGA_CONFIG = {
    "population_size"   : 50,      # number of chromosomes
    "max_generations"   : 100,     # termination condition 1
    "crossover_rate"    : 0.8,     # 80% of parents undergo crossover
    "mutation_rate_low" : 0.05,    # when population diversity is high
    "mutation_rate_high": 0.20,    # when population is converging
    "mutation_rate_base": 0.10,    # starting mutation rate
    "mutation_std"      : 0.05,    # Gaussian noise std for mutation
    "tournament_size"   : 3,       # tournament selection pool size
    "diversity_check"   : 10,      # check diversity every N generations
    "diversity_threshold": 0.10,   # below this → raise mutation rate
    "early_stop_patience": 20,     # stop if no improvement for N gens
    "early_stop_delta"  : 0.001,   # minimum improvement to count
    "chromosome_length" : 5        # one weight per criterion
}


# ─────────────────────────────────────────────
# Chromosome Encoding
# Each chromosome = [w1, w2, w3, w4, w5]
# All weights are positive and sum to 1
# ─────────────────────────────────────────────

def normalise_chromosome(chromosome: np.ndarray) -> np.ndarray:
    """
    Ensure all weights are positive and sum to 1.
    Called after every operation that creates or modifies a chromosome.
    """
    # Force all weights to be positive
    chromosome = np.abs(chromosome)

    # Avoid division by zero
    total = chromosome.sum()
    if total == 0:
        return np.ones(5) / 5.0  # fallback to equal weights

    return chromosome / total


def initialise_population(
    pop_size: int,
    n_genes: int,
    seed: int = None
) -> np.ndarray:
    """
    Create the initial population of chromosomes.
    Each chromosome is a random weight vector normalised to sum to 1.
    Shape: (pop_size, n_genes)
    """
    rng = np.random.default_rng(seed)
    population = rng.uniform(0, 1, size=(pop_size, n_genes))

    # Normalise each chromosome so weights sum to 1
    population = np.array([
        normalise_chromosome(chrom) for chrom in population
    ])

    return population


# ─────────────────────────────────────────────
# Fitness Function
# fitness(W, u) = MAE across all training ratings
# Lower MAE = better fitness (Chapter 3.5.3)
# ─────────────────────────────────────────────

def compute_fitness(
    chromosome: np.ndarray,
    user_ratings: pd.DataFrame
) -> float:
    """
    Evaluate a chromosome by computing MAE between:
    - predicted rating = weighted sum of 5 normalised criteria scores
    - actual overall rating (normalised to [0,1])
    
    fitness(W, u) = (1/|Ru|) * SUM_i |R(u,i) - R_actual(u,i)|
    
    Lower is better.
    """
    if len(user_ratings) == 0:
        return 1.0  # worst possible fitness if no ratings

    # Predicted rating = w1*c1 + w2*c2 + w3*c3 + w4*c4 + w5*c5
    criteria_matrix = user_ratings[CRITERIA].values   # shape (n, 5)
    predicted = criteria_matrix @ chromosome           # shape (n,)

    # Normalise actual rating to [0, 1] for fair comparison
    actual = user_ratings["rating"].values
    actual_norm = (actual - 1) / 4.0                  # scale [1,5] → [0,1]

    mae = np.mean(np.abs(predicted - actual_norm))
    return float(mae)


def evaluate_population(
    population: np.ndarray,
    user_ratings: pd.DataFrame
) -> np.ndarray:
    """
    Compute fitness for every chromosome in the population.
    Returns array of MAE values, shape (pop_size,).
    """
    return np.array([
        compute_fitness(chrom, user_ratings)
        for chrom in population
    ])


# ─────────────────────────────────────────────
# Selection — Tournament Selection
# Pick 3 random chromosomes, return the best one
# (Chapter 3.5.3)
# ─────────────────────────────────────────────

def tournament_selection(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    tournament_size: int = 3,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Select one parent chromosome via tournament selection.
    Draw tournament_size random chromosomes, return the one
    with the lowest MAE (best fitness).
    """
    if rng is None:
        rng = np.random.default_rng()

    indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = indices[np.argmin(fitness_scores[indices])]
    return population[best_idx].copy()


# ─────────────────────────────────────────────
# Crossover — Arithmetic Crossover
# Child = alpha * Parent1 + (1 - alpha) * Parent2
# (Chapter 3.5.3)
# ─────────────────────────────────────────────

def arithmetic_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator
) -> tuple:
    """
    Combine two parent chromosomes using arithmetic crossover.
    Returns two child chromosomes.
    alpha is drawn randomly from [0, 1].
    """
    alpha = rng.uniform(0, 1)
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2

    return (
        normalise_chromosome(child1),
        normalise_chromosome(child2)
    )


# ─────────────────────────────────────────────
# Mutation — Gaussian Mutation
# Add small noise from Normal(0, std) to each weight
# (Chapter 3.5.3)
# ─────────────────────────────────────────────

def gaussian_mutation(
    chromosome: np.ndarray,
    mutation_rate: float,
    mutation_std: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Apply Gaussian mutation to a chromosome.
    Each weight has mutation_rate probability of being mutated.
    Mutation adds noise from Normal(mean=0, std=mutation_std).
    """
    mutated = chromosome.copy()

    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            noise = rng.normal(loc=0.0, scale=mutation_std)
            mutated[i] += noise

    # Re-normalise after mutation (weights must still sum to 1)
    return normalise_chromosome(mutated)


# ─────────────────────────────────────────────
# Adaptive Mutation Rate
# Reduce rate when diversity is high (exploring well)
# Raise rate when diversity drops (converging too fast)
# Inspired by Srinivas & Patnaik (1994) — Chapter 3.5.3
# ─────────────────────────────────────────────

def compute_diversity(population: np.ndarray) -> float:
    """
    Measure population diversity as the average pairwise
    Euclidean distance between chromosomes.
    High diversity = population is exploring widely.
    Low diversity = population is converging.
    """
    n = len(population)
    if n < 2:
        return 0.0

    total_distance = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            total_distance += np.linalg.norm(
                population[i] - population[j]
            )
            count += 1

    return total_distance / count if count > 0 else 0.0


def adapt_mutation_rate(
    population: np.ndarray,
    config: dict
) -> float:
    """
    Dynamically adjust mutation rate based on population diversity.
    If diversity > threshold → use low mutation rate (0.05)
    If diversity <= threshold → use high mutation rate (0.20)
    Called every diversity_check generations.
    """
    diversity = compute_diversity(population)

    if diversity > config["diversity_threshold"]:
        return config["mutation_rate_low"]
    else:
        return config["mutation_rate_high"]


# ─────────────────────────────────────────────
# Main AGA Loop
# Full evolutionary process for one user
# ─────────────────────────────────────────────

def run_aga(
    user_id: int,
    user_ratings: pd.DataFrame,
    config: dict = None,
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """
    Run the full Adaptive Genetic Algorithm for one user.
    
    Args:
        user_id     : the user we are optimising weights for
        user_ratings: their training ratings DataFrame
        config      : AGA hyperparameters (uses AGA_CONFIG if None)
        seed        : random seed for reproducibility
        verbose     : print progress every 10 generations
    
    Returns a dict containing:
        best_weights   : np.ndarray of shape (5,) — the optimised weights
        best_fitness   : float — the MAE achieved by the best weights
        generations_run: int — how many generations the AGA ran
        fitness_history: list — best MAE at each generation
        diversity_history: list — diversity at each generation
    """
    if config is None:
        config = AGA_CONFIG

    if len(user_ratings) < 5:
        # Not enough ratings to run AGA — return equal weights
        if verbose:
            print(f"[AGA] User {user_id} has too few ratings "
                  f"({len(user_ratings)}). Using equal weights.")
        return {
            "best_weights"     : np.ones(5) / 5.0,
            "best_fitness"     : None,
            "generations_run"  : 0,
            "fitness_history"  : [],
            "diversity_history": []
        }

    rng = np.random.default_rng(seed)
    pop_size    = config["population_size"]
    n_genes     = config["chromosome_length"]
    max_gen     = config["max_generations"]
    c_rate      = config["crossover_rate"]
    mut_std     = config["mutation_std"]
    t_size      = config["tournament_size"]
    div_check   = config["diversity_check"]
    patience    = config["early_stop_patience"]
    delta       = config["early_stop_delta"]

    if verbose:
        print(f"\n[AGA] Starting for user {user_id} | "
              f"{len(user_ratings)} training ratings | "
              f"pop={pop_size} | max_gen={max_gen}")

    # ── Initialise population ──
    population = initialise_population(pop_size, n_genes, seed)
    mutation_rate = config["mutation_rate_base"]

    # ── Evaluate initial population ──
    fitness_scores = evaluate_population(population, user_ratings)

    best_idx     = np.argmin(fitness_scores)
    best_weights = population[best_idx].copy()
    best_fitness = fitness_scores[best_idx]

    fitness_history   = [best_fitness]
    diversity_history = [compute_diversity(population)]
    no_improve_count  = 0

    # ── Generational loop ──
    for generation in range(1, max_gen + 1):

        new_population = []

        # Elitism — keep the best chromosome unchanged
        new_population.append(best_weights.copy())

        # Fill rest of population with offspring
        while len(new_population) < pop_size:

            # Selection
            parent1 = tournament_selection(
                population, fitness_scores, t_size, rng
            )
            parent2 = tournament_selection(
                population, fitness_scores, t_size, rng
            )

            # Crossover
            if rng.random() < c_rate:
                child1, child2 = arithmetic_crossover(parent1, parent2, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            child1 = gaussian_mutation(child1, mutation_rate, mut_std, rng)
            child2 = gaussian_mutation(child2, mutation_rate, mut_std, rng)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population     = np.array(new_population)
        fitness_scores = evaluate_population(population, user_ratings)

        # Track best
        gen_best_idx = np.argmin(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]

        if gen_best_fitness < best_fitness - delta:
            best_fitness = gen_best_fitness
            best_weights = population[gen_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        fitness_history.append(best_fitness)

        # ── Adaptive mutation rate (every div_check generations) ──
        if generation % div_check == 0:
            mutation_rate = adapt_mutation_rate(population, config)
            diversity = compute_diversity(population)
            diversity_history.append(diversity)

            if verbose:
                print(f"[AGA] Gen {generation:3d} | "
                      f"Best MAE: {best_fitness:.4f} | "
                      f"Diversity: {diversity:.4f} | "
                      f"Mut rate: {mutation_rate:.2f}")

        # ── Early stopping ──
        if no_improve_count >= patience:
            if verbose:
                print(f"[AGA] Early stop at generation {generation} "
                      f"(no improvement for {patience} generations)")
            break

    if verbose:
        print(f"\n[AGA] Complete for user {user_id}")
        print(f"[AGA] Best MAE    : {best_fitness:.4f}")
        print(f"[AGA] Best weights: {np.round(best_weights, 4)}")
        labels = ["Storyline", "Acting", "Visuals",
                  "Emot.Impact", "Enjoyment"]
        for label, w in zip(labels, best_weights):
            print(f"       {label:<12}: {w:.4f}")

    return {
        "best_weights"     : best_weights,
        "best_fitness"     : best_fitness,
        "generations_run"  : generation,
        "fitness_history"  : fitness_history,
        "diversity_history": diversity_history
    }
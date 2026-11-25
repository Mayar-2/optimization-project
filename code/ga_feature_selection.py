"""
Genetic Algorithm for Feature Selection  (Project Tasks 6 and 7)
Dataset : Gas Sensor Array Drift (batch1 ... batch10, 128 features, 6 gas classes)

Idea:
    Each individual (chromosome) is a binary vector of length 128:
        1 = feature is selected, 0 = feature is not selected
    Fitness = macro-F1 on the VALIDATION set  -  LAMBDA * (selected / total)

IMPORTANT (no data leakage):
    - The GA uses ONLY the old period (OLD_BATCHES), split into train / validation / hist-test.
    - Later batches (NEW_BATCHES) are never used during feature selection.
    - The hist-test part is not used by the GA either; it is only reported once at the end.
"""
import glob
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------------------
# Paths and data settings
# ---------------------------------------
# Put this file in the same folder as batch1.dat ... batch10.dat.
# All outputs are saved in the same folder.
OLD_BATCHES = [1, 2, 3, 4, 5]                          # historical period (used by the GA)
NEW_BATCHES = [6, 7, 8, 9, 10]                         # future period (NOT used by the GA)
SEED = 42

# ---------------------------------------
# GA parameters
# ---------------------------------------
N_FEATURES = 128
POP_SIZE = 30            # number of individuals
N_GENERATIONS = 30       # number of generations
CROSSOVER_RATE = 0.8
MUTATION_RATE = 1.0 / N_FEATURES * 2   # per-bit flip probability (about 2 bits per child)
TOURNAMENT_SIZE = 3
ELITISM = 2              # best individuals copied directly to next generation
LAMBDA = 0.05            # penalty weight for the number of selected features
INIT_ONES_PROB = 0.5     # probability of a gene being 1 in the initial population

rng = np.random.default_rng(SEED)


# ---------------------------------------
# Load data
# ---------------------------------------
def load_batch(b):
    path = glob.glob(f"data/*batch{b}.dat")[0]
    X, y = load_svmlight_file(path, n_features=N_FEATURES)
    return X.toarray(), y.astype(int)


def load_period(batches):
    Xs, ys, bs = [], [], []
    for b in batches:
        X, y = load_batch(b)
        Xs.append(X); ys.append(y); bs.append(np.full(len(y), b))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(bs)


# ---------------------------------------
# Fitness function: MAXIMIZE
# ---------------------------------------
def make_classifier():
    return RandomForestClassifier(n_estimators=40, max_depth=12, n_jobs=-1, random_state=SEED)


_cache = {}   # same chromosome is not evaluated twice


def evaluate(chrom, X_tr, y_tr, X_val, y_val):
    key = chrom.tobytes()
    if key in _cache:
        return _cache[key]
    if chrom.sum() == 0:                 # empty subset is invalid
        return -1.0, 0.0
    cols = np.where(chrom == 1)[0]
    clf = make_classifier()
    clf.fit(X_tr[:, cols], y_tr)
    f1 = f1_score(y_val, clf.predict(X_val[:, cols]), average="macro")
    fit = f1 - LAMBDA * (chrom.sum() / N_FEATURES)
    _cache[key] = (fit, f1)
    return fit, f1


# ---------------------------------------
# Initialize population (each individual is a binary vector)
# ---------------------------------------
def init_population(pop_size):
    pop = (rng.random((pop_size, N_FEATURES)) < INIT_ONES_PROB).astype(int)
    for ind in pop:                      # make sure no individual is empty
        if ind.sum() == 0:
            ind[rng.integers(N_FEATURES)] = 1
    return pop


# ---------------------------------------
# Tournament selection (for MAXIMIZATION)
# ---------------------------------------
def tournament_selection(pop, fitness, k=TOURNAMENT_SIZE):
    idx = rng.integers(0, len(pop), size=k)
    best_idx = idx[np.argmax(fitness[idx])]
    return pop[best_idx].copy()


# ---------------------------------------
# Crossover: single-point (swap the tails of two parents)
# ---------------------------------------
def crossover(p1, p2):
    if rng.random() < CROSSOVER_RATE:
        point = rng.integers(1, N_FEATURES)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2
    return p1.copy(), p2.copy()


# ---------------------------------------
# Mutation: flip each bit with small probability
# ---------------------------------------
def mutate(chrom):
    flips = rng.random(N_FEATURES) < MUTATION_RATE
    chrom = np.where(flips, 1 - chrom, chrom)
    if chrom.sum() == 0:
        chrom[rng.integers(N_FEATURES)] = 1
    return chrom


# ---------------------------------------
# Main GA loop (MAXIMIZATION)
# ---------------------------------------
def genetic_algorithm(X_tr, y_tr, X_val, y_val):
    population = init_population(POP_SIZE)
    history = []

    for gen in range(N_GENERATIONS):
        scores = [evaluate(ind, X_tr, y_tr, X_val, y_val) for ind in population]
        fitness = np.array([s[0] for s in scores])
        f1s = np.array([s[1] for s in scores])

        best_idx = np.argmax(fitness)
        history.append({
            "generation": gen,
            "best_fitness": fitness[best_idx],
            "best_val_f1": f1s[best_idx],
            "best_n_features": int(population[best_idx].sum()),
            "mean_fitness": fitness.mean(),
        })
        print(f"Generation {gen:3d}: best fitness = {fitness[best_idx]:.4f}, "
              f"val macro-F1 = {f1s[best_idx]:.4f}, "
              f"features = {int(population[best_idx].sum()):3d}, "
              f"mean fitness = {fitness.mean():.4f}")

        # create new population
        order = np.argsort(-fitness)
        new_population = [population[i].copy() for i in order[:ELITISM]]   # elitism

        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            if len(new_population) < POP_SIZE:
                new_population.append(mutate(c2))

        population = np.array(new_population)

    scores = [evaluate(ind, X_tr, y_tr, X_val, y_val) for ind in population]
    fitness = np.array([s[0] for s in scores])
    best_idx = np.argmax(fitness)
    return population[best_idx], fitness[best_idx], pd.DataFrame(history)


# ---------------------------------------
# Run everything
# ---------------------------------------
def main():
    t0 = time.time()

    # 1) old period only -> train / validation / historical test (stratified)
    X_old, y_old, b_old = load_period(OLD_BATCHES)
    idx = np.arange(len(y_old))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.4, stratify=y_old, random_state=SEED)
    idx_val, idx_ht = train_test_split(idx_tmp, test_size=0.5, stratify=y_old[idx_tmp], random_state=SEED)
    X_tr, y_tr = X_old[idx_tr], y_old[idx_tr]
    X_val, y_val = X_old[idx_val], y_old[idx_val]
    X_ht, y_ht = X_old[idx_ht], y_old[idx_ht]
    print(f"Old period batches {OLD_BATCHES}: train={len(idx_tr)}, val={len(idx_val)}, hist-test={len(idx_ht)}")

    # 2) run GA (uses train + validation only)
    best_chrom, best_fit, hist = genetic_algorithm(X_tr, y_tr, X_val, y_val)
    cols = np.where(best_chrom == 1)[0]
    print(f"\nBest chromosome selects {len(cols)} / {N_FEATURES} features")
    print("Selected feature indices (1-based):", (cols + 1).tolist())

    # 3) report once: all features vs selected features
    def report(cols_used):
        clf = make_classifier().fit(X_tr[:, cols_used], y_tr)
        out = {}
        for name, (X, y) in {"validation": (X_val, y_val), "historical_test": (X_ht, y_ht)}.items():
            p = clf.predict(X[:, cols_used])
            out[name] = {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro")}
        return out

    res_all = report(np.arange(N_FEATURES))
    res_sel = report(cols)
    print("\nAll 128 features :", res_all)
    print("GA selected      :", res_sel)

    # 4) save results (Task 6)
    hist.to_csv("results/ga_history.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["generation"], hist["best_fitness"], label="best fitness")
    ax[0].plot(hist["generation"], hist["mean_fitness"], label="mean fitness")
    ax[0].set_xlabel("Generation"); ax[0].set_ylabel("Fitness"); ax[0].legend(); ax[0].set_title("GA convergence")
    ax[1].plot(hist["generation"], hist["best_n_features"], color="tab:green")
    ax[1].set_xlabel("Generation"); ax[1].set_ylabel("# selected features"); ax[1].set_title("Selected features of the best individual")
    plt.tight_layout(); plt.savefig("results/ga_convergence.png", dpi=150)

    # 5) Task 7: new dataset = selected features + original label (+ batch and split info)
    X_new, y_new, b_new = load_period(NEW_BATCHES)
    split_old = np.empty(len(y_old), dtype=object)
    split_old[idx_tr], split_old[idx_val], split_old[idx_ht] = "train", "validation", "hist_test"
    names = [f"F{c + 1}" for c in cols]

    def to_df(X, y, b, split):
        df = pd.DataFrame(X[:, cols], columns=names)
        df["label"] = y; df["batch"] = b; df["split"] = split
        return df

    full = pd.concat([to_df(X_old, y_old, b_old, split_old),
                      to_df(X_new, y_new, b_new, np.array(["future"] * len(y_new), dtype=object))],
                     ignore_index=True)
    full.to_csv("results/selected_features_dataset.csv", index=False)
    print(f"\nSaved selected-features dataset: {full.shape}  -> selected_features_dataset.csv")
    print(f"Total time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

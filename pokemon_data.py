"""
Pokemon item-level duel data for PREF-SHAP.

Originally: load_experiments.py:
(Pokemon() / save_data() / reshape_data())

Change: The type one-hot columns are dropped so that d_eff stays under 16. 

    original repo : numeric stats (6) + Legendary (1) + one-hot Type (~18)  ->  d ~ 25
    now           : numeric stats (6) + Legendary (1) + one-hot Generation (6) -> d = 13

With d_eff < 16 the exact Shapley/interaction values can be computed with
`shapiq.ExactComputer`(2**13 = 8192 coalitions), which is what 
the approximators below are then benchmarked against. 

Sign convention :
    y = +1  <=>  left item won
    y = -1  <=>  right item won
The original repo's `reshape_data`: signs of y are flipped for otherwise-identical duels.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# pokemon.xls
#    Pokémon ID → feature vector
# combats.xls
#    Pokémon ID, Pokémon ID, winner ID
POKEMON_CSV = "datasets/pokemon/pokemon.xls"
COMBATS_CSV = "datasets/pokemon/combats.xls"


@dataclass
class PokemonItemData:
    stats: np.ndarray            # (n_items: Pokemon, d: features_names)
    feature_names: list[str]     # length d
    id_to_row: dict[int, int]    # pokemon ID -> row index into `stats`
    names: list[str]             # pokemon names, row-aligned with `stats`


def load_item_stats() -> PokemonItemData:
    """
    Load and preprocess `pokemon.xls`.
    Produces one feature vector for each Pokemon with 13 dimensions.

    numeric(6) + Legendary(1) + Generation one-hot(6) = 13 dims
    
    Original repo:    
    numeric(6) + Legendary(1) + Type one-hot(~18)     = ~25 dims
    """
    df = pd.read_csv(POKEMON_CSV)
    df = df.sort_values("#").reset_index(drop=True)

    id_to_row = {int(pid): i for i, pid in enumerate(df["#"].values)}
    names = df["Name"].tolist()

    numeric_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    numeric = df[numeric_cols].astype(float).values
    # Standardizatiion to make numerical dimensions more comparable
    numeric = StandardScaler().fit_transform(numeric)

    #legendary as binary feature
    legendary = df["Legendary"].astype(int).values.reshape(-1, 1)
    
    gen_dummies = pd.get_dummies(df["Generation"], prefix="gen").astype(float).values
    gen_names = [f"gen_{g}" for g in sorted(df["Generation"].unique())]
    stats = np.hstack([numeric, legendary, gen_dummies])
    feature_names = numeric_cols + ["Legendary"] + gen_names

    return PokemonItemData(
        stats=stats, feature_names=feature_names, id_to_row=id_to_row, names=names,
    )


def load_combats() -> pd.DataFrame:
    df = pd.read_csv(COMBATS_CSV)

    # assert: Winner_i ∈ {First_i, Second_i}
    assert (df["Winner"] == df["First_pokemon"]).sum() + \
           (df["Winner"] == df["Second_pokemon"]).sum() == len(df)
    return df


def build_duels(
    item_data: PokemonItemData,
    combats: pd.DataFrame,
    n_duels: int | None = None,
    random_state: int = 0,
):
    """
    Turn (First_pokemon, Second_pokemon, Winner) combats into (X_l, X_r, y)
    with a randomised left/right assignment (so the model can't shortcut on
    position)
     
    y = +1 if the left item won else -1.
    """
    rng = np.random.default_rng(random_state)
    combats = combats if n_duels is None else combats.sample(
        n=n_duels, random_state=random_state
    ).reset_index(drop=True)

    stats = item_data.stats
    id_to_row = item_data.id_to_row

    m = len(combats)
    d = stats.shape[1]
    X_l = np.empty((m, d))
    X_r = np.empty((m, d))
    y = np.empty(m)

    # randomised left/right assignment
    directions = rng.integers(0, 2, size=m)
    first = combats["First_pokemon"].values
    second = combats["Second_pokemon"].values
    winner = combats["Winner"].values

    for i in range(m):
        w_row = stats[id_to_row[int(winner[i])]]
        loser_id = second[i] if winner[i] == first[i] else first[i]
        l_row = stats[id_to_row[int(loser_id)]]

        if directions[i] == 1:  # winner on the left
            X_l[i], X_r[i], y[i] = w_row, l_row, 1.0
        else:                    # winner on the right
            X_l[i], X_r[i], y[i] = l_row, w_row, -1.0

    return X_l, X_r, y


def background_sample(item_data: PokemonItemData, n_ref: int = 200, random_state: int = 0):
    """
    Reference/background items for the conditional-mean-embedding term.
    
    X_bg is not duells, but single item vectors.
    """
    rng = np.random.default_rng(random_state)
    n = item_data.stats.shape[0]
    idx = rng.choice(n, size=min(n_ref, n), replace=False)
    return item_data.stats[idx]


if __name__ == "__main__":
    item_data = load_item_stats()
    combats = load_combats()
    X_l, X_r, y = build_duels(item_data, combats, n_duels=8000, random_state=0)
    X_bg = background_sample(item_data, n_ref=200)

    print("\nFeatures:", item_data.feature_names)
    print("\nDimension d =", X_l.shape[1])
    print("\nX_l", X_l.shape, "X_r", X_r.shape, "y", y.shape, "X_bg", X_bg.shape)
    print("\ny balance:", (y > 0).mean())

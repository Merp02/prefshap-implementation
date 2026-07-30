#!/usr/bin/env python3
"""
Minimaler Integrationstest für pref_fit.py.

Getestet werden:
1. Daten laden oder synthetisch erzeugen
2. reproduzierbarer Train/Test-Split
3. GPM mit Nyström-Kernel-Logistic-Regression fitten
4. Train- und Test-AUC berechnen
5. Shapes von alpha und Nyström-Centern prüfen
6. Skew-Symmetrie g(a,b) = -g(b,a) testen
7. Wahrscheinlichkeiten und Klassifikationsgenauigkeit prüfen
8. Modellbestandteile als .npz speichern

Beispiele
---------
Synthetischer Test:
    python run_g_learn_test.py

Eigene NumPy-Dateien:
    python test_g_learning.py --data-dir ./mein_datensatz

Erwartete Dateien im Datenordner:
    X_l.npy oder X_left.npy
    X_r.npy oder X_right.npy
    y.npy

Optional:
    U.npy oder context.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from prefshap_g_learning.pref_fit import fit_preference_model, median_lengthscale, rbf


LEFT_NAMES = ("X_l.npy", "X_left.npy", "Xl.npy", "l_processed.npy")
RIGHT_NAMES = ("X_r.npy", "X_right.npy", "Xr.npy","r_processed.npy")
LABEL_NAMES = ("y.npy", "Y.npy", "labels.npy")
CONTEXT_NAMES = ("U.npy", "context.npy", "contexts.npy")


def find_file(directory: Path, candidates: tuple[str, ...]) -> Optional[Path]:
    """Gibt die erste vorhandene Datei aus candidates zurück."""
    for name in candidates:
        path = directory / name
        if path.exists():
            print("Path: ", {path})
            return path
    return None


def load_numpy_dataset(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Lädt Duel-Daten aus einem Ordner."""
    left_path = find_file(data_dir, LEFT_NAMES)
    right_path = find_file(data_dir, RIGHT_NAMES)
    label_path = find_file(data_dir, LABEL_NAMES)
    context_path = find_file(data_dir, CONTEXT_NAMES)

    missing = []
    if left_path is None:
        missing.append(f"linke Items: {LEFT_NAMES}")
    if right_path is None:
        missing.append(f"rechte Items: {RIGHT_NAMES}")
    if label_path is None:
        missing.append(f"Labels: {LABEL_NAMES}")

    if missing:
        raise FileNotFoundError(
            "Im Datenordner fehlen benötigte Dateien:\n- "
            + "\n- ".join(missing)
        )

    X_l = np.load(left_path)
    X_r = np.load(right_path)
    y = np.load(label_path).reshape(-1)
    U = np.load(context_path) if context_path is not None else None

    return validate_dataset(X_l, X_r, y, U)


def make_synthetic_dataset(
    n_items: int = 500,
    n_matches: int = 5000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    """
    Erzeugt nicht vollständig rankbare Präferenzdaten.

    Je nach Clusterpaar entscheidet ein anderes kontinuierliches Feature
    über den Sieger. Dadurch kann ein allgemeines Präferenzmodell g sinnvoll
    getestet werden.
    """
    rng = np.random.default_rng(seed)

    continuous = rng.normal(size=(n_items, 4))
    clusters = rng.integers(0, 3, size=n_items)
    one_hot = np.eye(3)[clusters]
    X = np.hstack([continuous, one_hot])

    left_idx = rng.integers(0, n_items, size=n_matches)
    right_idx = rng.integers(0, n_items, size=n_matches)

    keep = left_idx != right_idx
    left_idx = left_idx[keep]
    right_idx = right_idx[keep]

    c_l = clusters[left_idx]
    c_r = clusters[right_idx]

    # Innerhalb desselben Clusters entscheidet Feature 0.
    # Zwischen Clustern entscheidet abhängig vom Clusterpaar Feature 1, 2 oder 3.
    feature_idx = np.zeros(len(left_idx), dtype=int)
    different = c_l != c_r

    pair_sum = c_l + c_r
    feature_idx[different & (pair_sum == 1)] = 1  # Cluster 0 vs. 1
    feature_idx[different & (pair_sum == 2)] = 2  # Cluster 0 vs. 2
    feature_idx[different & (pair_sum == 3)] = 3  # Cluster 1 vs. 2

    left_strength = continuous[left_idx, feature_idx]
    right_strength = continuous[right_idx, feature_idx]

    # Kleine logistische Störung vermeidet ein komplett triviales Problem.
    margin = left_strength - right_strength + rng.normal(
        loc=0.0, scale=0.15, size=len(left_idx)
    )
    y = np.where(margin >= 0.0, 1, -1)

    return validate_dataset(X[left_idx], X[right_idx], y, None)


def validate_dataset(
    X_l: np.ndarray,
    X_r: np.ndarray,
    y: np.ndarray,
    U: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validiert Shapes, Datentypen und Labels."""
    X_l = np.asarray(X_l, dtype=float)
    X_r = np.asarray(X_r, dtype=float)
    y = np.asarray(y).reshape(-1)
    U = None if U is None else np.asarray(U, dtype=float)

    if X_l.ndim != 2 or X_r.ndim != 2:
        raise ValueError(
            f"X_l und X_r müssen 2D sein, erhalten: {X_l.shape}, {X_r.shape}"
        )
    if X_l.shape != X_r.shape:
        raise ValueError(
            f"X_l und X_r müssen dieselbe Form haben: {X_l.shape} != {X_r.shape}"
        )
    if len(y) != len(X_l):
        raise ValueError(
            f"Labelanzahl {len(y)} passt nicht zu {len(X_l)} Duellen."
        )
    if U is not None and len(U) != len(X_l):
        raise ValueError(
            f"Kontextanzahl {len(U)} passt nicht zu {len(X_l)} Duellen."
        )
    if len(X_l) < 10:
        raise ValueError("Für einen sinnvollen Test werden mindestens 10 Duelle benötigt.")
    if not np.isfinite(X_l).all() or not np.isfinite(X_r).all():
        raise ValueError("X_l oder X_r enthält NaN/Inf.")
    if U is not None and not np.isfinite(U).all():
        raise ValueError("U enthält NaN/Inf.")

    y = np.where(y > 0, 1, -1)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(
            f"Es werden beide Klassen -1 und +1 benötigt, gefunden: {classes}"
        )

    return X_l, X_r, y, U


def split_dataset(
    X_l: np.ndarray,
    X_r: np.ndarray,
    y: np.ndarray,
    U: Optional[np.ndarray],
    test_size: float,
    seed: int,
):
    """Gemeinsamer, stratifizierter Split für alle Duel-Komponenten."""
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )

    U_train = None if U is None else U[train_idx]
    U_test = None if U is None else U[test_idx]

    return (
        X_l[train_idx],
        X_r[train_idx],
        y[train_idx],
        U_train,
        X_l[test_idx],
        X_r[test_idx],
        y[test_idx],
        U_test,
        train_idx,
        test_idx,
    )


def run_test(args: argparse.Namespace) -> dict:
    """Führt den vollständigen Integrationstest aus."""
    if args.data_dir is None:
        print("[INFO] Verwende synthetische Präferenzdaten.")
        X_l, X_r, y, U = make_synthetic_dataset(
            n_items=args.n_items,
            n_matches=args.n_matches,
            seed=args.seed,
        )
        source = "synthetic"
    else:
        data_dir = Path(args.data_dir).expanduser().resolve()
        print(f"[INFO] Lade Daten aus: {data_dir}")
        X_l, X_r, y, U = load_numpy_dataset(data_dir)
        source = str(data_dir)

    print(f"[INFO] X_l shape: {X_l.shape}")
    print(f"[INFO] X_r shape: {X_r.shape}")
    print(f"[INFO] y shape:   {y.shape}")
    print(f"[INFO] Kontext:   {'ja' if U is not None else 'nein'}")
    print(f"[INFO] Klassen:   {dict(zip(*np.unique(y, return_counts=True)))}")

    (
        X_l_train,
        X_r_train,
        y_train,
        U_train,
        X_l_test,
        X_r_test,
        y_test,
        U_test,
        train_idx,
        test_idx,
    ) = split_dataset(
        X_l=X_l,
        X_r=X_r,
        y=y,
        U=U,
        test_size=args.test_size,
        seed=args.seed,
    )

    print(f"[INFO] Train-Duelle: {len(y_train)}")
    print(f"[INFO] Test-Duelle:  {len(y_test)}")

    stacked_train = np.vstack([X_l_train, X_r_train])
    lengthscale = median_lengthscale(stacked_train, seed=args.seed)
    base_kernel = rbf(lengthscale)

    print(f"[INFO] RBF lengthscale: {lengthscale:.6g}")
    print("[INFO] Fitte Präferenzmodell ...")

    model = fit_preference_model(
        X_left=X_l_train,
        X_right=X_r_train,
        y=y_train,
        kernel=base_kernel,
        context=U_train,
        n_centres=args.n_centres,
        lam=args.lam,
        random_state=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    train_scores = model.decision_function(X_l_train, X_r_train, U_train)
    test_scores = model.decision_function(X_l_test, X_r_test, U_test)

    train_auc = roc_auc_score(y_train > 0, train_scores)
    test_auc = roc_auc_score(y_test > 0, test_scores)

    test_pred = np.where(test_scores >= 0.0, 1, -1)
    test_accuracy = accuracy_score(y_test, test_pred)

    test_prob = model.predict_proba(X_l_test, X_r_test, U_test)

    n_skew = min(args.skew_samples, len(y_test))
    ab = model.decision_function(
        X_l_test[:n_skew], X_r_test[:n_skew], None if U_test is None else U_test[:n_skew]
    )
    ba = model.decision_function(
        X_r_test[:n_skew], X_l_test[:n_skew], None if U_test is None else U_test[:n_skew]
    )
    skew_error = float(np.max(np.abs(ab + ba)))

    kernel_matrix = model._kernel_against_centres(
        X_l_test[: min(10, len(X_l_test))],
        X_r_test[: min(10, len(X_r_test))],
        None if U_test is None else U_test[: min(10, len(U_test))],
    )
    reconstructed_scores = model.fit.scores_from_kernel(kernel_matrix)
    direct_scores = test_scores[: len(reconstructed_scores)]
    reconstruction_error = float(
        np.max(np.abs(reconstructed_scores - direct_scores))
    )

    results = {
        "source": source,
        "seed": args.seed,
        "n_duels": int(len(y)),
        "n_features": int(X_l.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_centres": int(len(model.Xl_c)),
        "alpha_shape": list(model.alpha.shape),
        "centre_left_shape": list(model.Xl_c.shape),
        "centre_right_shape": list(model.Xr_c.shape),
        "lengthscale": float(lengthscale),
        "lambda": float(args.lam),
        "jitter_used": float(model.fit.jitter_used),
        "train_auc": float(train_auc),
        "test_auc": float(test_auc),
        "test_accuracy": float(test_accuracy),
        "probability_min": float(test_prob.min()),
        "probability_max": float(test_prob.max()),
        "skew_symmetry_max_error": skew_error,
        "kernel_score_reconstruction_max_error": reconstruction_error,
    }

    print("\n" + "=" * 68)
    print("TESTERGEBNIS")
    print("=" * 68)
    print(f"alpha shape:                 {model.alpha.shape}")
    print(f"Center links shape:          {model.Xl_c.shape}")
    print(f"Center rechts shape:         {model.Xr_c.shape}")
    print(f"Jitter verwendet:            {model.fit.jitter_used:.3e}")
    print(f"Train AUC:                   {train_auc:.4f}")
    print(f"Test AUC:                    {test_auc:.4f}")
    print(f"Test Accuracy:               {test_accuracy:.4f}")
    print(f"Wahrscheinlichkeitsbereich:  [{test_prob.min():.6f}, {test_prob.max():.6f}]")
    print(f"Max. Skew-Symmetriefehler:   {skew_error:.3e}")
    print(f"Max. Score-Rekonstruktionsf.: {reconstruction_error:.3e}")

    passed = True

    if not np.isfinite(model.alpha).all():
        print("[FAIL] alpha enthält NaN oder Inf.")
        passed = False
    if not np.isfinite(test_scores).all():
        print("[FAIL] Test-Scores enthalten NaN oder Inf.")
        passed = False
    if not (0.0 <= test_prob.min() <= test_prob.max() <= 1.0):
        print("[FAIL] predict_proba liefert Werte außerhalb [0, 1].")
        passed = False
    if skew_error > args.skew_tolerance:
        print(
            f"[FAIL] Skew-Symmetriefehler {skew_error:.3e} "
            f"> Toleranz {args.skew_tolerance:.3e}."
        )
        passed = False
    if reconstruction_error > args.reconstruction_tolerance:
        print(
            f"[FAIL] Score-Rekonstruktionsfehler {reconstruction_error:.3e} "
            f"> Toleranz {args.reconstruction_tolerance:.3e}."
        )
        passed = False
    if test_auc < args.min_auc:
        print(
            f"[WARN] Test-AUC {test_auc:.4f} liegt unter dem gesetzten "
            f"Mindestwert {args.min_auc:.4f}."
        )

    results["passed"] = passed

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_payload = {
        "alpha": model.alpha,
        "beta": model.fit.beta,
        "L": model.fit.L,
        "Xl_centres": model.Xl_c,
        "Xr_centres": model.Xr_c,
        "centre_indices_in_train": model.centres,
        "train_indices": train_idx,
        "test_indices": test_idx,
        "lengthscale": np.array(lengthscale),
        "lam": np.array(args.lam),
        "jitter_used": np.array(model.fit.jitter_used),
    }
    if model.U_c is not None:
        save_payload["U_centres"] = model.U_c

    np.savez_compressed(output_path, **save_payload)

    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[INFO] Modellbestandteile gespeichert: {output_path}")
    print(f"[INFO] Testergebnisse gespeichert:    {metrics_path}")

    if passed:
        print("\n[PASS] Der GPM-Integrationstest war erfolgreich.")
    else:
        print("\n[FAIL] Mindestens eine harte Prüfung ist fehlgeschlagen.")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrationstest für das Lernen des Präferenzmodells g."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Ordner mit X_l.npy, X_r.npy und y.npy. Ohne Angabe: synthetische Daten.",
    )
    parser.add_argument("--output", type=str, default="artifacts/gpm_test_model.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-items", type=int, default=500)
    parser.add_argument("--n-matches", type=int, default=5000)
    parser.add_argument(
        "--n-centres",
        type=int,
        default=300,
        help="Anzahl Nyström-Center.",
    )
    parser.add_argument("--lam", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--skew-samples", type=int, default=200)
    parser.add_argument("--skew-tolerance", type=float, default=1e-8)
    parser.add_argument("--reconstruction-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.5,
        help="Nur Warnschwelle, kein harter Abbruch.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        results = run_test(args)
        return 0 if results["passed"] else 1
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Exact synthetic check; see round-three-b/method-check/protocol.md."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/intelligence-history/round-three-b/method-check"


def gram(rows):
    return [[sum(F(x) * F(y) for x, y in zip(a, b)) for b in rows] for a in rows]


def shared_cov(equations):
    # Independent scalar path: match named independent unit-variance inputs.
    result = []
    for left in equations:
        row = []
        for right in equations:
            value = F(0)
            for symbol, coefficient in left.items():
                if symbol in right:
                    value += F(coefficient) * F(right[symbol])
            row.append(value)
        result.append(row)
    return result


def matvec(matrix, vector):
    return [sum(x * y for x, y in zip(row, vector)) for row in matrix]


def check():
    a = [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]
    b = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    sa = shared_cov([{"g": 1, "e1": 1}, {"g": 1, "e2": 1}, {"g": 1, "e3": 1}])
    sb = shared_cov([{"z12": 1, "z13": 1}, {"z12": 1, "z23": 1}, {"z13": 1, "z23": 1}])
    expected = [[F(2 if i == j else 1) for j in range(3)] for i in range(3)]
    ca, cb = gram(a), gram(b)
    assert ca == cb == sa == sb == expected
    # Three linearly independent eigenvectors, determinant 3, span R^3.
    pairs = [(4, [1, 1, 1]), (1, [1, -1, 0]), (1, [1, 0, -1])]
    for eigenvalue, vector in pairs:
        assert matvec(ca, vector) == [eigenvalue * x for x in vector]
    determinant = 1 * (1 - 0) - 1 * (-1 - 0) + 1 * (1 - 0)
    assert determinant == 3
    corr = [[v / 2 for v in row] for row in ca]
    assert all(corr[i][j] == F(1 if i == j else 1, 1 if i == j else 2)
               for i in range(3) for j in range(3))
    negative_b = [[2, 1, 0], [1, 0, 1], [0, 1, 1]]
    negative = gram(negative_b)
    independent_negative = shared_cov([
        {"z12": 2, "z13": 1}, {"z12": 1, "z23": 1}, {"z13": 1, "z23": 1}])
    assert negative == independent_negative and negative != expected
    # Mean path one: intercept moves, latent mean remains zero.
    intercept_case = [F(1, 2) + loading * 0 for loading in [1, 1, 1]]
    # Mean path two: intercept stays zero, latent mean moves.
    latent_case = matvec([[1], [1], [1]], [F(1, 2)])
    assert intercept_case == latent_case == [F(1, 2)] * 3
    return {
        "status": "passed",
        "kind": "synthetic-exact-algebra-not-paper-replication",
        "protocolSHA256": hashlib.sha256((OUT / "protocol.md").read_bytes()).hexdigest(),
        "implementationSHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coefficientMatrices": {"A": a, "B": b},
        "covariance": ca, "correlation": corr,
        "eigenpairs": pairs, "eigenvectorBasisDeterminant": determinant,
        "independentScalarAgreement": True,
        "negativeControl": {"matrix": negative_b, "covariance": negative, "equalityRejected": True},
        "meanDifference": {"interceptShift": intercept_case, "latentMeanShift": latent_case},
        "analyticConclusion": "Equal zero means and covariances imply identical observed multivariate Gaussian distributions under the stated independent Gaussian input assumptions.",
        "limits": "No human data, no mechanism identification, no POT simulation, no empirical replication.",
    }


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    matrices = [result["covariance"], result["covariance"], result["negativeControl"]["covariance"]]
    for ax, values, title in zip(axes, matrices, ["Model A: shared g", "Model B: pairwise inputs", "Negative control: changed B"]):
        arr = np.array(values, dtype=float)
        ax.imshow(arr, vmin=0, vmax=5, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(values[i][j]), ha="center", va="center", color="white" if arr[i, j] > 3 else "#193d50", fontsize=18)
        ax.set_xticks(range(3), ["X1", "X2", "X3"])
        ax.set_yticks(range(3), ["X1", "X2", "X3"])
        ax.set_title(title, fontsize=12)
    fig.suptitle("Exact covariance: two constructions agree; the negative control differs", fontsize=13)
    fig.savefig(OUT / "covariance.svg", metadata={"Date": None})
    fig.savefig("/private/tmp/intelligence-history-r3b/covariance-preview.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = check()
    content = json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"
    target = OUT / "results.json"
    if args.check:
        assert target.read_text() == content, "Stale method-check result"
    else:
        target.write_text(content)
    if args.plot:
        plot(result)
    print("PASS: exact covariance, independent scalar path, eigenbasis, negative control, mean ambiguity")

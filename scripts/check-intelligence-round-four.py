"""Exact counterexamples for round four. Standard library only; no empirical data."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import itertools
import json
import sys

ROOT = Path(__file__).resolve().parents[1] / "docs/intelligence-history/round-four"


def solve(weights, offsets, interventions=None):
    """Topologically ordered linear SCM; intervention replaces the whole equation."""
    interventions = interventions or {}
    values = []
    for i, row in enumerate(weights):
        assert not any(row[i:]), "Only acyclic, ordered models are accepted"
        values.append(interventions[i] if i in interventions else
                      offsets[i] + sum(row[j] * values[j] for j in range(i)))
    return values


def covariance(weights, noise_variance):
    n = len(weights)
    responses = [solve(weights, [F(i == j) for i in range(n)]) for j in range(n)]
    return [[sum(responses[k][i] * responses[k][j] * noise_variance[k]
                 for k in range(n)) for j in range(n)] for i in range(n)]


def determinant(m):
    return sum((F(-1) ** sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3)))
               * m[0][p[0]] * m[1][p[1]] * m[2][p[2]]
               for p in itertools.permutations(range(3)))


# F uses G'=G/sqrt(2), e'=e/sqrt(2), with variance 1/2: exact reparameterization
# of protocol F, so square roots never need floating point approximation.
common = [[F(0)] * 4, [F(1), F(0), F(0), F(0)],
          [F(1), F(0), F(0), F(0)], [F(1), F(0), F(0), F(0)]]
chain = [[F(0)] * 3, [F(1, 2), F(0), F(0)], [F(1, 3), F(1, 3), F(0)]]
sigma_f = [row[1:] for row in covariance(common, [F(1, 2)] * 4)[1:]]
sigma_c = covariance(chain, [F(1), F(3, 4), F(2, 3)])
target = [[F(1) if i == j else F(1, 2) for j in range(3)] for i in range(3)]
interventions = []
for target_index in [0, 1]:
    interventions.append({"target": f"do(X{target_index + 1}=1)",
                          "F": solve(common, [F(0)] * 4, {target_index + 1: F(1)})[1:],
                          "C": solve(chain, [F(0)] * 3, {target_index: F(1)})})
circuits = [{"A": a, "B": b, "D1": a & b, "D2": a,
             "testedPreviously": b == 1} for a, b in itertools.product([0, 1], repeat=2)]
checks = {
    "exact_covariance": sigma_f == sigma_c == target,
    "determinants": determinant(sigma_f) == determinant(sigma_c) == F(1, 2),
    "intervention_x1": interventions[0]["F"] == [1, 0, 0] and
                       interventions[0]["C"] == [1, F(1, 2), F(1, 2)],
    "intervention_x2": interventions[1]["F"] == [0, 1, 0] and
                       interventions[1]["C"] == [0, 1, F(1, 3)],
    "circuit_old_context": all(c["D1"] == c["D2"] for c in circuits if c["testedPreviously"]),
    "circuit_new_context": any(c["D1"] != c["D2"] for c in circuits if not c["testedPreviously"]),
}
assert all(checks.values()), checks


def encode(x):
    if isinstance(x, F):
        return str(x)
    raise TypeError(type(x))


result = {"kind": "exact-synthetic-counterexample", "date": "2026-10-28",
          "protocolSHA256": hashlib.sha256((ROOT / "method-check/protocol.md").read_bytes()).hexdigest(),
          "scriptSHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "assumptions": "Independent zero-mean Gaussian disturbances; ideal surgical interventions; fixed equations.",
          "covarianceF": sigma_f, "covarianceC": sigma_c, "determinant": determinant(sigma_c),
          "interventions": interventions, "circuitTruthTable": circuits, "checks": checks}
output = json.dumps(result, ensure_ascii=False, indent=2, default=encode) + "\n"
destination = ROOT / "method-check/results.json"
if "--check" in sys.argv:
    assert destination.read_text() == output, "Results are stale"
else:
    destination.write_text(output)
print(json.dumps({"checks": checks, "mode": "checked" if "--check" in sys.argv else "generated"}))

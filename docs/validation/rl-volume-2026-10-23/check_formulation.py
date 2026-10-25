"""Independent arithmetic checks; no tokenizer or model execution."""
import json
import math
from pathlib import Path

rows = [(0.2, 1), (0.3, 0), (0.4, 0), (0.1, 0)]
assert math.isclose(sum(p for p, _ in rows), 1)
expected = sum(p * r for p, r in rows)
at_least_one = 1 - (1 - expected) ** 4
assert math.isclose(at_least_one, 0.5904)
cases = []
for a, b in [(2, 12), (3, 3), (4, 2), (2, 3), (2, 5), (2, 6)]:
    n = next(n for n in range(b + 1) if 3 * n + a >= b)
    assert n == max(0, (b - a + 2) // 3)
    cases.append(dict(a=a, b=b, n=n))
result = dict(kind='constructed arithmetic only; no model run',
              expected_reward=expected, four_independent_samples=at_least_one,
              boundaries=cases)
Path(__file__).with_name('formulation-examples.json').write_text(
    json.dumps(result, indent=2) + '\n')
print('Arithmetic and boundary checks passed')

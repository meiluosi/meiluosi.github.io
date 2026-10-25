"""Executed toy contracts, not model/quantizer/server validation."""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
reference = [1, 2, 10, 3, 4]
assert reference != [1, 1, 2, 10, 3, 4]
assert reference != [1, 2, 10, 3]
q = lambda x: x.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
w, d = Decimal('0.14'), Decimal('0.04')
paths = [q(w+d), q(w)+d, q(q(w)+d)]
assert paths == [Decimal('0.2'), Decimal('0.14'), Decimal('0.1')]
argmax = lambda xs: max(range(len(xs)), key=xs.__getitem__)
assert argmax([1.001, 1.000]) == 0
assert argmax([0.999, 1.000]) == 1
stream = ['JSON_START', 'JSON_BODY', 'JSON_END']
def consume(limit, stop):
    result = []
    for token in stream[:limit]:
        result.append(token)
        if token == stop:
            break
    return result
assert consume(3, 'JSON_END') == stream
assert consume(2, 'JSON_END') == stream[:2]
assert consume(3, 'JSON_BODY') == stream[:2]
counts = {'both_pass':70, 'repaired':5, 'regressed':5, 'both_fail':20}
assert sum(counts.values()) == 100
assert (70+5)/100 == .75
out = {'scope':'toy arithmetic and protocol only', 'rounding_paths':list(map(str,paths)), 'paired_counts':counts, 'reference_rate':.75, 'candidate_rate':.75, 'regressions':5}
Path(__file__).with_name('delivery-examples.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))

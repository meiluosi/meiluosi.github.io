"""Independent arithmetic checks for SFT 04–06; no model training."""
import json
import math

A = [[1, -1, 0]]
B = [[2], [1]]
x = [3, 1, 4]
BA = [[sum(B[i][k] * A[k][j] for k in range(1))
       for j in range(3)] for i in range(2)]
Ax = [sum(a*b for a, b in zip(row, x)) for row in A]
branch = [sum(a*b for a, b in zip(row, Ax)) for row in B]
direct = [sum(a*b for a, b in zip(row, x)) for row in BA]
assert BA == [[2, -2, 0], [1, -1, 0]]
assert branch == direct == [4, 2]
assert all(BA[0][j] == 2*BA[1][j] for j in range(3))
full = 4096**2
adapter = 8*(4096+4096)
assert full == 16777216 and adapter == 65536
assert adapter/full*100 == .390625
four_bit_gib = 10**9*4/8/2**30
sixteen_bit_gib = 10**9*16/8/2**30
assert round(four_bit_gib, 3) == .466
assert round(sixteen_bit_gib, 3) == 1.863
paired = dict(both_correct=66, fixed=12, regressed=4, both_wrong=18)
assert sum(paired.values()) == 100
assert paired['both_correct'] + paired['fixed'] == 78
assert paired['both_correct'] + paired['regressed'] == 70
assert paired['fixed'] - paired['regressed'] == 8
assert math.isclose(200/300, 2/3)
# Ensure the field definition does not accept a JSON boolean as an integer.
assert isinstance(True, int)  # This is why a loose isinstance check is insufficient.
def steps_valid(value):
    return value is None or type(value) is int
assert steps_valid(0) and steps_valid(None)
assert not steps_valid(True) and not steps_valid(1.2)
print(json.dumps({
    'kind': 'independent-teaching-example-validation',
    'BA': BA,
    'adapter_output': branch,
    'full_matrix_parameters': full,
    'adapter_parameters': adapter,
    'adapter_percentage': adapter/full*100,
    'ideal_weight_storage_GiB': {'4_bit': four_bit_gib, '16_bit': sixteen_bit_gib},
    'paired_outcomes': paired,
    'all_null_field_accuracy': 200/300,
    'boolean_step_rejected': True,
    'scope': 'Toy algebra, units and metric counts only; no training or hardware measurement.'
}, ensure_ascii=False, indent=2))

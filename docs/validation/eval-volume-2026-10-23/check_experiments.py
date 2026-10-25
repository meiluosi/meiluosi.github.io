"""Constructed experiment-design arithmetic, no training runs."""
import json
import math
from pathlib import Path
A=[60,70,65];B=[64,72,66]
differences=[b-a for a,b in zip(A,B)]
assert differences==[4,2,1]
assert math.isclose(sum(differences)/3,7/3)
assert max(B)-min(A)==12
interaction=(74-65)-(62-60)
assert interaction==7
assert 88/99<81/90 and 0/1<2/10
assert (88+0)/100>(81+2)/100
# Standardize both to A's declared mixture, removing the mixture advantage.
standardized=.9*(88/99)+.1*0
assert math.isclose(standardized,.8) and standardized<.83
result=dict(paired_differences=differences,paired_mean=sum(differences)/3,
            cherry_picked_difference=12,interaction=interaction,
            mixture_A=.83,mixture_B=.88,standardized_B=standardized,
            training_runs=0)
Path(__file__).with_name('experiment-examples.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

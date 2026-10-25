"""Executed scalar checkpoint example; no neural network or GPU."""
import json
from pathlib import Path
from math import isclose

def step(state,target=1):
    v=.9*state['v']+(state['w']-target)
    return dict(w=state['w']-.1*v,v=v,step=state['step']+1)
state=dict(w=0.,v=0.,step=0)
state=step(step(state))
assert isclose(state['w'],.28) and isclose(state['v'],-1.8)
path=Path(__file__).with_name('scalar-checkpoint.json')
path.write_text(json.dumps(state)+'\n')
loaded=json.loads(path.read_text())
uninterrupted=step(state)
resumed=step(loaded)
weights_only=step(dict(w=loaded['w'],v=0.,step=loaded['step']))
changed_data=step(loaded,2)
assert uninterrupted==resumed
assert isclose(resumed['w'],.514)
assert isclose(weights_only['w'],.352)
assert isclose(changed_data['w'],.614)
assert step(resumed)==step(uninterrupted)
a=(1e16+-1e16)+1
b=1e16+(-1e16+1)
assert (a,b)==(1.,0.)
assert isclose((.2+1)/2,.6)
assert isclose((10*.2+30*1)/40,.8)
result=dict(checkpoint=state,resumed=resumed,weights_only=weights_only,
            changed_data=changed_data,float_orders=[a,b],
            microbatch_mean=.6,token_mean=.8,neural_network_runs=0)
Path(__file__).with_name('reproduction-examples.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

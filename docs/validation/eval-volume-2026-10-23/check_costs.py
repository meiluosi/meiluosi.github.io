"""Synthetic accounting only, not hardware measurements."""
import json
from pathlib import Path
from math import isclose
inputs=2*(800+200);targets=2*200;positions=2*1024
assert (inputs,targets,positions-inputs)==(2000,400,48)
assert [inputs/.5,targets/.5,positions/.5]==[4000,800,4096]
flops=6*10**8*10**7
assert flops==6*10**15
length_ratio=2048**2/(2*1024**2)
assert length_ratio==2
ledger={'weights':2,'gradients':2,'master':4,'adam_m':4,'adam_v':4}
assert sum(ledger.values())==16
memory=10**8*sum(ledger.values());gib=memory/2**30
assert isclose(gib,1.4901161193847656)
time=70/2+30
assert time==65 and isclose(100/time,20/13)
result=dict(nonpadding=inputs,targets=targets,padding=48,
            throughputs=[4000,800,4096],approx_flops=flops,
            attention_square_ratio=length_ratio,assumed_bytes=memory,
            assumed_gib=gib,optimized_minutes=time,speedup=100/time,
            search_hours=10*2,hardware_measurements=0)
Path(__file__).with_name('cost-examples.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

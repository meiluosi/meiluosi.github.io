import math,json
from pathlib import Path

def corr(a,b):
 x=[v-sum(a)/len(a) for v in a];y=[v-sum(b)/len(b) for v in b]
 return sum(i*j for i,j in zip(x,y))/math.sqrt(sum(i*i for i in x)*sum(j*j for j in y))
lengths=[12,13,7,-12,4];correct=[1,1,1,1,0]
assert sum(x>0 for x in lengths)/5==.8
pairs=[(60,28,False,True),(55,25,False,True),(52,31,False,True),(30,44,True,False)]
assert sum(a>b for a,b,_,_ in pairs)/4==.75
assert sum(c for _,_,c,_ in pairs)/4==.25
assert sum(c if a>b else d for a,b,c,d in pairs)==0
p=[.5,.6,.69,.78,.84];w=[.5,.62,.71,.78,.8];q=[.6,.58,.49,.31,.18];m=[120,145,190,260,410]
rs=[corr(p,w),corr(p,q),corr(m,w),corr(m,q)]
assert [round(x,3) for x in rs]==[.987,-.949,.840,-.976]
assert round(corr(lengths,correct),3)==.044
out={'scope':'constructed data only','length_correctness_correlation':corr(lengths,correct),'checkpoint_correlations':rs,'length_ratio':410/120,'judge_selected_correct':.25,'length_selected_correct':0}
Path(__file__).with_name('correctness-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

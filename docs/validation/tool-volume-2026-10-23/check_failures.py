"""Validate constructed recovery data, not model performance."""
import copy
import json
import math
from pathlib import Path

def execute(request):
    assert request['tool']=='get_stock' and request['warehouse']=='HZ'
    if request['sku']!='A17':
        return {'status':'error','code':'unknown_sku'}
    return {'status':'ok','available':2}

wrong={'tool':'get_stock','sku':'A18','warehouse':'HZ'}
right={**wrong,'sku':'A17'}
request={'role':'user','content':'Need 3 A17; query HZ first.'}
error=execute(wrong)
result=execute(right)
assert error=={'status':'error','code':'unknown_sku'}
assert result=={'status':'ok','available':2}
first=dict(context=[request],target=right)
recovery=dict(context=[request,{'role':'assistant','call':wrong},
                       {'role':'tool','content':error}],target=right)
assert first['target']==recovery['target']
assert len(first['context'])==1
assert recovery['context'][1]['call']==wrong
# The selected event targets exclude history. This is not a tokenizer mask test.
selected_targets=[first['target'],recovery['target']]
assert all(t['sku']=='A17' for t in selected_targets)
assert all(t!=wrong for t in selected_targets)

def validate_pair(call,observation):
    assert execute(call)==observation
validate_pair(right,result)
try:
    validate_pair(right,error)
except AssertionError:
    pass
else:
    raise AssertionError('stale result accepted')
original=copy.deepcopy(recovery)
branch={'parent':'failed-001','checkpoint':2,'call':copy.deepcopy(right),
        'result':copy.deepcopy(result),'validation':'local call only'}
branch['call']['sku']='mutated-copy'
assert recovery==original
pool=[('easy',i<45) for i in range(50)]+[('hard',i<10) for i in range(50)]
kept=[kind for kind,success in pool if success]
assert len(kept)==55 and kept.count('easy')==45
fraction=kept.count('easy')/len(kept)
assert math.isclose(fraction,9/11)
output=dict(first_action_targets=1,recovery_targets=1,wrong_action_targets=0,
            stale_result_rejected=True,original_preserved=True,
            retained=55,easy_fraction=fraction,model_runs=0,
            scope='event target selection and fixture execution; no tokenization')
Path(__file__).with_name('failure-examples.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output))

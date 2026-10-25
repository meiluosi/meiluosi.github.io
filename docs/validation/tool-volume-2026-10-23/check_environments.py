"""Serial in-memory contracts, deliberately no model or remote service."""
import copy
import json
from pathlib import Path

class Stock:
    def __init__(self, clear_cache=True):
        self.clear_cache=clear_cache
        self.cache={}
    def reset(self, instance, hz, sz, faults=False):
        self.instance=instance
        self.stock={'HZ':hz,'SZ':sz}
        self.faults=faults
        self.attempts={}
        if self.clear_cache:
            self.cache={}
    def query(self, sku, warehouse):
        assert sku=='A17' and warehouse in self.stock
        key=(sku,warehouse)
        self.attempts[key]=self.attempts.get(key,0)+1
        if self.faults and warehouse=='HZ' and self.attempts[key]==1:
            return {'status':'timeout'}
        if key not in self.cache:
            self.cache[key]={'status':'ok','available':self.stock[warehouse]}
        return copy.deepcopy(self.cache[key])

bad=Stock(False)
bad.reset('A',2,5);assert bad.query('A17','HZ')['available']==2
bad.reset('B',4,5);assert bad.query('A17','HZ')['available']==2
orders=[]
for order in [('A','B'),('B','A')]:
    env=Stock();results={}
    for instance in order:
        hz={'A':2,'B':4}[instance]
        env.reset(instance,hz,5)
        result=env.query('A17','HZ')
        results[instance]=result['available']
        result['available']=999
        assert env.query('A17','HZ')['available']==hz
    assert results=={'A':2,'B':4}
    orders.append(results)
for order in [('HZ','SZ','HZ'),('SZ','HZ','HZ')]:
    env=Stock();env.reset('fault-A',2,5,True)
    observations=[(w,env.query('A17',w)) for w in order]
    assert [r for w,r in observations if w=='HZ']==[{'status':'timeout'},{'status':'ok','available':2}]
    env.reset('fault-A',2,5,True)
    assert env.query('A17','HZ')=={'status':'timeout'}

class Replay:
    def __init__(self, records):
        self.records=copy.deepcopy(records)
        self.cursor=0
    def query(self, request):
        if self.cursor>=len(self.records):
            raise ValueError('replay exhausted')
        wanted,result=self.records[self.cursor]
        if request!=wanted:
            raise ValueError('request mismatch')
        self.cursor+=1
        return copy.deepcopy(result)

request={'tool':'get_stock','sku':'A17','warehouse':'HZ'}
replay=Replay([(request,{'status':'ok','available':2})])
try:
    replay.query({**request,'warehouse':'SZ'})
except ValueError:
    pass
else:
    raise AssertionError('mismatch accepted')
assert replay.cursor==0
assert replay.query(request)['available']==2
try:
    replay.query(request)
except ValueError:
    pass
else:
    raise AssertionError('exhausted replay accepted')
result=dict(stale_cache_reproduced=True,corrected_orders=orders,
            observation_copy=True,keyed_fault_orders=2,counter_reset=True,
            replay_mismatch_rejected=True,replay_exhaustion_rejected=True,
            scope='serial in-memory fixture; no model calls')
Path(__file__).with_name('environment-examples.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

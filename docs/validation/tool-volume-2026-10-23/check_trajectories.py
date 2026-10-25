"""Check article fixture and deliberate corruptions; never invokes a model."""
import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / 'src/content/posts/2026-10-23-llm-training-tool-01-trajectories.md'
FIXTURE = json.loads(re.search(r'```json\n(.*?)\n```', ARTICLE.read_text(), re.S)[1])


def validate(events, stock):
    """Narrow serial, success-only protocol; free text deliberately not scored."""
    assert set(stock) == {'HZ', 'SZ'}
    assert all(type(v) is int and v >= 0 for v in stock.values())
    expected = 'HZ' if stock['HZ'] >= 3 else 'SZ' if stock['SZ'] >= 3 else None
    route = ['HZ'] if stock['HZ'] >= 3 else ['HZ', 'SZ']
    assert isinstance(events, list) and len(events) == 2 * len(route) + 1
    ids = set()
    for i, warehouse in enumerate(route):
        call, result = events[2*i:2*i+2]
        assert set(call) == {'kind', 'id', 'name', 'arguments'}
        assert call['kind'] == 'call' and call['name'] == 'get_stock'
        assert type(call['id']) is str and call['id'] and call['id'] not in ids
        ids.add(call['id'])
        assert call['arguments'] == {'sku': 'A17', 'warehouse': warehouse}
        assert set(result) == {'kind', 'call_id', 'status', 'sku', 'warehouse', 'available', 'snapshot'}
        assert result['kind'] == 'result' and result['status'] == 'ok'
        assert result['call_id'] == call['id']
        assert result['sku'] == 'A17' and result['warehouse'] == warehouse
        assert type(result['available']) is int and result['available'] == stock[warehouse]
        assert result['snapshot'] == 'stock-v1'
    final = events[-1]
    assert set(final) == {'kind', 'selected_warehouse', 'reserved', 'content'}
    assert final['kind'] == 'final' and final['selected_warehouse'] == expected
    assert final['reserved'] is False
    assert isinstance(final['content'], str) and final['content']


def construct(stock):
    """Independent procedure: sequential lookup and stop when enough units exist."""
    events = []
    selected = None
    for warehouse in ['HZ', 'SZ']:
        cid = f'c{len(events)//2+1}'
        events += [dict(kind='call', id=cid, name='get_stock',
                        arguments=dict(sku='A17', warehouse=warehouse)),
                   dict(kind='result', call_id=cid, status='ok', sku='A17',
                        warehouse=warehouse, available=stock[warehouse], snapshot='stock-v1')]
        if stock[warehouse] >= 3:
            selected = warehouse
            break
    events.append(dict(kind='final', selected_warehouse=selected, reserved=False,
                       content='Structured fixture; natural-language correctness is not scored.'))
    return events


validate(FIXTURE, {'HZ': 2, 'SZ': 5})
counts = {'HZ': 0, 'SZ': 0, 'none': 0}
for hz in range(6):
    for sz in range(6):
        stock = {'HZ': hz, 'SZ': sz}
        trajectory = construct(stock)
        validate(trajectory, stock)
        counts[trajectory[-1]['selected_warehouse'] or 'none'] += 1
assert counts == {'HZ': 18, 'SZ': 9, 'none': 9}

bad = {}
def change(label, index, field, value):
    events = copy.deepcopy(FIXTURE)
    events[index][field] = value
    bad[label] = events

change('orphan-result', 1, 'call_id', 'c9')
change('wrong-item', 0, 'arguments', {'sku': 'A18', 'warehouse': 'HZ'})
change('wrong-argument-type', 0, 'arguments', {'sku': 'A17', 'warehouse': 3})
change('unknown-tool', 0, 'name', 'reserve_stock')
change('invented-stock', 1, 'available', 9)
change('boolean-stock', 1, 'available', True)
change('stale-snapshot', 1, 'snapshot', 'stock-v0')
change('duplicate-call-id', 2, 'id', 'c1')
change('failure-as-stock', 1, 'status', 'error')
change('wrong-selection', 4, 'selected_warehouse', 'HZ')
change('false-reservation', 4, 'reserved', True)
change('wrong-result-warehouse', 1, 'warehouse', 'SZ')
bad['missing-result'] = FIXTURE[:1] + FIXTURE[2:]
bad['duplicate-result'] = FIXTURE[:2] + FIXTURE[1:]
bad['premature-final'] = FIXTURE[:2] + FIXTURE[-1:]
bad['action-after-final'] = FIXTURE + FIXTURE[:1]
bad['result-before-call'] = [FIXTURE[1], FIXTURE[0]] + FIXTURE[2:]
for label, events in bad.items():
    try:
        validate(events, {'HZ': 2, 'SZ': 5})
    except AssertionError:
        continue
    raise AssertionError(f'Accepted corruption: {label}')
# Even otherwise genuine results cannot justify an unnecessary second query.
extra = copy.deepcopy(FIXTURE)
extra[1]['available'] = 3
extra[-1]['selected_warehouse'] = 'HZ'
try:
    validate(extra, {'HZ': 3, 'SZ': 5})
except AssertionError:
    pass
else:
    raise AssertionError('Unnecessary query accepted')
# Explicit boundary witness: this validator does NOT grade prose semantics.
ungraded = copy.deepcopy(FIXTURE)
ungraded[-1]['content'] = 'This sentence contradicts the structured result.'
validate(ungraded, {'HZ': 2, 'SZ': 5})
result = dict(article_fixture='passed', stock_pairs=36, selections=counts,
              rejected_corruptions=list(bad) + ['unnecessary-query'],
              natural_language_semantics='not checked', model_invocations=0)
Path(__file__).with_name('trajectory-examples.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))

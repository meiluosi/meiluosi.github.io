"""Event-level supervision checks, not real tokenizer or trainer validation."""
import json
import math
import re
from pathlib import Path
root = Path(__file__).resolve().parents[3]
article = root/'src/content/posts/2026-10-23-llm-training-tool-01-trajectories.md'
events = json.loads(re.search(r'```json\n(.*?)\n```', article.read_text(), re.S)[1])
indices = [i for i, event in enumerate(events) if event['kind'] in ('call', 'final')]
assert indices == [0, 2, 4]
prefixes = [dict(history=events[:i], target=events[i]) for i in indices]
assert prefixes[0]['history'] == []
assert prefixes[1]['history'][-1]['available'] == 2
assert prefixes[2]['history'][-1]['available'] == 5
# Dependency check from original event order, not text matching.
for i, sample in zip(indices, prefixes):
    visible_results = {e['call_id'] for e in sample['history'] if e['kind']=='result'}
    expected_results = {e['id'] for e in events[:i] if e['kind']=='call'}
    assert visible_results == expected_results
    if sample['target']['kind'] == 'call':
        assert sample['target']['id'] not in visible_results
lengths = [4, 6, 10]
means = [0.5, 1.0, 0.2]
# Explicit synthetic token labels, including neutral event delimiters as needed
# in the assumed lengths. These are not actual token IDs from a chat template.
spans = [[(decision, token, means[decision]) for token in range(length)]
         for decision, length in enumerate(lengths)]
once = [x for span in spans for x in span]
repeated = [x for last in range(3) for span in spans[:last+1] for x in span]
counts = [sum(x[0] == d for x in repeated) for d in range(3)]
assert counts == [12,12,10]
assert len(once)==20 and len(repeated)==34
base_mean = sum(x[2] for x in once)/len(once)
repeated_mean = sum(x[2] for x in repeated)/len(repeated)
decision_mean = sum(sum(x[2] for x in span)/len(span) for span in spans)/3
assert math.isclose(base_mean, .5)
assert math.isclose(repeated_mean, 20/34)
assert math.isclose(decision_mean, 17/30)
# A completion-only target appears exactly once across split examples.
assert [x for span in spans for x in span] == once
result = dict(decision_event_indices=indices, target_counts_once=20,
              target_counts_repeated=34, repeated_span_counts=counts,
              token_mean=base_mean, repeated_token_mean=repeated_mean,
              equal_decision_mean=decision_mean,
              illustrative_rates=dict(syntax=.9,semantic=.72,conditional_semantic=72/90,task=.54),
              scope='event prefixes and synthetic labels; no tokenizer or model run')
Path(__file__).with_name('supervision-examples.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

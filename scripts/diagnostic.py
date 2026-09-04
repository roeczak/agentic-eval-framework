import json, glob
from collections import Counter

reasons = Counter()
valid_lost_examples = []

for f in sorted(glob.glob('results/raw/unit/task4/qwen2.5-7b/*.json')):
    d = json.load(open(f))
    po = d.get('parsed_output', {})
    if not po.get('lost'):
        continue
    
    history = d.get('metadata', {}).get('dialogue_history', [])
    if not history:
        reasons['no_history'] += 1
        continue

    # Check ALL turns, not just last
    for turn in history:
        parsed = turn.get('parsed_response')
        node_type = turn.get('node_type', '')
        if parsed is None:
            continue
        decision = parsed.get('decision', '')
        if node_type == 'decision':
            if decision == 'NO':
                reasons['uppercase_NO'] += 1
            elif decision == 'YES':
                reasons['uppercase_YES'] += 1
            elif decision not in ('Yes', 'No'):
                reasons[f'other_invalid: {repr(decision[:20])}'] += 1
            # Valid decision — check if next_step returned None
        
    # For valid_decision_but_lost: check graph for missing edges
    last = history[-1]
    parsed_last = last.get('parsed_response', {})
    if parsed_last and last.get('node_type') == 'decision':
        decision = parsed_last.get('decision', '')
        if decision in ('Yes', 'No'):
            valid_lost_examples.append({
                'scenario': d['scenario_id'],
                'node_id': last.get('node_id'),
                'node_text': last.get('node_text', '')[:60],
                'decision': decision,
                'turns': len(history),
            })

print('Reasons across all turns:')
for r, c in reasons.most_common(20):
    print(f'  {c:>4}  {r}')

print(f'\nValid decision but still lost ({len(valid_lost_examples)} cases):')
for ex in valid_lost_examples[:10]:
    print(f'  {ex}')


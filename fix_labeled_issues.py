"""
Apply improved commentary generator to the labeled issues CSV.
Produces a side-by-side comparison showing all fixes applied.
"""
import sys
import pandas as pd
import json
import re
from pathlib import Path

# Load definitions from generate_commentary.py without running main pipeline
sys.argv = ['generate_commentary.py', '--shard', '.']
src = open('generate_commentary.py').read()
exec(src.split('# Main pipeline')[0])

df = pd.read_csv('c:/Users/tsega/Downloads/matched_commentary_strict_train_issues_labeled.csv')

JSON_BASE = Path('data/json/Soccer_Data_Json')
json_cache = {}


def find_raw_json_local(game_label):
    if game_label in json_cache:
        return json_cache[game_label]
    for split in ['train', 'test', 'valid']:
        split_dir = JSON_BASE / split
        if not split_dir.exists():
            continue
        for league_dir in split_dir.iterdir():
            if not league_dir.is_dir():
                continue
            game_dir = league_dir / game_label
            if game_dir.exists():
                jsons = list(game_dir.glob('*.json'))
                if jsons:
                    with open(jsons[0], encoding='utf-8') as f:
                        json_cache[game_label] = json.load(f)
                        return json_cache[game_label]
    json_cache[game_label] = None
    return None


results = []
entity_hits = 0

for _, row in df.iterrows():
    game_label = row['game_label']
    event_label = row['generated_event_label']
    half = row['generated_half']
    ts = row['generated_time']

    facts = EVENT_FACTS.get(event_label, EVENT_FACTS.get('unknown', {})).copy()

    raw = find_raw_json_local(game_label)
    if raw:
        entities = build_entity_context(raw, event_label, str(ts), int(half))
        facts = format_event_with_entities(facts, entities, event_label)
        if entities.get('player'):
            entity_hits += 1
    else:
        entities = {}

    beg = generate_beginner(facts)
    exp = generate_expert(facts)
    lv = generate_low_vision(facts, None)

    results.append({
        'window_id': row['window_id'],
        'game_label': game_label,
        'event_label': event_label,
        'improved_beginner': beg,
        'improved_expert': exp,
        'improved_low_vision': lv,
        'original_beginner': row['beginner_commentary'],
        'original_expert': row['expert_commentary'],
        'original_low_vision': row['low_vision_commentary'],
        'has_entity': bool(entities.get('player')),
        'orig_has_jargon': row['has_jargon_leak'],
        'orig_is_fallback': row['is_fallback_junk_label'],
    })

rdf = pd.DataFrame(results)

print('=' * 70)
print('BEFORE vs AFTER — matched_commentary_strict_train_issues_labeled.csv')
print('=' * 70)
print(f'Total events: {len(rdf)}')
print(f'Entity detail matched: {entity_hits}/{len(rdf)} ({entity_hits/len(rdf)*100:.1f}%)')
print()

for aud in ['beginner', 'expert', 'low_vision']:
    orig_u = rdf[f'original_{aud}'].nunique()
    impr_u = rdf[f'improved_{aud}'].nunique()
    print(f'{aud:12s}  original: {orig_u:6d} unique  ->  improved: {impr_u:6d} unique  ({impr_u/max(orig_u,1):.1f}x)')

print()
jargon_pat = r'retrieval memory|close example'
orig_jargon = rdf['original_expert'].str.contains(jargon_pat, case=False, na=False).sum()
impr_jargon = rdf['improved_expert'].str.contains(jargon_pat, case=False, na=False).sum()
print(f'Expert jargon leak:  original={orig_jargon} ({orig_jargon/len(rdf)*100:.1f}%)  '
      f'improved={impr_jargon} ({impr_jargon/len(rdf)*100:.0f}%)')

fallback_pat = r'occurs\. In simple terms, this changes the current phase'
orig_fb = rdf['original_beginner'].str.contains(fallback_pat, na=False).sum()
impr_fb = rdf['improved_beginner'].str.contains(fallback_pat, na=False).sum()
print(f'Beginner fallback:   original={orig_fb} ({orig_fb/len(rdf)*100:.1f}%)  '
      f'improved={impr_fb} ({impr_fb/len(rdf)*100:.0f}%)')

grammar_pat = r'\bA injury\b|\bA off_side\b|\ba var occurs\b|\ba unknown occurs\b'
orig_gram = rdf['original_beginner'].str.contains(grammar_pat, na=False).sum()
impr_gram = rdf['improved_beginner'].str.contains(grammar_pat, na=False).sum()
print(f'Grammar errors (a/an): original={orig_gram}  improved={impr_gram}')

rdf.to_csv('c:/Users/tsega/Downloads/improved_commentary_comparison.csv', index=False)
print(f'\nSaved -> c:/Users/tsega/Downloads/improved_commentary_comparison.csv')

# Side-by-side examples
print()
print('=' * 70)
print('SIDE-BY-SIDE EXAMPLES (with entity detail)')
print('=' * 70)

entity_rows = rdf[rdf['has_entity']]
if len(entity_rows) > 0:
    seen_events = set()
    for _, r in entity_rows.iterrows():
        if r['event_label'] in seen_events:
            continue
        seen_events.add(r['event_label'])
        if len(seen_events) > 6:
            break
        print(f'\n  Event: {r["event_label"]}  |  Game: {r["game_label"][:55]}')
        print(f'  BEG ORIG: {str(r["original_beginner"])[:150]}')
        print(f'  BEG IMPR: {str(r["improved_beginner"])[:150]}')
        print(f'  EXP ORIG: {str(r["original_expert"])[:150]}')
        print(f'  EXP IMPR: {str(r["improved_expert"])[:150]}')

# Fallback fix examples
print()
print('--- FALLBACK FIX EXAMPLES ---')
fallback_rows = rdf[rdf['orig_is_fallback'] == True]
if len(fallback_rows) > 0:
    seen = set()
    for _, r in fallback_rows.iterrows():
        if r['event_label'] in seen:
            continue
        seen.add(r['event_label'])
        if len(seen) > 4:
            break
        print(f'\n  Event: {r["event_label"]}')
        print(f'  ORIG: {str(r["original_beginner"])[:150]}')
        print(f'  IMPR: {str(r["improved_beginner"])[:150]}')

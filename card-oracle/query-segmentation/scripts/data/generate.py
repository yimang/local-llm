#!/usr/bin/env python3
"""Generate offline card-search JSONL: python3 scripts/data/generate.py.

Only standard-library Python is required. Defaults: 8000/1000/1000 examples.
Every query has subject, release year, set OR number, and raw OR graded state.
Reprints of the same normalized subject stay together, across games as well.
All 24 permutations of the four required blocks are possible. A game block is
added with probability .3. No misspellings are injected in this first baseline.
Output files are replaced only with --overwrite. No network access is used.
"""
import argparse
import itertools
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = {'SUBJECT', 'YEAR', 'SET', 'CARD_NUMBER', 'GAME', 'GRADER', 'GRADE', 'CONDITION'}


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def read_seeds(directory):
    cards = [json.loads(line) for line in (directory / 'cards.jsonl').read_text().splitlines() if line.strip()]
    attrs = json.loads((directory / 'attributes.json').read_text())
    if not cards:
        raise ValueError('Empty card seed data')
    for card in cards:
        expected_aliases = attrs.get('set_aliases', {}).get(card.get('game'), {}).get(card.get('set'))
        if expected_aliases is not None and card.get('set_aliases') != expected_aliases:
            raise ValueError(f'Set aliases disagree with configured mapping: {card}')
        for key in ('subject', 'game', 'set', 'card_number'):
            if not isinstance(card.get(key), str) or not card[key].strip():
                raise ValueError(f'Invalid {key}: {card}')
        if card['game'] not in attrs['games'] or not normalize(card['subject']):
            raise ValueError(f'Invalid subject/game: {card}')
        if type(card.get('year')) is not int or not 1900 <= card['year'] <= 2100:
            raise ValueError(f'Invalid year: {card}')
        if not isinstance(card.get('set_aliases'), list) or any(not isinstance(a, str) or not a.strip() for a in card['set_aliases']):
            raise ValueError(f'Invalid set aliases: {card}')
    return cards, attrs


def partition(cards, seed):
    # Assign each global subject once; stratify by its sorted game membership.
    groups = defaultdict(list)
    for card in cards:
        groups[normalize(card['subject'])].append(card)
    strata = defaultdict(list)
    for key, values in groups.items():
        strata[tuple(sorted({v['game'] for v in values}))].append(key)
    result = {s: [] for s in ('train', 'val', 'test')}
    rng = random.Random(seed)
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        if len(keys) < 10:
            raise ValueError(f'Too few independent subjects for {stratum}')
        n = max(1, len(keys) // 10)
        for split, selected in zip(result, (keys[2*n:], keys[:n], keys[n:2*n]), strict=True):
            for key in selected:
                result[split].extend(groups[key])
    return result


def render(card, attrs, rng, graded, numbered):
    blocks = [[('SUBJECT', card['subject'])], [('YEAR', str(card['year']))]]
    blocks.append([('CARD_NUMBER', card['card_number'])] if numbered else [('SET', rng.choice(card['set_aliases']))])
    if graded:
        grader = attrs['graders'][rng.choice(sorted(attrs['graders']))]
        blocks.append([('GRADER', rng.choice(grader['aliases'])), ('GRADE', rng.choice(grader['grades']))])
    else:
        condition = attrs['conditions'][rng.choice(sorted(attrs['conditions']))]
        blocks.append([('CONDITION', rng.choice(condition))])
    if rng.random() < .3:
        blocks.append([('GAME', rng.choice(attrs['games'][card['game']]))])
    rng.shuffle(blocks)
    case = rng.choices(['lower', 'original', 'upper'], [70, 20, 10])[0]
    query, spans = '', []
    for label, value in itertools.chain.from_iterable(blocks):
        value = value.lower() if case == 'lower' else value.upper() if case == 'upper' else value
        if query:
            query += ' '
        start = len(query)
        query += value
        spans.append({'start': start, 'end': len(query), 'label': label})
    return {'query': query, 'spans': spans}


def validate(record):
    counts = Counter(s['label'] for s in record['spans'])
    assert set(counts) <= LABELS
    assert counts['SUBJECT'] == counts['YEAR'] == 1
    assert counts['SET'] + counts['CARD_NUMBER'] == 1
    assert (counts['CONDITION'], counts['GRADER'], counts['GRADE']) in [(1, 0, 0), (0, 1, 1)]
    end = 0
    for span in record['spans']:
        assert end <= span['start'] < span['end'] <= len(record['query'])
        assert record['query'][span['start']:span['end']].strip()
        end = span['end']


def generate(cards, attrs, count, rng, seen):
    by_game = defaultdict(lambda: defaultdict(list))
    for card in cards:
        by_game[card['game']][normalize(card['subject'])].append(card)
    games = sorted(by_game)
    # Balance game x graded/raw x set/number, independent of subject selection.
    cells = list(itertools.product(games, (False, True), (False, True)))
    schedule = (cells * (count // len(cells) + 1))[:count]
    rng.shuffle(schedule)
    records = []
    for game, graded, numbered in schedule:
        for _attempt in range(1000):
            subjects = {k: [c for c in v if numbered or c['set_aliases']] for k, v in by_game[game].items()}
            subjects = {k: v for k, v in subjects.items() if v}
            if not subjects:
                raise ValueError(f'No eligible set aliases for {game}; add aliases or seed records')
            card = rng.choice(subjects[rng.choice(sorted(subjects))])
            record = render(card, attrs, rng, graded, numbered)
            key = normalize(record['query'])
            if key not in seen:
                validate(record)
                seen.add(key)
                records.append(record)
                break
        else:
            raise ValueError('Could not generate enough unique queries; add seeds or reduce count')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds-dir', type=Path, default=ROOT / 'data/seeds')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/synthetic')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train', type=int, default=8000)
    parser.add_argument('--val', type=int, default=1000)
    parser.add_argument('--test', type=int, default=1000)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    sizes = {s: getattr(args, s) for s in ('train', 'val', 'test')}
    if min(sizes.values()) < 1:
        parser.error('All split sizes must be positive')
    paths = {s: args.output_dir / f'{s}.jsonl' for s in sizes}
    if not args.overwrite and any(p.exists() for p in paths.values()):
        parser.error('Output exists; use --overwrite to replace splits')
    cards, attrs = read_seeds(args.seeds_dir)
    parts = partition(cards, args.seed)
    seen, outputs = set(), {}
    for i, (split, count) in enumerate(sizes.items()):
        outputs[split] = generate(parts[split], attrs, count, random.Random(args.seed + i + 1), seen)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in outputs.items():
        paths[split].write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        print(f'{split}: {len(records)} queries, {len({normalize(c["subject"]) for c in parts[split]})} subjects')


if __name__ == '__main__':
    main()

# Synthetic card-search data

## Deliverables

Only the following are intended as the data-generation commit:
- data/seeds/cards.jsonl
- data/seeds/attributes.json
- scripts/data/generate.py
- data/synthetic/train.jsonl
- data/synthetic/val.jsonl
- data/synthetic/test.jsonl

This planning document is outside that commit. No upstream URLs, repositories, download manifests, images, or source code are included in the seed or output files.

## Seed schema

cards.jsonl contains one compatible card record per line with subject, game, set, set_aliases, card_number, and integer year. The set field holds the full name; set_aliases contains abbreviated codes only (for example Mega Evolution / MEG and Romance Dawn / OP01). XY is a legitimate code identical to the full set name. An empty alias list restricts the record to card-number queries; full names are never substituted as aliases. Year is the English set release year. All fields are required. Multiple printings can share a subject. Missing release years are not invented.

Riftbound mappings: Origins / OGN, Spiritforged / SFD, Unleashed / UNL, Vendetta / VEN. The latter two have no card records in this baseline. Origins: Proving Grounds has no configured alias and uses card-number queries only.

Initial coverage: 150 sampled Pokémon records, 150 sampled Riftbound records, and 121 distinct Romance Dawn One Piece records. This is a small baseline sample, not an exhaustive catalog. One Piece is limited to Romance Dawn (2022) because additional set release dates were not available in the copied metadata.

attributes.json contains game aliases, grader aliases and numerical grade allowlists, and the five TCGplayer condition categories with query abbreviations. Supported games are Pokemon, Riftbound, and One Piece.

PSA allows integers 1 through 10 only. Beckett/BGS, SGC, and CGC allow half-point steps from 1 through 10. TAG allows half-point steps from 1 through 9, then 10; no 9.5. Only numerical overall grades are generated, without label tiers, legacy descriptors, autograph grades, or TAG scores.

Conditions: Near Mint (NM), Lightly Played (LP), Moderately Played (MP), Heavily Played (HP), Damaged (DMG).

## Required query grammar

Every query contains:
1. Exactly one SUBJECT.
2. Exactly one YEAR.
3. Exactly one SET alias or CARD_NUMBER, never both.
4. Either one CONDITION or a GRADER followed by a valid GRADE, never both modes.

GAME is optional with probability 0.3. All permutations of the required blocks are eligible, with grader/grade kept together. Case sampling is 70% lowercase, 20% original case, 10% uppercase. Aliases are randomly selected. The initial baseline does not add typos, quantities, prices, generic negatives, or incomplete queries.

Games, condition modes, and set-identifier modes are approximately balanced jointly; subject groups are sampled uniformly within game before choosing a printing. Graders and conditions are sampled uniformly, as are valid numerical grades. These experimental sampling choices do not represent actual traffic frequencies.

## Output

One JSON object per line:
{"query":"roronoa zoro psa 10 op01 2022","spans":[{"start":0,"end":12,"label":"SUBJECT"},{"start":13,"end":16,"label":"GRADER"},{"start":17,"end":19,"label":"GRADE"},{"start":20,"end":24,"label":"SET"},{"start":25,"end":29,"label":"YEAR"}]}

Offsets use Python Unicode character indices, zero-based and end-exclusive. Annotations are recorded while rendering. BIO conversion is deferred to training. No provenance metadata is added to queries.

Active span types: SUBJECT, GAME, SET, CARD_NUMBER, YEAR, GRADER, GRADE, CONDITION. Future token classification therefore uses 17 BIO labels including O.

## Splits and reproducibility

Default sizes: 8000 train, 1000 validation, 1000 test. Partition normalized subject names before generation, keeping reprints and case/punctuation variants together across games. Stratify by game membership. About 80/10/10 of subjects belong to each partition. This tests held-out names, not held-out sets or template families; semantic aliases not captured by normalization remain a limitation.

Reject normalized duplicate query strings across all splits. Random seed, stable iteration, and unchanged seed files reproduce byte-identical outputs.

Run:
    python3 scripts/data/generate.py
    python3 scripts/data/generate.py --overwrite

Optional arguments: --seed, --train, --val, --test, --seeds-dir, --output-dir. Python standard library only; no network access. Existing output requires --overwrite.

## Verification

Check every output for required labels and exclusivity, valid grade/condition values, valid offsets, no duplicates, and no normalized subject overlap. Regenerate into a temporary directory and compare bytes. Tiny fixtures or invalid inputs must fail clearly. This synthetic dataset is for pipeline testing; evaluation does not establish performance on natural ecommerce queries. Training and inference implementation are deferred.

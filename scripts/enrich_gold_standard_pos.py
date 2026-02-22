#!/usr/bin/env python3
"""
Enrich the intent gold standard with tokens and POS tags using NLTK.

This script reads the intent gold standard (which has text + intent only)
and adds tokens and pos_tags fields to each example using NLTK's tokenizer
and POS tagger with Universal POS tagset mapping.

Usage:
    python scripts/enrich_gold_standard_pos.py
    python scripts/enrich_gold_standard_pos.py --dry-run
    python scripts/enrich_gold_standard_pos.py --input path/to/gold_standard.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
except ImportError:
    print("Error: NLTK is required. Install with: pip install nltk")
    sys.exit(1)


def ensure_nltk_data():
    """Download required NLTK data if not present."""
    required = ['punkt', 'averaged_perceptron_tagger', 'punkt_tab', 'averaged_perceptron_tagger_eng']
    for package in required:
        try:
            nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'taggers/{package}')
        except LookupError:
            print(f"Downloading NLTK package: {package}")
            nltk.download(package, quiet=True)


# Penn Treebank to Universal POS tag mapping (same as generate_pos_annotations.py)
PTB_TO_UNIVERSAL = {
    'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
    'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
    'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
    'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV', 'WRB': 'ADV',
    'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON', 'WP$': 'PRON',
    'DT': 'DET', 'PDT': 'DET', 'WDT': 'DET',
    'IN': 'ADP', 'TO': 'PART',
    'CC': 'CONJ',
    'CD': 'NUM',
    'RP': 'PART',
    'UH': 'INTJ',
    '.': 'PUNCT', ',': 'PUNCT', ':': 'PUNCT', ';': 'PUNCT',
    '``': 'PUNCT', "''": 'PUNCT', '-LRB-': 'PUNCT', '-RRB-': 'PUNCT',
    'MD': 'AUX',
    'EX': 'PRON',
    'FW': 'X',
    'LS': 'X',
    'POS': 'PART',
    'SYM': 'SYM',
    '$': 'SYM',
    '#': 'SYM',
}


def convert_to_universal_tag(ptb_tag: str) -> str:
    """Convert Penn Treebank tag to Universal POS tag."""
    return PTB_TO_UNIVERSAL.get(ptb_tag, 'X')


def enrich_example(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Add tokens and pos_tags to a gold standard example.

    If the example already has tokens and pos_tags, skip it.
    Returns the enriched example, or None if text is empty.
    """
    text = example.get('text', '').strip()
    if not text:
        return None

    # Skip if already enriched
    if example.get('tokens') and example.get('pos_tags'):
        return example

    # Tokenize and POS tag
    tokens = word_tokenize(text)
    pos_tagged = pos_tag(tokens)
    pos_tags = [convert_to_universal_tag(tag) for _, tag in pos_tagged]

    # Build enriched example, preserving all existing fields
    enriched = dict(example)
    enriched['tokens'] = tokens
    enriched['pos_tags'] = pos_tags

    return enriched


def main():
    parser = argparse.ArgumentParser(
        description='Enrich intent gold standard with tokens and POS tags using NLTK'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('apps/brain/priv/evaluation/intent/gold_standard.json'),
        help='Path to the intent gold standard JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path (defaults to overwriting input file)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )

    args = parser.parse_args()
    output_path = args.output or args.input

    print("Checking NLTK data...")
    ensure_nltk_data()

    # Load gold standard
    if not args.input.exists():
        print(f"Error: Input file does not exist: {args.input}")
        sys.exit(1)

    print(f"Loading gold standard from: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    if not isinstance(examples, list):
        print("Error: Gold standard must be a JSON array")
        sys.exit(1)

    print(f"Total examples: {len(examples)}")

    # Count pre-existing enrichment
    already_enriched = sum(1 for e in examples if e.get('tokens') and e.get('pos_tags'))
    needs_enrichment = len(examples) - already_enriched
    print(f"Already enriched: {already_enriched}")
    print(f"Needs enrichment: {needs_enrichment}")

    if needs_enrichment == 0:
        print("All examples already enriched. Nothing to do.")
        return

    # Enrich
    enriched_examples = []
    skipped = 0
    processed = 0

    for i, example in enumerate(examples):
        enriched = enrich_example(example)
        if enriched:
            enriched_examples.append(enriched)
            if not (example.get('tokens') and example.get('pos_tags')):
                processed += 1
        else:
            skipped += 1

        if args.verbose and (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(examples)}")

    print(f"\nEnriched: {processed}")
    print(f"Skipped (empty text): {skipped}")
    print(f"Total output: {len(enriched_examples)}")

    # Verify a sample
    sample = next((e for e in enriched_examples if e.get('tokens')), None)
    if sample:
        print(f"\nSample enriched entry:")
        print(f"  text: {sample['text']}")
        print(f"  intent: {sample['intent']}")
        print(f"  tokens: {sample['tokens']}")
        print(f"  pos_tags: {sample['pos_tags']}")

    # Collect POS tag distribution
    all_tags = {}
    for e in enriched_examples:
        for tag in e.get('pos_tags', []):
            all_tags[tag] = all_tags.get(tag, 0) + 1

    print(f"\nPOS tag distribution:")
    for tag, count in sorted(all_tags.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

    if args.dry_run:
        print(f"\nDRY RUN - would write to: {output_path}")
    else:
        print(f"\nWriting enriched gold standard to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_examples, f, indent=2, ensure_ascii=False)
        print("Done!")


if __name__ == '__main__':
    main()

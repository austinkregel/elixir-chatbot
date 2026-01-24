#!/usr/bin/env python3
"""
Analyze POS tag distribution and coverage in training data.

Provides insights into:
- Distribution of POS tags
- Underrepresented tag types
- Token examples for each tag
- Suggestions for improving coverage

Usage:
    python scripts/pos_tag_coverage.py [--training-dir data/training/intents]
    python scripts/pos_tag_coverage.py --examples 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter


def load_training_files(training_dir: Path) -> List[Dict[str, Any]]:
    """Load all examples from training files."""
    all_examples = []
    
    for file_path in training_dir.glob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
                if isinstance(examples, list):
                    for ex in examples:
                        ex['_source_file'] = file_path.name
                    all_examples.extend(examples)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return all_examples


def analyze_coverage(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze POS tag coverage across all examples.
    
    Returns coverage statistics.
    """
    tag_counts = Counter()
    tag_tokens = defaultdict(list)  # tag -> list of (token, context) pairs
    tag_by_intent = defaultdict(lambda: Counter())  # intent -> tag -> count
    
    for example in examples:
        tokens = example.get('tokens', [])
        pos_tags = example.get('pos_tags', [])
        intent = example.get('intent', 'unknown')
        text = example.get('text', '')
        
        for i, (token, tag) in enumerate(zip(tokens, pos_tags)):
            tag_counts[tag] += 1
            tag_by_intent[intent][tag] += 1
            
            # Store example tokens (limit per tag)
            if len(tag_tokens[tag]) < 50:
                context = text[:80] + '...' if len(text) > 80 else text
                tag_tokens[tag].append((token, context))
    
    total_tags = sum(tag_counts.values())
    
    return {
        'tag_counts': dict(tag_counts),
        'tag_percentages': {tag: count/total_tags*100 for tag, count in tag_counts.items()},
        'tag_tokens': {tag: tokens[:10] for tag, tokens in tag_tokens.items()},
        'tag_by_intent': {intent: dict(counts) for intent, counts in tag_by_intent.items()},
        'total_tags': total_tags,
        'total_examples': len(examples),
        'unique_tags': len(tag_counts)
    }


def identify_gaps(coverage: Dict[str, Any]) -> List[str]:
    """
    Identify underrepresented POS tags that may need more examples.
    """
    # Expected Universal POS tags
    expected_tags = {
        'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ',
        'NUM', 'PART', 'INTJ', 'PROPN', 'PUNCT', 'AUX', 'SYM', 'X'
    }
    
    tag_counts = coverage['tag_counts']
    total = coverage['total_tags']
    
    gaps = []
    
    # Missing tags
    for tag in expected_tags:
        if tag not in tag_counts:
            gaps.append(f"Missing: {tag} - No examples found")
    
    # Very rare tags (less than 0.1% of total)
    threshold = total * 0.001
    for tag, count in tag_counts.items():
        if count < threshold and count > 0:
            gaps.append(f"Rare: {tag} - Only {count} occurrences ({count/total*100:.2f}%)")
    
    return gaps


def print_coverage_report(coverage: Dict[str, Any], show_examples: int = 3):
    """Print a detailed coverage report."""
    
    print("\n" + "=" * 60)
    print("POS TAG COVERAGE REPORT")
    print("=" * 60)
    
    print(f"\nTotal examples: {coverage['total_examples']}")
    print(f"Total tokens tagged: {coverage['total_tags']}")
    print(f"Unique POS tags: {coverage['unique_tags']}")
    
    # Tag distribution
    print("\n--- Tag Distribution ---")
    sorted_tags = sorted(
        coverage['tag_counts'].items(),
        key=lambda x: -x[1]
    )
    
    for tag, count in sorted_tags:
        pct = coverage['tag_percentages'][tag]
        bar = '#' * int(pct / 2)
        print(f"  {tag:8s} {count:6d} ({pct:5.1f}%) {bar}")
    
    # Example tokens
    if show_examples > 0:
        print(f"\n--- Example Tokens (up to {show_examples} per tag) ---")
        for tag, count in sorted_tags:
            examples = coverage['tag_tokens'].get(tag, [])[:show_examples]
            tokens_str = ', '.join(f"'{t}'" for t, _ in examples)
            print(f"  {tag:8s}: {tokens_str}")
    
    # Coverage gaps
    gaps = identify_gaps(coverage)
    if gaps:
        print("\n--- Coverage Gaps ---")
        for gap in gaps:
            print(f"  * {gap}")
    else:
        print("\n--- Coverage looks good! All major POS tags are represented. ---")
    
    # Intent diversity
    print(f"\n--- Intent Coverage ---")
    print(f"Number of intents: {len(coverage['tag_by_intent'])}")
    
    # Find intents with limited tag diversity
    low_diversity = []
    for intent, tag_counts in coverage['tag_by_intent'].items():
        if len(tag_counts) < 3:
            low_diversity.append((intent, len(tag_counts)))
    
    if low_diversity:
        print("Intents with limited POS diversity:")
        for intent, count in sorted(low_diversity, key=lambda x: x[1])[:10]:
            print(f"  {intent}: only {count} unique tags")
    
    print("\n" + "=" * 60)


def export_coverage_json(coverage: Dict[str, Any], output_path: Path):
    """Export coverage data to JSON for further analysis."""
    # Remove non-serializable parts
    export_data = {
        'tag_counts': coverage['tag_counts'],
        'tag_percentages': coverage['tag_percentages'],
        'total_tags': coverage['total_tags'],
        'total_examples': coverage['total_examples'],
        'unique_tags': coverage['unique_tags'],
        'gaps': identify_gaps(coverage)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Coverage data exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze POS tag distribution and coverage'
    )
    parser.add_argument(
        '--training-dir',
        type=Path,
        default=Path('data/training/intents'),
        help='Directory with enriched training files'
    )
    parser.add_argument(
        '--examples',
        type=int,
        default=3,
        help='Number of example tokens to show per tag'
    )
    parser.add_argument(
        '--export',
        type=Path,
        help='Export coverage data to JSON file'
    )
    
    args = parser.parse_args()
    
    # Load training data
    if not args.training_dir.exists():
        print(f"Error: Training directory does not exist: {args.training_dir}")
        sys.exit(1)
    
    print(f"Loading training data from {args.training_dir}...")
    examples = load_training_files(args.training_dir)
    
    if not examples:
        print("No examples found!")
        sys.exit(1)
    
    print(f"Loaded {len(examples)} examples")
    
    # Analyze coverage
    coverage = analyze_coverage(examples)
    
    # Print report
    print_coverage_report(coverage, args.examples)
    
    # Export if requested
    if args.export:
        export_coverage_json(coverage, args.export)


if __name__ == '__main__':
    main()

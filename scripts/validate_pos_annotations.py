#!/usr/bin/env python3
"""
Validate POS annotations in training data against NLTK predictions.

Compares stored POS tags against fresh NLTK predictions and reports
discrepancies for manual review.

Usage:
    python scripts/validate_pos_annotations.py [--training-dir data/training/intents]
    python scripts/validate_pos_annotations.py --summary
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
except ImportError:
    print("Error: NLTK is required. Install with: pip install nltk")
    sys.exit(1)

# Import tag conversion from generate script
from generate_pos_annotations import convert_to_universal_tag, ensure_nltk_data


def validate_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single example's POS tags against NLTK.
    
    Returns validation result with discrepancies.
    """
    text = example.get('text', '')
    stored_tokens = example.get('tokens', [])
    stored_tags = example.get('pos_tags', [])
    
    if not text or not stored_tokens or not stored_tags:
        return {
            'valid': False,
            'error': 'Missing required fields',
            'text': text
        }
    
    # Re-tokenize and tag with NLTK
    nltk_tokens = word_tokenize(text)
    nltk_tagged = pos_tag(nltk_tokens)
    nltk_tags = [convert_to_universal_tag(tag) for _, tag in nltk_tagged]
    
    # Compare
    discrepancies = []
    
    # Check token alignment
    if stored_tokens != nltk_tokens:
        return {
            'valid': False,
            'error': 'Token mismatch',
            'text': text,
            'stored_tokens': stored_tokens,
            'nltk_tokens': nltk_tokens
        }
    
    # Compare tags position by position
    for i, (stored, nltk) in enumerate(zip(stored_tags, nltk_tags)):
        if stored != nltk:
            discrepancies.append({
                'position': i,
                'token': stored_tokens[i],
                'stored': stored,
                'nltk': nltk
            })
    
    return {
        'valid': len(discrepancies) == 0,
        'text': text,
        'discrepancies': discrepancies,
        'agreement_rate': 1.0 - (len(discrepancies) / len(stored_tags)) if stored_tags else 0.0
    }


def validate_file(file_path: Path) -> Dict[str, Any]:
    """
    Validate all examples in a training file.
    
    Returns aggregated validation results.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return {
            'file': str(file_path),
            'error': str(e),
            'valid': False
        }
    
    if not isinstance(examples, list):
        return {
            'file': str(file_path),
            'error': 'Not a list of examples',
            'valid': False
        }
    
    results = []
    total_agreement = 0.0
    total_discrepancies = 0
    total_tags = 0
    
    for example in examples:
        result = validate_example(example)
        results.append(result)
        
        if 'agreement_rate' in result:
            total_agreement += result['agreement_rate']
            total_discrepancies += len(result.get('discrepancies', []))
            total_tags += len(example.get('pos_tags', []))
    
    num_examples = len(examples)
    
    return {
        'file': str(file_path),
        'valid': all(r.get('valid', False) for r in results),
        'num_examples': num_examples,
        'avg_agreement': total_agreement / num_examples if num_examples else 0.0,
        'total_discrepancies': total_discrepancies,
        'total_tags': total_tags,
        'discrepancy_rate': total_discrepancies / total_tags if total_tags else 0.0,
        'examples_with_issues': [r for r in results if not r.get('valid', False)]
    }


def print_discrepancy_report(results: List[Dict[str, Any]], verbose: bool = False):
    """Print a summary report of validation results."""
    
    print("\n" + "=" * 60)
    print("POS ANNOTATION VALIDATION REPORT")
    print("=" * 60)
    
    total_files = len(results)
    valid_files = sum(1 for r in results if r.get('valid', False))
    total_discrepancies = sum(r.get('total_discrepancies', 0) for r in results)
    total_tags = sum(r.get('total_tags', 0) for r in results)
    
    print(f"\nFiles validated: {total_files}")
    print(f"Files with perfect agreement: {valid_files}")
    print(f"Total tags checked: {total_tags}")
    print(f"Total discrepancies: {total_discrepancies}")
    
    if total_tags:
        print(f"Overall agreement rate: {100 * (1 - total_discrepancies/total_tags):.2f}%")
    
    # Files with issues
    issues = [r for r in results if not r.get('valid', False)]
    if issues:
        print(f"\n--- Files with discrepancies ({len(issues)}) ---")
        for result in sorted(issues, key=lambda x: x.get('discrepancy_rate', 0), reverse=True):
            file_name = Path(result['file']).name
            rate = result.get('discrepancy_rate', 0) * 100
            count = result.get('total_discrepancies', 0)
            print(f"  {file_name}: {count} discrepancies ({rate:.1f}%)")
            
            if verbose and result.get('examples_with_issues'):
                for ex in result['examples_with_issues'][:3]:  # Show first 3
                    if 'discrepancies' in ex:
                        print(f"    Text: {ex['text'][:60]}...")
                        for d in ex['discrepancies'][:3]:
                            print(f"      '{d['token']}': stored={d['stored']}, nltk={d['nltk']}")
    else:
        print("\nAll files have perfect agreement with NLTK!")
    
    print("\n" + "=" * 60)


def analyze_common_discrepancies(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze which tag pairs are most commonly confused."""
    confusion = defaultdict(int)
    
    for result in results:
        for example in result.get('examples_with_issues', []):
            for d in example.get('discrepancies', []):
                pair = (d['stored'], d['nltk'])
                confusion[pair] += 1
    
    return dict(sorted(confusion.items(), key=lambda x: -x[1]))


def main():
    parser = argparse.ArgumentParser(
        description='Validate POS annotations against NLTK predictions'
    )
    parser.add_argument(
        '--training-dir',
        type=Path,
        default=Path('data/training/intents'),
        help='Directory with enriched training files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed discrepancy examples'
    )
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='Show tag confusion matrix'
    )
    
    args = parser.parse_args()
    
    # Ensure NLTK data
    print("Checking NLTK data...")
    ensure_nltk_data()
    
    # Find training files
    if not args.training_dir.exists():
        print(f"Error: Training directory does not exist: {args.training_dir}")
        sys.exit(1)
    
    training_files = list(args.training_dir.glob('*.json'))
    
    if not training_files:
        print(f"No training files found in {args.training_dir}")
        sys.exit(1)
    
    print(f"Found {len(training_files)} training files")
    
    # Validate each file
    results = []
    for file_path in sorted(training_files):
        print(f"Validating: {file_path.name}...", end=' ', flush=True)
        result = validate_file(file_path)
        results.append(result)
        status = "OK" if result.get('valid', False) else f"{result.get('total_discrepancies', '?')} issues"
        print(status)
    
    # Print report
    print_discrepancy_report(results, args.verbose)
    
    # Confusion analysis
    if args.confusion:
        print("\n--- Tag Confusion Analysis ---")
        confusion = analyze_common_discrepancies(results)
        if confusion:
            print("Most common tag disagreements (stored -> nltk):")
            for (stored, nltk), count in list(confusion.items())[:10]:
                print(f"  {stored} -> {nltk}: {count} occurrences")
        else:
            print("No discrepancies to analyze.")


if __name__ == '__main__':
    main()

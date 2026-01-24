#!/usr/bin/env python3
"""
Generate POS annotations for training data using NLTK.

This script reads Dialogflow-format intent files and adds POS (Part-of-Speech)
annotations to each example. The output is written to the data/training/
directory in an enriched format.

Usage:
    python scripts/generate_pos_annotations.py [--input-dir data/intents] [--output-dir data/training/intents]
    python scripts/generate_pos_annotations.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
except ImportError:
    print("Error: NLTK is required. Install with: pip install nltk")
    sys.exit(1)

# Ensure NLTK data is downloaded
def ensure_nltk_data():
    """Download required NLTK data if not present."""
    required = ['punkt', 'averaged_perceptron_tagger', 'universal_tagset', 'punkt_tab', 'averaged_perceptron_tagger_eng']
    for package in required:
        try:
            nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'taggers/{package}')
        except LookupError:
            print(f"Downloading NLTK package: {package}")
            nltk.download(package, quiet=True)

# Penn Treebank to Universal POS tag mapping
PTB_TO_UNIVERSAL = {
    # Nouns
    'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
    # Verbs
    'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
    # Adjectives
    'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
    # Adverbs
    'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV', 'WRB': 'ADV',
    # Pronouns
    'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON', 'WP$': 'PRON',
    # Determiners
    'DT': 'DET', 'PDT': 'DET', 'WDT': 'DET',
    # Prepositions/Adpositions
    'IN': 'ADP', 'TO': 'PART',
    # Conjunctions
    'CC': 'CONJ',
    # Numerals
    'CD': 'NUM',
    # Particles
    'RP': 'PART',
    # Interjections
    'UH': 'INTJ',
    # Punctuation
    '.': 'PUNCT', ',': 'PUNCT', ':': 'PUNCT', ';': 'PUNCT',
    '``': 'PUNCT', "''": 'PUNCT', '-LRB-': 'PUNCT', '-RRB-': 'PUNCT',
    # Other
    'MD': 'AUX',  # Modal verbs
    'EX': 'PRON',  # Existential there
    'FW': 'X',  # Foreign word
    'LS': 'X',  # List item marker
    'POS': 'PART',  # Possessive ending
    'SYM': 'SYM',  # Symbol
    '$': 'SYM',
    '#': 'SYM',
}


def convert_to_universal_tag(ptb_tag: str) -> str:
    """Convert Penn Treebank tag to Universal POS tag."""
    return PTB_TO_UNIVERSAL.get(ptb_tag, 'X')


def extract_text_from_example(example: Dict[str, Any]) -> str:
    """
    Extract the full text from a Dialogflow-format example.
    
    Dialogflow format can have either:
    - A simple "text" field
    - A "data" array with text segments
    """
    # Simple text field
    if 'text' in example and isinstance(example['text'], str):
        return example['text']
    
    # Data array format (Dialogflow annotated)
    if 'data' in example and isinstance(example['data'], list):
        return ''.join(item.get('text', '') for item in example['data'])
    
    return ''


def extract_entities_from_example(example: Dict[str, Any], tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Extract entity annotations from a Dialogflow-format example.
    
    Returns entities with token-based positions (not character positions).
    """
    entities = []
    
    if 'data' not in example:
        return entities
    
    # Track token position
    current_token_idx = 0
    char_to_token = {}  # Map character positions to token indices
    
    # Build character-to-token mapping from original text
    text = extract_text_from_example(example)
    token_idx = 0
    char_idx = 0
    
    for token in tokens:
        # Find token in text starting from char_idx
        while char_idx < len(text) and text[char_idx:char_idx+len(token)] != token:
            char_idx += 1
        
        if char_idx < len(text):
            for i in range(len(token)):
                char_to_token[char_idx + i] = token_idx
            char_idx += len(token)
        token_idx += 1
    
    # Process Dialogflow data segments
    char_pos = 0
    for item in example['data']:
        item_text = item.get('text', '')
        meta = item.get('meta')
        alias = item.get('alias')
        
        if meta and alias:
            # This segment is an entity annotation
            entity_type = meta.replace('@sys.', '').replace('@', '').replace('-', '_')
            
            # Find token range for this entity
            start_char = char_pos
            end_char = char_pos + len(item_text) - 1
            
            start_token = char_to_token.get(start_char, -1)
            end_token = char_to_token.get(end_char, start_token)
            
            if start_token >= 0:
                entities.append({
                    'text': item_text.strip(),
                    'type': entity_type,
                    'start': start_token,
                    'end': end_token
                })
        
        char_pos += len(item_text)
    
    return entities


def annotate_example(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Add POS annotations to a single example.
    
    Returns enriched example or None if text is empty.
    """
    text = extract_text_from_example(example)
    if not text.strip():
        return None
    
    # Tokenize and POS tag
    tokens = word_tokenize(text)
    pos_tagged = pos_tag(tokens)
    
    # Convert to universal tags
    pos_tags = [convert_to_universal_tag(tag) for _, tag in pos_tagged]
    
    # Extract entities with token positions
    entities = extract_entities_from_example(example, tokens)
    
    # Build enriched example
    enriched = {
        'text': text,
        'tokens': tokens,
        'pos_tags': pos_tags,
        'entities': entities,
    }
    
    # Preserve original fields
    if 'id' in example:
        enriched['id'] = example['id']
    
    return enriched


def process_intent_file(input_path: Path, output_path: Path, intent_name: str, dry_run: bool = False) -> Tuple[int, int]:
    """
    Process a single intent file, adding POS annotations.
    
    Returns (processed_count, skipped_count).
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Error reading {input_path}: {e}")
        return 0, 0
    
    if not isinstance(examples, list):
        print(f"  Skipping {input_path}: not a list of examples")
        return 0, 0
    
    enriched_examples = []
    skipped = 0
    
    for example in examples:
        enriched = annotate_example(example)
        if enriched:
            enriched['intent'] = intent_name
            enriched_examples.append(enriched)
        else:
            skipped += 1
    
    if not dry_run and enriched_examples:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_examples, f, indent=2, ensure_ascii=False)
    
    return len(enriched_examples), skipped


def extract_intent_name(filename: str) -> str:
    """Extract intent name from Dialogflow filename."""
    name = filename
    # Remove common suffixes
    for suffix in ['_usersays_en.json', '_usersays.json', '.json']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    
    # Normalize separators
    name = name.replace(' - ', '.').replace(' ', '.')
    
    return name


def main():
    parser = argparse.ArgumentParser(
        description='Generate POS annotations for training data using NLTK'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/intents'),
        help='Input directory with Dialogflow intent files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/training/intents'),
        help='Output directory for enriched intent files'
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
    
    # Ensure NLTK data is available
    print("Checking NLTK data...")
    ensure_nltk_data()
    
    # Find all intent files
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    intent_files = list(args.input_dir.glob('*_usersays_en.json'))
    intent_files.extend(args.input_dir.glob('*_usersays.json'))
    
    if not intent_files:
        print(f"No intent files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(intent_files)} intent files")
    
    if args.dry_run:
        print("DRY RUN - no files will be written")
    
    total_processed = 0
    total_skipped = 0
    
    for input_path in sorted(intent_files):
        intent_name = extract_intent_name(input_path.name)
        output_path = args.output_dir / f"{intent_name}.json"
        
        if args.verbose:
            print(f"Processing: {input_path.name} -> {intent_name}")
        
        processed, skipped = process_intent_file(
            input_path, output_path, intent_name, args.dry_run
        )
        
        total_processed += processed
        total_skipped += skipped
        
        if args.verbose and (processed or skipped):
            print(f"  Processed: {processed}, Skipped: {skipped}")
    
    print(f"\nTotal: {total_processed} examples processed, {total_skipped} skipped")
    
    if not args.dry_run:
        print(f"Output written to: {args.output_dir}")


if __name__ == '__main__':
    main()

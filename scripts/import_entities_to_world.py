#!/usr/bin/env python3
"""
Convert extracted entities from preprocessing into a format suitable for 
importing into a training world's gazetteer overlay.

Usage:
    python import_entities_to_world.py extracted_entities.json output.json

This creates a JSON file that can be loaded using the mix task.
"""

import argparse
import json
import sys
from pathlib import Path


def convert_entities(input_path: Path, output_path: Path):
    """Convert extracted entities to gazetteer format."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    entries = []
    
    # Convert characters to person entities
    for char in data.get('characters', []):
        # Skip generic descriptors
        skip_terms = {'ALL', 'BOTH', 'CROWD', 'CHILDREN', 'VOICE', 'VOICES', 
                      'COMPUTER', 'CAPTAIN', 'COMMANDER', 'LIEUTENANT', 'DOCTOR',
                      'GUARD', 'GUARDS', 'CREWMAN', 'CREWWOMAN', 'MAN', 'WOMAN',
                      'BOY', 'GIRL', 'CHILD', 'SECURITY', 'ENGINEER'}
        
        if char.upper() in skip_terms:
            continue
            
        # Check if it looks like a name (not a number suffix)
        name = char.title()
        
        entries.append({
            'value': name,
            'canonical': name,
            'entity_type': 'person',
            'metadata': {
                'source': 'script_extraction',
                'role': 'character'
            }
        })
    
    # Convert locations to location entities
    for loc in data.get('locations', []):
        # Skip very generic locations
        if len(loc) < 3:
            continue
            
        entries.append({
            'value': loc,
            'canonical': loc,
            'entity_type': 'location',
            'metadata': {
                'source': 'script_extraction',
                'role': 'scene'
            }
        })
    
    # Convert mentioned names (these are less certain, so mark them as candidates)
    for name in data.get('mentioned_names', []):
        # Skip common words that slipped through
        if len(name) < 3:
            continue
            
        entries.append({
            'value': name,
            'canonical': name,
            'entity_type': 'unknown',  # Type needs to be inferred
            'metadata': {
                'source': 'script_extraction',
                'role': 'mentioned',
                'needs_review': True
            }
        })
    
    output = {
        'entities': entries,
        'source_file': str(input_path),
        'total_count': len(entries),
        'by_type': {
            'person': len([e for e in entries if e['entity_type'] == 'person']),
            'location': len([e for e in entries if e['entity_type'] == 'location']),
            'unknown': len([e for e in entries if e['entity_type'] == 'unknown'])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Converted {len(entries)} entities:")
    print(f"  Persons: {output['by_type']['person']}")
    print(f"  Locations: {output['by_type']['location']}")
    print(f"  Unknown (need review): {output['by_type']['unknown']}")
    print(f"\nOutput: {output_path}")
    

def main():
    parser = argparse.ArgumentParser(
        description="Convert extracted entities for import into training world"
    )
    parser.add_argument('input_file', help="Path to extracted_entities.json")
    parser.add_argument('output_file', help="Path for output JSON file")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    convert_entities(input_path, output_path)


if __name__ == '__main__':
    main()

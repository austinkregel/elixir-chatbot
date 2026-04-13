#!/usr/bin/env python3
"""
Preprocess TV/movie scripts for entity discovery.

This script processes transcript-format scripts and:
1. Extracts character names from speaker labels (SPOCK:, PIKE:, etc.)
2. Extracts location names from scene headers [Bridge], [Planet surface], etc.
3. Cleans dialogue text for NLP processing
4. Outputs a cleaned version and a separate entities file

Usage:
    python preprocess_scripts.py input_dir output_dir [--pattern "*.txt"]
    
Example:
    python preprocess_scripts.py data/scripts/TOS data/scripts/TOS_processed
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def extract_speaker_names(text: str) -> Set[str]:
    """Extract character names from speaker labels like 'SPOCK:' or 'PIKE [OC]:'."""
    speakers = set()
    
    # Match patterns like "SPOCK:", "PIKE [OC]:", "BOYCE [OC}:" (note typo in source)
    # Pattern: Start of line or after newline, ALL CAPS word(s), optional bracketed text, colon
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with a speaker label
        # Speaker names are typically ALL CAPS followed by optional [stuff] and :
        if ':' in line:
            before_colon = line.split(':')[0].strip()
            
            # Remove bracketed annotations like [OC], [on monitor]
            name_part = re.sub(r'\s*\[.*?\]\s*', '', before_colon)
            name_part = re.sub(r'\s*\{.*?\}\s*', '', name_part)  # Handle typos like {
            
            # Remove leading/trailing punctuation and whitespace
            name_part = name_part.strip(' \t.,;:!?()[]{}"\'-')
            
            # Check if it's all uppercase (speaker label)
            if name_part and name_part.isupper() and len(name_part) > 1:
                # Skip if it's just numbers or has weird characters
                if not re.match(r'^[A-Z][A-Z0-9\s]+$', name_part):
                    continue
                    
                # Could be multi-word like "NUMBER ONE" or single like "SPOCK"
                speakers.add(name_part.strip())
    
    return speakers


def extract_scene_locations(text: str) -> Set[str]:
    """Extract location names from scene headers like [Bridge] or [Planet surface]."""
    locations = set()
    
    # Match [Location Name] at the start of lines
    pattern = r'^\[([^\]]+)\]'
    
    for line in text.split('\n'):
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            location = match.group(1).strip()
            # Filter out things that aren't locations (stage directions embedded in brackets)
            # Locations are typically short and don't contain verbs
            if len(location) < 50 and not any(word in location.lower() for word in ['enters', 'exits', 'walks', 'runs']):
                locations.add(location)
    
    return locations


def extract_mentioned_names(text: str, known_speakers: Set[str]) -> Set[str]:
    """Extract proper nouns mentioned in dialogue (names not as speaker labels)."""
    mentioned = set()
    
    # Look for capitalized words that might be names
    # This is a heuristic - the ML system will validate these
    words = text.split()
    
    for i, word in enumerate(words):
        # Clean punctuation
        clean_word = re.sub(r'[^\w]', '', word)
        
        if not clean_word:
            continue
            
        # Check if it's a capitalized word (potential proper noun)
        if clean_word[0].isupper() and len(clean_word) > 1:
            # Skip if it's a known speaker (we already have those)
            if clean_word.upper() in known_speakers:
                continue
                
            # Skip common words that are often capitalized
            skip_words = {'The', 'This', 'That', 'What', 'When', 'Where', 'Why', 'How',
                         'Yes', 'No', 'Oh', 'Well', 'Now', 'Then', 'Here', 'There',
                         'Captain', 'Doctor', 'Mister', 'Sir', 'Lieutenant', 'Commander',
                         'Enterprise', 'Federation'}  # Keep Enterprise as it's a ship name
            
            if clean_word in skip_words:
                continue
                
            # Skip if preceded by start of sentence indicators
            if i > 0:
                prev_word = words[i-1]
                if prev_word.endswith(('.', '?', '!', ':')):
                    # Could be start of sentence, be cautious
                    # Only add if it looks like a name (not too long, no numbers)
                    if len(clean_word) < 15 and not any(c.isdigit() for c in clean_word):
                        mentioned.add(clean_word)
                else:
                    # Mid-sentence capitalization = likely proper noun
                    if len(clean_word) < 15 and not any(c.isdigit() for c in clean_word):
                        mentioned.add(clean_word)
    
    return mentioned


def clean_dialogue(text: str) -> str:
    """Clean script text for NLP processing while preserving entity context."""
    lines = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Skip episode metadata lines
        if line.startswith('Episode Number:') or line.startswith('The Star Trek Transcripts'):
            continue
            
        # Convert scene headers to narrative
        if line.startswith('[') and line.endswith(']'):
            location = line[1:-1]
            lines.append(f"Scene: {location}.")
            continue
            
        # Handle speaker lines
        if ':' in line:
            parts = line.split(':', 1)
            speaker_part = parts[0].strip()
            
            # Remove bracketed annotations from speaker
            speaker_clean = re.sub(r'\s*\[.*?\]\s*', '', speaker_part)
            speaker_clean = re.sub(r'\s*\{.*?\}\s*', '', speaker_clean)
            
            if speaker_clean.isupper() and len(speaker_clean) > 1:
                # It's a speaker label
                dialogue = parts[1].strip() if len(parts) > 1 else ""
                
                # Convert to narrative form for better NLP processing
                # "SPOCK: Hello" -> "Spock said: Hello"
                speaker_name = speaker_clean.title()
                if dialogue:
                    lines.append(f"{speaker_name} said: {dialogue}")
                continue
        
        # Handle stage directions - convert to narrative
        # (Kirk enters) -> Kirk enters
        cleaned_line = line
        # Remove parentheses from stage directions
        cleaned_line = re.sub(r'\(([^)]+)\)', r'\1', cleaned_line)
        
        if cleaned_line.strip():
            lines.append(cleaned_line)
    
    return '\n'.join(lines)


def process_script(input_path: Path) -> Tuple[str, Dict[str, List[str]]]:
    """Process a single script file and return cleaned text + extracted entities."""
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Extract entities
    speakers = extract_speaker_names(text)
    locations = extract_scene_locations(text)
    mentioned = extract_mentioned_names(text, speakers)
    
    # Clean the text
    cleaned = clean_dialogue(text)
    
    entities = {
        'characters': sorted(list(speakers)),
        'locations': sorted(list(locations)),
        'mentioned_names': sorted(list(mentioned)),
        'source_file': str(input_path.name)
    }
    
    return cleaned, entities


def process_directory(input_dir: Path, output_dir: Path, pattern: str = "*.txt") -> Dict:
    """Process all scripts in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    cleaned_dir = output_dir / "cleaned"
    cleaned_dir.mkdir(exist_ok=True)
    
    all_entities = {
        'characters': set(),
        'locations': set(),
        'mentioned_names': set()
    }
    
    file_entities = []
    processed_count = 0
    
    for input_path in sorted(input_dir.glob(pattern)):
        print(f"Processing: {input_path.name}")
        
        try:
            cleaned, entities = process_script(input_path)
            
            # Save cleaned text
            output_path = cleaned_dir / input_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            # Accumulate entities
            all_entities['characters'].update(entities['characters'])
            all_entities['locations'].update(entities['locations'])
            all_entities['mentioned_names'].update(entities['mentioned_names'])
            
            file_entities.append(entities)
            processed_count += 1
            
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
    
    # Save consolidated entities
    consolidated = {
        'characters': sorted(list(all_entities['characters'])),
        'locations': sorted(list(all_entities['locations'])),
        'mentioned_names': sorted(list(all_entities['mentioned_names'])),
        'total_files_processed': processed_count
    }
    
    entities_path = output_dir / "extracted_entities.json"
    with open(entities_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, indent=2)
    
    # Save per-file entities
    per_file_path = output_dir / "entities_by_file.json"
    with open(per_file_path, 'w', encoding='utf-8') as f:
        json.dump(file_entities, f, indent=2)
    
    return consolidated


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess TV/movie scripts for entity discovery"
    )
    parser.add_argument('input_dir', help="Directory containing script files")
    parser.add_argument('output_dir', help="Directory for processed output")
    parser.add_argument('--pattern', default="*.txt", help="File pattern to match (default: *.txt)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing scripts from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Pattern: {args.pattern}")
    print()
    
    result = process_directory(input_dir, output_dir, args.pattern)
    
    print()
    print("=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print(f"Files processed: {result['total_files_processed']}")
    print(f"Characters found: {len(result['characters'])}")
    print(f"Locations found: {len(result['locations'])}")
    print(f"Other names mentioned: {len(result['mentioned_names'])}")
    print()
    print("Sample characters:", result['characters'][:10])
    print("Sample locations:", result['locations'][:10])
    print()
    print(f"Cleaned scripts saved to: {output_dir / 'cleaned'}")
    print(f"Entities saved to: {output_dir / 'extracted_entities.json'}")


if __name__ == '__main__':
    main()

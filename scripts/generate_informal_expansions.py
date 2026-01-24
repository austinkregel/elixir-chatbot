#!/usr/bin/env python3
"""
Generate informal speech expansions dataset for the ChatBot tokenizer.

This script uses phonetic analysis to build a comprehensive mapping of
informal/colloquial English forms to their standard equivalents.

Usage:
    python scripts/generate_informal_expansions.py

Output:
    data/informal_expansions.json
"""

import json
import os
from pathlib import Path
from typing import Optional

# Try to import phonemizer for IPA conversion
try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    HAS_PHONEMIZER = True
except ImportError:
    HAS_PHONEMIZER = False
    print("Warning: phonemizer not installed. Install with: pip install espeak-phonemizer")
    print("Phonetic validation will be skipped.")


def get_ipa(text: str) -> Optional[str]:
    """Convert text to IPA phonemes using espeak."""
    if not HAS_PHONEMIZER:
        return None
    try:
        return phonemize(
            text,
            language='en-us',
            backend='espeak',
            strip=True,
            preserve_punctuation=False
        )
    except Exception as e:
        print(f"Warning: Could not phonemize '{text}': {e}")
        return None


def phonetic_similarity(ipa1: Optional[str], ipa2: Optional[str]) -> float:
    """
    Calculate similarity between two IPA strings.
    Returns 1.0 for identical, 0.0 for completely different.
    """
    if not ipa1 or not ipa2:
        return 0.5  # Unknown similarity
    
    # Simple character-level similarity
    # Could be improved with edit distance or phoneme-aware comparison
    set1 = set(ipa1.replace(' ', ''))
    set2 = set(ipa2.replace(' ', ''))
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


# =============================================================================
# Core expansion mappings
# =============================================================================

# Contractions with apostrophes (standard)
APOSTROPHE_CONTRACTIONS = {
    # Pronoun + be
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "there's": "there is",
    "here's": "here is",
    
    # Pronoun + will
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "that'll": "that will",
    
    # Pronoun + would/had
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would",
    "he'd": "he would",
    "she'd": "she would",
    "it'd": "it would",
    
    # Pronoun + have
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "could've": "could have",
    "would've": "would have",
    "should've": "should have",
    "might've": "might have",
    "must've": "must have",
    
    # Negations
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "can't": "cannot",
    "cannot": "can not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    
    # Other common
    "let's": "let us",
    "how's": "how is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
}

# Informal phonetic reductions (no apostrophe)
PHONETIC_REDUCTIONS = {
    # Modal + to reductions
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "hafta": "have to",
    "hasta": "has to",
    "oughta": "ought to",
    "needa": "need to",
    "useta": "used to",
    "supposta": "supposed to",
    "sposta": "supposed to",
    
    # Modal + have reductions  
    "coulda": "could have",
    "woulda": "would have",
    "shoulda": "should have",
    "mighta": "might have",
    "musta": "must have",
    
    # "of" reductions (often confused with above)
    "kinda": "kind of",
    "sorta": "sort of",
    "lotta": "lot of",
    "outta": "out of",
    "coupla": "couple of",
    "buncha": "bunch of",
    "loadsa": "loads of",
    "lotsa": "lots of",
    "alla": "all of",
    
    # "let me" reductions
    "lemme": "let me",
    "gimme": "give me",
    
    # "I am" / "I'm going to" reductions
    "imma": "i am going to",
    "ima": "i am going to",
    
    # "you" reductions
    "ya": "you",
    "yer": "your",
    "yall": "you all",
    "y'all": "you all",
    
    # "them/the" reductions
    "em": "them",
    "'em": "them",
    "da": "the",
    
    # Common word shortenings
    "dunno": "do not know",
    "donno": "do not know",
    "innit": "is it not",
    "init": "is it not",
    "aint": "am not",
    "ain't": "am not",
    "aight": "alright",
    "alright": "all right",
    
    # Casual greetings/responses
    "howdy": "how do you do",
    "sup": "what is up",
    "wassup": "what is up",
    "whatup": "what is up",
    "whassup": "what is up",
    
    # Regional/dialectal
    "finna": "fixing to",
    "fitna": "fixing to",
    "tryna": "trying to",
    "boutta": "about to",
    "abouta": "about to",
}

# Consonant coalescence patterns (rapid speech)
# These involve phonetic merging across word boundaries
COALESCENCE_PATTERNS = {
    # did + you -> /dʒ/
    "didja": "did you",
    "didya": "did you",
    "dija": "did you",
    
    # would + you -> /dʒ/
    "wouldja": "would you",
    "wouldya": "would you",
    
    # could + you -> /dʒ/
    "couldja": "could you",
    "couldya": "could you",
    
    # should + you -> /dʒ/
    "shouldja": "should you",
    
    # do + you
    "doya": "do you",
    "d'ya": "do you",
    "d'you": "do you",
    
    # don't + you
    "dontcha": "do you not",
    "doncha": "do you not",
    
    # got + you -> /tʃ/
    "gotcha": "got you",
    "gotya": "got you",
    
    # bet + you -> /tʃ/
    "betcha": "bet you",
    
    # what + you / what + are + you
    "whatcha": "what are you",
    "watcha": "what are you",
    "whaddya": "what do you",
    "whaddaya": "what do you",
    "whadya": "what do you",
    
    # where + did + you
    "wheredja": "where did you",
    
    # how + did + you
    "howdja": "how did you",
    
    # meet + you
    "meetcha": "meet you",
    
    # let + you
    "letcha": "let you",
    
    # get + you
    "getcha": "get you",
    
    # catch + you
    "catchya": "catch you",
}

# Word-initial elision (vowel dropping)
# Common in very casual speech
ELISION_PATTERNS = {
    # 'cause / 'cos
    "cause": "because",
    "'cause": "because",
    "cuz": "because",
    "coz": "because",
    "'coz": "because",
    "cos": "because",
    "'cos": "because",
    
    # 'bout
    "bout": "about",
    "'bout": "about",
    
    # 'til / till
    "til": "until",
    "'til": "until",
    
    # 'round
    "round": "around",  # Note: "round" can also be standalone
    "'round": "around",
    
    # 'fore
    "fore": "before",
    "'fore": "before",
    
    # 'nother
    "nother": "another",
    "'nother": "another",
}

# Question compressions (very informal/dialectal)
QUESTION_COMPRESSIONS = {
    # "did you eat" -> jeet
    "jeet": "did you eat",
    
    # "did you ever" -> jever  
    "jever": "did you ever",
    
    # "did you" variations
    "jew": "did you",
    "ju": "did you",
    
    # "do you know" 
    "dyaknow": "do you know",
    "yaknow": "you know",
    "yanno": "you know",
    "yknow": "you know",
    "y'know": "you know",
    
    # "what do you mean"
    "whadyamean": "what do you mean",
    
    # "are you"
    "arya": "are you",
    "areya": "are you",
    
    # "have you"
    "havya": "have you",
}

# Internet/texting abbreviations (optional - can be enabled/disabled)
INTERNET_ABBREVIATIONS = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "b4": "before",
    "2": "to",
    "4": "for",
    "bc": "because",
    "w/": "with",
    "w/o": "without",
    "thru": "through",
    "tho": "though",
    "altho": "although",
    "nite": "night",
    "lite": "light",
    "rite": "right",
    "tonite": "tonight",
    "thanx": "thanks",
    "thx": "thanks",
    "pls": "please",
    "plz": "please",
    "msg": "message",
    "pic": "picture",
    "pics": "pictures",
    "info": "information",
}


def build_dataset(
    include_internet_abbrevs: bool = False,
    validate_phonetics: bool = True
) -> dict:
    """
    Build the complete informal expansions dataset.
    
    Args:
        include_internet_abbrevs: Whether to include internet/texting abbreviations
        validate_phonetics: Whether to validate phonetic similarity (requires espeak)
    
    Returns:
        Dictionary with expansions and metadata
    """
    all_expansions = {}
    
    # Merge all categories
    categories = [
        ("apostrophe_contractions", APOSTROPHE_CONTRACTIONS),
        ("phonetic_reductions", PHONETIC_REDUCTIONS),
        ("coalescence_patterns", COALESCENCE_PATTERNS),
        ("elision_patterns", ELISION_PATTERNS),
        ("question_compressions", QUESTION_COMPRESSIONS),
    ]
    
    if include_internet_abbrevs:
        categories.append(("internet_abbreviations", INTERNET_ABBREVIATIONS))
    
    entries = []
    
    for category, mappings in categories:
        for informal, formal in mappings.items():
            entry = {
                "informal": informal.lower(),
                "formal": formal.lower(),
                "category": category,
            }
            
            # Optionally validate with phonetics
            if validate_phonetics and HAS_PHONEMIZER:
                informal_ipa = get_ipa(informal)
                formal_ipa = get_ipa(formal)
                
                if informal_ipa:
                    entry["informal_ipa"] = informal_ipa
                if formal_ipa:
                    entry["formal_ipa"] = formal_ipa
                
                similarity = phonetic_similarity(informal_ipa, formal_ipa)
                entry["phonetic_similarity"] = round(similarity, 3)
            
            entries.append(entry)
            
            # Also add to flat lookup
            all_expansions[informal.lower()] = formal.lower()
    
    return {
        "version": "1.0.0",
        "description": "Informal English to standard English expansions",
        "generated_by": "scripts/generate_informal_expansions.py",
        "include_internet_abbrevs": include_internet_abbrevs,
        "phonetic_validation": validate_phonetics and HAS_PHONEMIZER,
        "total_entries": len(entries),
        "expansions": all_expansions,  # Flat lookup for runtime
        "entries_with_metadata": entries,  # Full metadata for analysis
    }


def main():
    """Generate the dataset and save to JSON."""
    # Determine output path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_path = project_root / "data" / "informal_expansions.json"
    
    print("Generating informal expansions dataset...")
    print(f"Output: {output_path}")
    
    if HAS_PHONEMIZER:
        print("Phonemizer available - will include IPA validation")
    else:
        print("Phonemizer not available - skipping IPA validation")
    
    # Build dataset
    dataset = build_dataset(
        include_internet_abbrevs=False,  # Keep it focused on speech patterns
        validate_phonetics=True
    )
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {dataset['total_entries']} expansion entries")
    print(f"Saved to: {output_path}")
    
    # Print some stats
    print("\nCategories:")
    category_counts = {}
    for entry in dataset['entries_with_metadata']:
        cat = entry['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()

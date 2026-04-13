#!/usr/bin/env python3
"""
Generate seeded heuristics from intent training data.

This script analyzes data/intents/*_usersays_en.json files to extract
common patterns that can be used for fast-path intent matching.

Patterns are extracted based on:
- First words that strongly indicate an intent (greetings: hi, hello, hey)
- Keywords that correlate with specific intents (weather, forecast)
- Phrase patterns (what's the weather, turn on, remind me)

Output: data/heuristics/seeded_heuristics.json (source of truth)
The HeuristicStore loads directly from this file.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import re

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
INTENTS_DIR = DATA_DIR / "intents"
OUTPUT_DIR = DATA_DIR / "heuristics"
OUTPUT_FILE = OUTPUT_DIR / "seeded_heuristics.json"

# Intent mappings: simplified intent -> canonical template intent
# This ensures heuristics use the same intent names as TemplateStore
INTENT_MAPPINGS = {
    "smalltalk.greetings.hello": "smalltalk.greetings.hello",
    "smalltalk.greetings.bye": "smalltalk.greetings.bye",
    "smalltalk.greetings.goodmorning": "smalltalk.greetings.goodmorning",
    "smalltalk.greetings.goodnight": "smalltalk.greetings.goodnight",
    "smalltalk.greetings.how_are_you": "smalltalk.greetings.how_are_you",
    "smalltalk.appraisal.thank_you": "smalltalk.appraisal.thank_you",
    "weather": "weather.query",
    "music.play": "music.play",
    "reminder.create": "reminder.create",
    "smarthome.lights.switch.on": "smarthome.lights.switch.on",
    "smarthome.lights.switch.off": "smarthome.lights.switch.off",
}


def load_usersays_files():
    """Load all *_usersays_en.json files from data/intents."""
    intent_examples = defaultdict(list)
    
    for filepath in INTENTS_DIR.glob("*_usersays_en.json"):
        # Extract intent name from filename
        # e.g., "smalltalk.greetings.hello_usersays_en.json" -> "smalltalk.greetings.hello"
        intent_name = filepath.stem.replace("_usersays_en", "")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                examples = json.load(f)
                
            for example in examples:
                # Extract full text from data segments
                text_parts = []
                for part in example.get("data", []):
                    text_parts.append(part.get("text", ""))
                
                full_text = "".join(text_parts).strip().lower()
                if full_text:
                    intent_examples[intent_name].append(full_text)
                    
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}", file=sys.stderr)
            
    return intent_examples


def extract_first_words(examples, min_count=3):
    """Extract first words that commonly appear in examples."""
    first_word_counts = defaultdict(int)
    
    for text in examples:
        words = text.split()
        if words:
            first_word = re.sub(r'[^\w]', '', words[0])  # Remove punctuation
            if first_word:
                first_word_counts[first_word] += 1
                
    # Return words that appear at least min_count times
    return [word for word, count in first_word_counts.items() 
            if count >= min_count and len(word) > 1]


def extract_keywords(examples, min_count=2):
    """Extract keywords that commonly appear across examples."""
    # Common stop words to ignore
    stop_words = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'can', 'you', 
                  'me', 'my', 'i', 'to', 'in', 'for', 'of', 'and', 'or', 'it',
                  'be', 'do', 'like', 'going', 'will', 'would', 'could', 'tell'}
    
    keyword_counts = defaultdict(int)
    
    for text in examples:
        words = set(re.findall(r'\w+', text.lower()))
        for word in words:
            if word not in stop_words and len(word) > 2:
                keyword_counts[word] += 1
                
    # Return words that appear in at least min_count examples
    return [word for word, count in keyword_counts.items() 
            if count >= min_count]


def extract_phrases(examples, min_count=2):
    """Extract common 2-3 word phrases."""
    phrase_counts = defaultdict(int)
    
    for text in examples:
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        words = text_clean.split()
        
        # Extract 2-word and 3-word phrases
        for i in range(len(words) - 1):
            phrase2 = ' '.join(words[i:i+2])
            phrase_counts[phrase2] += 1
            
            if i < len(words) - 2:
                phrase3 = ' '.join(words[i:i+3])
                phrase_counts[phrase3] += 1
                
    # Return phrases that appear at least min_count times
    return [phrase for phrase, count in phrase_counts.items() 
            if count >= min_count]


def get_canonical_intent(intent_name):
    """Get canonical intent name for templates."""
    # Direct mapping if available
    if intent_name in INTENT_MAPPINGS:
        return INTENT_MAPPINGS[intent_name]
    
    # Keep as-is if it's already a full intent name
    if intent_name.count('.') >= 1:
        return intent_name
        
    return None


def generate_heuristics(intent_examples):
    """Generate heuristics from intent examples."""
    heuristics = []
    
    # Define high-confidence patterns for specific intents
    # These are manually curated for reliability
    
    # Greetings
    greeting_intents = [name for name in intent_examples.keys() 
                        if 'greeting' in name.lower() and 'hello' in name.lower()]
    if greeting_intents:
        canonical = get_canonical_intent(greeting_intents[0])
        if canonical:
            # Simple greeting words
            heuristics.append({
                "id": "greeting_simple",
                "pattern": {
                    "first_word": ["hi", "hello", "hey", "howdy", "hiya", "heya", "greetings"],
                    "word_count": [1, 5]
                },
                "conclusion": {
                    "intent": canonical,
                    "confidence_boost": 0.4
                }
            })
    
    # Good morning patterns
    morning_intents = [name for name in intent_examples.keys() 
                       if 'goodmorning' in name.lower()]
    if morning_intents:
        canonical = get_canonical_intent(morning_intents[0]) or "smalltalk.greetings.goodmorning"
        heuristics.append({
            "id": "greeting_good_morning",
            "pattern": {
                "phrase": "good morning"
            },
            "conclusion": {
                "intent": canonical,
                "confidence_boost": 0.4
            }
        })
    
    # Farewell patterns
    bye_intents = [name for name in intent_examples.keys() 
                   if 'bye' in name.lower()]
    if bye_intents:
        canonical = get_canonical_intent(bye_intents[0]) or "smalltalk.greetings.bye"
        heuristics.append({
            "id": "farewell_bye",
            "pattern": {
                "first_word": ["bye", "goodbye", "farewell", "cya", "later", "goodnight"],
                "word_count": [1, 4]
            },
            "conclusion": {
                "intent": canonical,
                "confidence_boost": 0.4
            }
        })
    
    # Weather patterns
    weather_intents = [name for name in intent_examples.keys() 
                       if 'weather' in name.lower()]
    if weather_intents:
        # Use first weather intent or default
        canonical = "weather.query"
        
        heuristics.append({
            "id": "weather_keyword",
            "pattern": {
                "keywords": ["weather", "forecast", "temperature"]
            },
            "conclusion": {
                "intent": canonical,
                "confidence_boost": 0.35
            }
        })
        
        heuristics.append({
            "id": "weather_whats_pattern",
            "pattern": {
                "phrase": "what's the weather"
            },
            "conclusion": {
                "intent": canonical,
                "confidence_boost": 0.4
            }
        })
        
        heuristics.append({
            "id": "weather_hows_pattern",
            "pattern": {
                "phrase": "how's the weather"
            },
            "conclusion": {
                "intent": canonical,
                "confidence_boost": 0.4
            }
        })
    
    # Thanks patterns
    thanks_intents = [name for name in intent_examples.keys() 
                      if 'thank' in name.lower()]
    if thanks_intents:
        canonical = get_canonical_intent("smalltalk.appraisal.thank_you")
        heuristics.append({
            "id": "thanks_pattern",
            "pattern": {
                "keywords": ["thanks", "thank you", "appreciate"]
            },
            "conclusion": {
                "intent": canonical,
                "confidence_boost": 0.4
            }
        })
    
    # Device control patterns
    device_intents = [name for name in intent_examples.keys() 
                      if 'smarthome' in name.lower() or 'device' in name.lower()]
    if device_intents:
        heuristics.append({
            "id": "device_turn_on",
            "pattern": {
                "phrase": "turn on"
            },
            "conclusion": {
                "intent": "device.control",
                "confidence_boost": 0.35
            }
        })
        
        heuristics.append({
            "id": "device_turn_off",
            "pattern": {
                "phrase": "turn off"
            },
            "conclusion": {
                "intent": "device.control",
                "confidence_boost": 0.35
            }
        })
    
    # Music patterns
    music_intents = [name for name in intent_examples.keys() 
                     if 'music' in name.lower()]
    if music_intents:
        heuristics.append({
            "id": "music_play",
            "pattern": {
                "first_word": ["play"],
                "keywords": ["music", "song", "album", "playlist"]
            },
            "conclusion": {
                "intent": "music.play",
                "confidence_boost": 0.35
            }
        })
    
    # Reminder patterns
    reminder_intents = [name for name in intent_examples.keys() 
                        if 'reminder' in name.lower()]
    if reminder_intents:
        heuristics.append({
            "id": "reminder_create",
            "pattern": {
                "phrase": "remind me"
            },
            "conclusion": {
                "intent": "reminder.create",
                "confidence_boost": 0.35
            }
        })
    
    # Timer patterns
    timer_intents = [name for name in intent_examples.keys() 
                     if 'timer' in name.lower()]
    if timer_intents:
        heuristics.append({
            "id": "timer_set",
            "pattern": {
                "keywords": ["timer", "set a timer", "set timer"]
            },
            "conclusion": {
                "intent": "timer.set",
                "confidence_boost": 0.35
            }
        })
    
    return heuristics


def main():
    """Main entry point."""
    print("Loading intent training data...")
    intent_examples = load_usersays_files()
    print(f"Loaded {len(intent_examples)} intents with training data")
    
    print("\nGenerating heuristics...")
    heuristics = generate_heuristics(intent_examples)
    print(f"Generated {len(heuristics)} heuristics")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write output
    print(f"\nWriting to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(heuristics, f, indent=2)
    
    print("Done!")
    
    # Print summary
    print("\nGenerated heuristics:")
    for h in heuristics:
        print(f"  - {h['id']}: {h['conclusion']['intent']}")


if __name__ == "__main__":
    main()

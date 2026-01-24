#!/usr/bin/env python3
"""
Clean repository script: Remove markdown files, plain text files, and comments from Elixir code.

This script:
1. Removes all .md files (except in data/ directory)
2. Removes all plain text files (.txt, .po, .pot, etc.)
3. Removes comments from Elixir code files (.ex, .exs, .heex)
4. Keeps only the data/ directory and raw Elixir code

Usage:
    python scripts/clean_repository.py [--dry-run] [--backup]
"""

import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Set

# File extensions to remove
TEXT_FILE_EXTENSIONS = {'.txt', '.po', '.pot'}
MARKDOWN_EXTENSIONS = {'.md'}
OTHER_NON_CODE_EXTENSIONS = {
    '.py',  # Python scripts
    '.json',  # JSON files (except in data/)
    '.csv',  # CSV files (except in data/)
    '.term',  # Term files
    '.beam',  # Compiled beam files
    '.ez',  # Archive files
    '.lock',  # Lock files (but we'll keep mix.lock as it's needed)
}

# File extensions to keep (Elixir code)
ELIXIR_EXTENSIONS = {'.ex', '.exs', '.heex'}

# Files to preserve even if they match removal patterns
PRESERVE_FILES = {
    'mix.lock',  # Needed for dependency resolution
    '.gitignore',  # Git configuration
    '.cursorrules',  # Cursor rules
    '.formatter.exs',  # Elixir formatter config (but comments will be removed)
}

# Directories to preserve entirely
PRESERVE_DIRECTORIES = {'data'}

# Directories to skip (build artifacts, dependencies, etc.)
SKIP_DIRECTORIES = {
    '_build', 'deps', 'cover', 'doc', '.git', 'node_modules',
    'priv/static', '.elixir_ls', 'tmp'
}


def is_in_preserved_directory(file_path: Path, repo_root: Path) -> bool:
    """Check if file is in a preserved directory."""
    rel_path = file_path.relative_to(repo_root)
    parts = rel_path.parts
    return parts[0] in PRESERVE_DIRECTORIES


def should_skip_directory(dir_path: Path, repo_root: Path) -> bool:
    """Check if directory should be skipped."""
    rel_path = dir_path.relative_to(repo_root)
    parts = rel_path.parts
    return any(part in SKIP_DIRECTORIES for part in parts)


def find_files_to_remove(repo_root: Path) -> tuple[List[Path], List[Path], List[Path]]:
    """Find all files that should be removed."""
    markdown_files = []
    text_files = []
    other_files = []
    
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        
        # Skip certain directories
        dirs[:] = [d for d in dirs if not should_skip_directory(root_path / d, repo_root)]
        
        for file in files:
            file_path = root_path / file
            
            # Skip files in preserved directories
            if is_in_preserved_directory(file_path, repo_root):
                continue
            
            # Skip the script itself
            if file_path.name == 'clean_repository.py':
                continue
            
            # Skip files that should be preserved
            if file_path.name in PRESERVE_FILES:
                continue
            
            ext = file_path.suffix.lower()
            
            if ext in MARKDOWN_EXTENSIONS:
                markdown_files.append(file_path)
            elif ext in TEXT_FILE_EXTENSIONS:
                text_files.append(file_path)
            elif ext in OTHER_NON_CODE_EXTENSIONS:
                other_files.append(file_path)
            elif ext not in ELIXIR_EXTENSIONS:
                # Unknown extension - could be config files, etc.
                # Only remove if it's clearly not a code file
                # For now, we'll be conservative and not remove unknown extensions
                pass
    
    return markdown_files, text_files, other_files


def remove_comments_from_elixir(content: str) -> str:
    """
    Remove comments from Elixir code while preserving strings.
    
    Handles:
    - Single-line comments: # comment
    - Inline comments: code # comment
    - Multi-line comments: # comment spanning lines (Elixir doesn't have /* */ but handles # at line start)
    """
    lines = content.split('\n')
    result_lines = []
    
    in_string = False
    string_char = None  # ' or " or """
    i = 0
    
    while i < len(lines):
        line = lines[i]
        new_line = []
        j = 0
        
        while j < len(line):
            char = line[j]
            
            # Handle string detection
            if not in_string:
                # Check for string start
                if char in ('"', "'"):
                    # Check for triple quotes (sigils)
                    if j + 2 < len(line) and line[j:j+3] == '"""':
                        in_string = True
                        string_char = '"""'
                        new_line.append(line[j:j+3])
                        j += 3
                        continue
                    elif j + 1 < len(line) and line[j:j+2] == '""':
                        # Empty string
                        new_line.append('""')
                        j += 2
                        continue
                    else:
                        in_string = True
                        string_char = char
                        new_line.append(char)
                        j += 1
                        continue
                # Check for comment start
                elif char == '#':
                    # It's a comment, skip rest of line
                    break
                else:
                    new_line.append(char)
                    j += 1
            else:
                # We're inside a string
                if string_char == '"""':
                    # Check for triple quote end
                    if j + 2 < len(line) and line[j:j+3] == '"""':
                        in_string = False
                        string_char = None
                        new_line.append('"""')
                        j += 3
                        continue
                    else:
                        new_line.append(char)
                        j += 1
                else:
                    # Regular string
                    if char == '\\':
                        # Escape sequence
                        new_line.append(char)
                        j += 1
                        if j < len(line):
                            new_line.append(line[j])
                            j += 1
                    elif char == string_char:
                        # String end
                        in_string = False
                        string_char = None
                        new_line.append(char)
                        j += 1
                    else:
                        new_line.append(char)
                        j += 1
        
        # Join the line, strip trailing whitespace
        cleaned_line = ''.join(new_line).rstrip()
        if cleaned_line or not result_lines or result_lines[-1].strip():
            # Keep the line if it has content, or if previous line had content
            # (to preserve some structure)
            result_lines.append(cleaned_line)
        
        i += 1
    
    # Remove trailing empty lines
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()
    
    return '\n'.join(result_lines)


def process_elixir_files(repo_root: Path, dry_run: bool = False) -> List[Path]:
    """Process all Elixir files to remove comments."""
    processed_files = []
    
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        
        # Skip certain directories
        dirs[:] = [d for d in dirs if not should_skip_directory(root_path / d, repo_root)]
        
        for file in files:
            file_path = root_path / file
            
            # Skip files in preserved directories
            if is_in_preserved_directory(file_path, repo_root):
                continue
            
            # Skip the script itself
            if file_path.name == 'clean_repository.py':
                continue
            
            ext = file_path.suffix.lower()
            
            if ext in ELIXIR_EXTENSIONS:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    cleaned_content = remove_comments_from_elixir(content)
                    
                    if cleaned_content != content:
                        if not dry_run:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(cleaned_content)
                            print(f"Removed comments from: {file_path.relative_to(repo_root)}")
                        else:
                            print(f"[DRY RUN] Would remove comments from: {file_path.relative_to(repo_root)}")
                        
                        processed_files.append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}", file=sys.stderr)
    
    return processed_files


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clean repository: remove markdown, text files, and comments from Elixir code'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually removing'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create a backup before making changes'
    )
    parser.add_argument(
        '--repo-root',
        type=str,
        default=None,
        help='Repository root directory (default: script parent directory)'
    )
    
    args = parser.parse_args()
    
    # Determine repository root
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        # Assume script is in scripts/ directory
        script_dir = Path(__file__).parent.resolve()
        repo_root = script_dir.parent
    
    if not repo_root.exists():
        print(f"Error: Repository root does not exist: {repo_root}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Repository root: {repo_root}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()
    
    # Create backup if requested
    if args.backup and not args.dry_run:
        backup_dir = repo_root.parent / f"{repo_root.name}_backup"
        if backup_dir.exists():
            print(f"Error: Backup directory already exists: {backup_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Creating backup to: {backup_dir}")
        shutil.copytree(repo_root, backup_dir, ignore=shutil.ignore_patterns(
            '_build', 'deps', 'node_modules', '.git', 'cover', 'doc'
        ))
        print("Backup created.")
        print()
    
    # Find files to remove
    print("Finding files to remove...")
    markdown_files, text_files, other_files = find_files_to_remove(repo_root)
    
    print(f"Found {len(markdown_files)} markdown files")
    print(f"Found {len(text_files)} text files")
    print(f"Found {len(other_files)} other non-code files")
    
    # Remove markdown files
    if markdown_files:
        print("\nRemoving markdown files:")
        for file_path in markdown_files:
            rel_path = file_path.relative_to(repo_root)
            if args.dry_run:
                print(f"  [DRY RUN] Would remove: {rel_path}")
            else:
                try:
                    file_path.unlink()
                    print(f"  Removed: {rel_path}")
                except Exception as e:
                    print(f"  Error removing {rel_path}: {e}", file=sys.stderr)
    
    # Remove text files
    if text_files:
        print("\nRemoving text files:")
        for file_path in text_files:
            rel_path = file_path.relative_to(repo_root)
            if args.dry_run:
                print(f"  [DRY RUN] Would remove: {rel_path}")
            else:
                try:
                    file_path.unlink()
                    print(f"  Removed: {rel_path}")
                except Exception as e:
                    print(f"  Error removing {rel_path}: {e}", file=sys.stderr)
    
    # Remove other non-code files
    if other_files:
        print("\nRemoving other non-code files:")
        for file_path in other_files:
            rel_path = file_path.relative_to(repo_root)
            if args.dry_run:
                print(f"  [DRY RUN] Would remove: {rel_path}")
            else:
                try:
                    file_path.unlink()
                    print(f"  Removed: {rel_path}")
                except Exception as e:
                    print(f"  Error removing {rel_path}: {e}", file=sys.stderr)
    
    # Process Elixir files to remove comments
    print("\nProcessing Elixir files to remove comments...")
    processed_files = process_elixir_files(repo_root, dry_run=args.dry_run)
    
    print(f"\nSummary:")
    print(f"  Markdown files: {len(markdown_files)}")
    print(f"  Text files: {len(text_files)}")
    print(f"  Other non-code files: {len(other_files)}")
    print(f"  Elixir files processed: {len(processed_files)}")
    
    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to apply changes.")
    else:
        print("\nCleanup complete!")


if __name__ == "__main__":
    main()

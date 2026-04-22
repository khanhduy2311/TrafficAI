#!/usr/bin/env python3

import os
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def calculate_file_hash(filepath, algorithm='md5', buffer_size=8192):
    """Calculate the hash of a file."""
    hash_obj = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        # Read the file in chunks to handle large files efficiently
        buffer = f.read(buffer_size)
        while buffer:
            hash_obj.update(buffer)
            buffer = f.read(buffer_size)
    
    return hash_obj.hexdigest()

def find_duplicates(folder1, folder2, compare_content=True, include_subfolders=True):
    """
    Find duplicate files between two folders.
    
    Args:
        folder1: Path to the first folder
        folder2: Path to the second folder
        compare_content: Whether to compare file content using hash
        include_subfolders: Whether to search in subfolders
    
    Returns:
        Dictionary with duplicate files information
    """
    # Convert to Path objects
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    
    # Validate folders
    if not folder1.is_dir():
        raise ValueError(f"'{folder1}' is not a valid directory.")
    if not folder2.is_dir():
        raise ValueError(f"'{folder2}' is not a valid directory.")
    
    print(f"Scanning folder: {folder1}")
    # Get all files from the first folder
    files1 = {}
    pattern = "**/*" if include_subfolders else "*"
    
    for file_path in tqdm(list(folder1.glob(pattern))):
        if file_path.is_file():
            rel_path = file_path.relative_to(folder1)
            files1[rel_path] = {
                'path': file_path,
                'size': file_path.stat().st_size,
                'hash': None  # Will be calculated later if needed
            }
    
    print(f"Scanning folder: {folder2}")
    # Get all files from the second folder
    files2 = {}
    for file_path in tqdm(list(folder2.glob(pattern))):
        if file_path.is_file():
            rel_path = file_path.relative_to(folder2)
            files2[rel_path] = {
                'path': file_path,
                'size': file_path.stat().st_size,
                'hash': None  # Will be calculated later if needed
            }
    
    # Find files with the same name
    duplicates = []
    
    print("Comparing files...")
    for rel_path in tqdm(set(files1.keys()).intersection(files2.keys())):
        file1 = files1[rel_path]
        file2 = files2[rel_path]
        
        # First check if sizes are different
        if file1['size'] != file2['size']:
            continue
        
        # If comparing content is enabled, calculate hashes
        if compare_content:
            if file1['hash'] is None:
                file1['hash'] = calculate_file_hash(file1['path'])
            if file2['hash'] is None:
                file2['hash'] = calculate_file_hash(file2['path'])
            
            if file1['hash'] != file2['hash']:
                continue
        
        # If we get here, the files are considered duplicates
        duplicates.append({
            'relative_path': str(rel_path),
            'path1': str(file1['path']),
            'path2': str(file2['path']),
            'size': file1['size'],
            'is_identical': True if compare_content else "Not verified"
        })
    
    return {
        'total_files1': len(files1),
        'total_files2': len(files2),
        'duplicate_count': len(duplicates),
        'duplicates': duplicates
    }

def main():
    parser = argparse.ArgumentParser(description="Find duplicate files between two folders.")
    parser.add_argument("folder1", help="Path to the first folder")
    parser.add_argument("folder2", help="Path to the second folder")
    parser.add_argument("--output", "-o", help="Output file to save results (optional)")
    parser.add_argument("--skip-content", "-s", action="store_true", help="Skip content comparison (faster but less accurate)")
    parser.add_argument("--no-subfolders", "-n", action="store_true", help="Do not search in subfolders")
    
    args = parser.parse_args()
    
    try:
        result = find_duplicates(
            args.folder1, 
            args.folder2, 
            compare_content=not args.skip_content,
            include_subfolders=not args.no_subfolders
        )
        
        # Print summary
        print("\nSummary:")
        print(f"Files in folder 1: {result['total_files1']}")
        print(f"Files in folder 2: {result['total_files2']}")
        print(f"Duplicate files found: {result['duplicate_count']}")
        
        # Print duplicate files
        if result['duplicate_count'] > 0:
            print("\nDuplicate files:")
            for i, dup in enumerate(result['duplicates'], 1):
                print(f"{i}. {dup['relative_path']}")
                print(f"   - {dup['path1']}")
                print(f"   - {dup['path2']}")
                print(f"   - Size: {dup['size']} bytes")
                print(f"   - Content identical: {dup['is_identical']}")
        
        # Save to output file if specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
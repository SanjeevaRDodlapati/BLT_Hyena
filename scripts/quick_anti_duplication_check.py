#!/usr/bin/env python3
"""
Quick Anti-Duplication Check

A simple script to quickly check for existing functionality before implementation.
This integrates with the session workflow and existing documentation system.
"""

import subprocess
import sys
import os
from pathlib import Path

def quick_check(feature_description):
    """Perform a quick anti-duplication check."""
    print(f"🔍 Quick Anti-Duplication Check for: {feature_description}")
    print("=" * 60)
    
    workspace_root = Path.cwd()
    
    # 1. Quick grep search
    print("\n📝 Content Search Results:")
    terms = feature_description.lower().split()
    for term in terms[:3]:  # Limit to first 3 terms
        try:
            cmd = f"grep -r -i '{term}' . --include='*.py' --include='*.md' --include='*.sh' | head -5"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                print(f"\n'{term}' found in:")
                for line in result.stdout.strip().split('\n')[:3]:
                    if ':' in line:
                        file_path = line.split(':')[0]
                        print(f"  - {file_path}")
        except:
            continue
    
    # 2. File pattern search
    print("\n📁 File Pattern Search:")
    for term in terms[:3]:
        try:
            cmd = f"find . -name '*{term}*' -type f | head -3"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                print(f"\nFiles matching '{term}':")
                for file_path in result.stdout.strip().split('\n')[:3]:
                    if file_path:
                        print(f"  - {file_path}")
        except:
            continue
    
    # 3. Documentation search
    print("\n📚 Documentation Search:")
    docs_dir = workspace_root / "docs"
    if docs_dir.exists():
        for term in terms[:2]:
            try:
                cmd = f"find docs/ -name '*.md' -exec grep -l -i '{term}' {{}} \\;"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.stdout:
                    print(f"\n'{term}' mentioned in docs:")
                    for doc_path in result.stdout.strip().split('\n')[:3]:
                        if doc_path:
                            print(f"  - {doc_path}")
            except:
                continue
    
    # 4. Session history search
    print("\n📋 Session History Search:")
    session_dirs = [workspace_root / "session_notes", workspace_root / "admin"]
    for session_dir in session_dirs:
        if session_dir.exists():
            for term in terms[:2]:
                try:
                    cmd = f"find {session_dir} -name '*.md' -exec grep -l -i '{term}' {{}} \\;"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.stdout:
                        print(f"\n'{term}' in session history:")
                        for file_path in result.stdout.strip().split('\n')[:2]:
                            if file_path:
                                print(f"  - {file_path}")
                except:
                    continue
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATION:")
    print("- Review the found files to check for existing functionality")
    print("- Run full discovery if multiple relevant files found:")
    print(f"  python scripts/anti_duplication_discovery.py \"{feature_description}\"")
    print("- Update session notes with discovery results")
    print("- Document decision rationale in ANTI_DUPLICATION_WORKFLOW.md format")

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_anti_duplication_check.py \"feature description\"")
        print("Example: python quick_anti_duplication_check.py \"push to multiple git repositories\"")
        sys.exit(1)
    
    feature_description = " ".join(sys.argv[1:])
    quick_check(feature_description)

if __name__ == "__main__":
    main()

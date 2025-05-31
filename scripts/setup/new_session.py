#!/usr/bin/env python3
"""
New Session Script for Hyena-GLT Framework

This script helps start a new development session by:
1. Running context recovery
2. Creating new session notes from template
3. Providing quick setup guidance

Usage:
    python scripts/new_session.py [--session-id XXX]
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run shell command and return output."""
    try:
        if isinstance(cmd, str):
            cmd = cmd.split()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception as e:
        print(f"Command failed: {' '.join(cmd)} - {e}")
        return None


def get_next_session_id(repo_path, date_str):
    """Find the next session ID for today."""
    session_dir = repo_path / "session_notes"
    if not session_dir.exists():
        session_dir.mkdir(parents=True)
        return "001"

    # Find existing session files for today
    pattern = f"session_notes_{date_str}-*.md"
    existing_files = list(session_dir.glob(pattern))

    if not existing_files:
        return "001"

    # Extract session numbers and find the next one
    session_numbers = []
    for file in existing_files:
        parts = file.stem.split("-")
        if len(parts) >= 4:
            try:
                session_numbers.append(int(parts[-1]))
            except ValueError:
                continue

    if not session_numbers:
        return "001"

    next_num = max(session_numbers) + 1
    return f"{next_num:03d}"


def create_session_notes(repo_path, session_id=None):
    """Create new session notes from template."""
    today = datetime.now().strftime("%Y-%m-%d")

    if session_id is None:
        session_id = get_next_session_id(repo_path, today)

    session_filename = f"session_notes_{today}-{session_id}.md"
    session_path = repo_path / "session_notes" / session_filename
    template_path = repo_path / "docs" / "SESSION_NOTES_TEMPLATE.md"

    if not template_path.exists():
        print(f"❌ Template not found: {template_path}")
        return None

    # Copy template to new session file
    shutil.copy2(template_path, session_path)

    # Replace placeholder values in the template
    content = session_path.read_text()

    # Replace basic placeholders
    content = content.replace("[YYYY-MM-DD]", today)
    content = content.replace(
        "[Unique identifier, e.g., 2025-01-28-001]", f"{today}-{session_id}"
    )

    session_path.write_text(content)

    print(f"✅ Created session notes: {session_path}")
    return session_path


def run_context_recovery(repo_path):
    """Run the context recovery script."""
    script_path = repo_path / "scripts" / "context_recovery.py"
    if script_path.exists():
        print("🔍 Running context recovery...")
        print("-" * 40)
        subprocess.run([sys.executable, str(script_path)], cwd=repo_path)
        print("-" * 40)
    else:
        print("⚠️  Context recovery script not found")


def check_environment(repo_path):
    """Check development environment setup."""
    print("\n🔧 Environment Check:")

    # Check if in repo directory
    if repo_path.name != "BLT_Hyena":
        print("⚠️  Not in BLT_Hyena repository directory")

    # Check virtual environment
    venv_active = "VIRTUAL_ENV" in os.environ
    print(f"  Virtual Environment: {'✅ Active' if venv_active else '❌ Not Active'}")

    if not venv_active:
        print("    💡 Activate with: source venv/bin/activate  # or your venv path")

    # Check if package is installed
    pip_show = run_command("pip show hyena-glt", cwd=repo_path)
    package_installed = pip_show is not None
    print(f"  Package Installed: {'✅ Yes' if package_installed else '❌ No'}")

    if not package_installed:
        print("    💡 Install with: pip install -e .")

    # Check git status
    git_status = run_command("git status --porcelain", cwd=repo_path)
    clean_repo = not git_status
    print(f"  Git Status: {'✅ Clean' if clean_repo else '⚠️  Uncommitted changes'}")

    if not clean_repo:
        print("    💡 Consider committing or stashing changes before starting")


def provide_session_guidance():
    """Provide guidance for starting the session."""
    print("\n🚀 Session Startup Guidance:")
    print("1. Review the context recovery summary above")
    print("2. Open your session notes file for tracking progress")
    print("3. Check PROJECT_STATE.md for current development stage")
    print("4. Run tests to verify system health: pytest tests/ -v")
    print("5. Begin work on current stage objectives")

    print("\n📋 Quick Commands:")
    print("  # Run tests")
    print("  pytest tests/ -v")
    print("")
    print("  # Run tests with coverage")
    print("  pytest --cov=hyena_glt --cov-report=html")
    print("")
    print("  # View current development stage")
    print("  grep -A 5 -B 5 '🔄' PROJECT_STATE.md")
    print("")
    print("  # Check recent changes")
    print("  git log --oneline -10")


def main():
    parser = argparse.ArgumentParser(
        description="Start new development session for Hyena-GLT Framework"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        help="Specific session ID (e.g., 001, 002). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--skip-context-recovery",
        action="store_true",
        help="Skip running context recovery script",
    )
    parser.add_argument(
        "--no-session-notes", action="store_true", help="Skip creating session notes"
    )

    args = parser.parse_args()

    repo_path = Path.cwd()

    print("🎯 Starting New Development Session")
    print("=" * 50)

    # Run context recovery unless skipped
    if not args.skip_context_recovery:
        run_context_recovery(repo_path)

    # Create session notes unless skipped
    session_path = None
    if not args.no_session_notes:
        print("\n📝 Creating Session Notes...")
        session_path = create_session_notes(repo_path, args.session_id)

    # Check environment
    check_environment(repo_path)

    # Provide guidance
    provide_session_guidance()

    # Final message
    print("\n" + "=" * 50)
    print("🎉 Session setup complete!")
    if session_path:
        print(f"📄 Session notes: {session_path}")
    print("💻 Happy coding!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Code Quality Management Script
==============================

Pragmatic approach to implementing code quality checks for BLT_Hyena.
Focuses on fixing critical issues first, then improving incrementally.
"""

import subprocess
from pathlib import Path


class CodeQualityManager:
    """Manages code quality checks and fixes for the project."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.core_dirs = ["hyena_glt", "tests", "scripts"]

    def run_command(
        self, cmd: list[str], description: str, check_only: bool = True
    ) -> bool:
        """Run a command and return success status."""
        print(f"\n🔍 {description}")
        print(f"   Command: {' '.join(cmd)}")

        try:
            if check_only:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    check=False,
                )
                if result.returncode == 0:
                    print("   ✅ PASSED")
                    return True
                else:
                    print("   ❌ ISSUES FOUND:")
                    if result.stdout:
                        lines = result.stdout.strip().split("\n")[
                            :10
                        ]  # Show first 10 lines
                        for line in lines:
                            print(f"      {line}")
                        if len(result.stdout.strip().split("\n")) > 10:
                            print("      ... and more")
                    return False
            else:
                # Actually fix issues
                result = subprocess.run(cmd, cwd=self.project_root, check=False)
                return result.returncode == 0
        except FileNotFoundError:
            print("   ❌ Tool not installed")
            return False

    def check_ruff(self, fix: bool = False) -> bool:
        """Check code with Ruff (modern linter)."""
        cmd = ["ruff", "check"]
        if fix:
            cmd.append("--fix")
        cmd.extend(self.core_dirs)

        return self.run_command(cmd, "Ruff Linting", check_only=not fix)

    def check_black(self, fix: bool = False) -> bool:
        """Check code formatting with Black."""
        cmd = ["black"]
        if not fix:
            cmd.extend(["--check", "--diff"])
        cmd.extend(self.core_dirs)

        return self.run_command(cmd, "Black Formatting", check_only=not fix)

    def check_isort(self, fix: bool = False) -> bool:
        """Check import sorting with isort."""
        cmd = ["isort"]
        if not fix:
            cmd.extend(["--check-only", "--diff"])
        cmd.extend(self.core_dirs)

        return self.run_command(cmd, "isort Import Sorting", check_only=not fix)

    def check_mypy(self) -> bool:
        """Run type checking with MyPy."""
        cmd = ["mypy", "hyena_glt", "--ignore-missing-imports", "--no-error-summary"]
        return self.run_command(cmd, "MyPy Type Checking", check_only=True)

    def run_all_checks(self) -> dict:
        """Run all code quality checks."""
        print("=" * 80)
        print("🎯 BLT_HYENA CODE QUALITY ASSESSMENT")
        print("=" * 80)

        results = {
            "ruff": self.check_ruff(),
            "black": self.check_black(),
            "isort": self.check_isort(),
            "mypy": self.check_mypy(),
        }

        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)

        passed = sum(results.values())
        total = len(results)

        for tool, passed_check in results.items():
            status = "✅ PASSED" if passed_check else "❌ NEEDS WORK"
            print(f"   {tool:<8} {status}")

        print(f"\n🎯 Overall Score: {passed}/{total} ({passed/total*100:.0f}%)")

        if passed == total:
            print("🎉 EXCELLENT! Code quality is production-ready!")
        elif passed >= total * 0.75:
            print("👍 GOOD! Minor issues to address")
        elif passed >= total * 0.5:
            print("⚠️  MODERATE! Some cleanup needed")
        else:
            print("🔧 NEEDS WORK! Significant cleanup required")

        return results

    def fix_issues(self) -> None:
        """Automatically fix what can be fixed."""
        print("=" * 80)
        print("🔧 AUTOMATIC CODE QUALITY FIXES")
        print("=" * 80)

        print("\n1. Fixing import sorting...")
        self.check_isort(fix=True)

        print("\n2. Fixing code formatting...")
        self.check_black(fix=True)

        print("\n3. Fixing auto-fixable linting issues...")
        self.check_ruff(fix=True)

        print("\n✅ Automatic fixes complete!")
        print("💡 Run quality checks again to see remaining issues.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="BLT_Hyena Code Quality Management")
    parser.add_argument("--check", action="store_true", help="Run quality checks only")
    parser.add_argument("--fix", action="store_true", help="Fix issues automatically")
    parser.add_argument("--all", action="store_true", help="Run checks then fixes")

    args = parser.parse_args()

    manager = CodeQualityManager()

    if args.fix or args.all:
        manager.fix_issues()
        if args.all:
            print("\n" + "=" * 80)
            print("🔍 RUNNING CHECKS AFTER FIXES")
            print("=" * 80)
            manager.run_all_checks()
    else:
        manager.run_all_checks()


if __name__ == "__main__":
    main()

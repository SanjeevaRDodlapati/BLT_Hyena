#!/usr/bin/env python3
"""
Context Recovery Script for Hyena-GLT Framework

This script provides automated assessment of the current project state
to enable quick context recovery at the start of development sessions.

Usage:
    python scripts/context_recovery.py [--verbose] [--full-report]
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path


class ContextRecovery:
    def __init__(self, repo_path=None, verbose=False):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.verbose = verbose
        self.report = {}

    def run_command(self, cmd, capture_output=True):
        """Run shell command and return output."""
        try:
            if isinstance(cmd, str):
                cmd = cmd.split()
            result = subprocess.run(
                cmd, capture_output=capture_output, text=True, cwd=self.repo_path
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception as e:
            if self.verbose:
                print(f"Command failed: {' '.join(cmd)} - {e}")
            return None

    def check_git_status(self):
        """Check git repository status."""
        print("🔍 Checking Git Status...")

        git_status = {}

        # Current branch
        branch = self.run_command("git branch --show-current")
        git_status["current_branch"] = branch or "unknown"

        # Git status
        status_output = self.run_command("git status --porcelain")
        if status_output:
            modified_files = []
            untracked_files = []
            for line in status_output.split("\n"):
                if line.startswith(" M") or line.startswith("M "):
                    modified_files.append(line[3:])
                elif line.startswith("??"):
                    untracked_files.append(line[3:])
            git_status["modified_files"] = modified_files
            git_status["untracked_files"] = untracked_files
        else:
            git_status["clean"] = True

        # Last commit
        last_commit = self.run_command(
            "git log -1 --pretty=format:'%h - %s (%cr) <%an>'"
        )
        git_status["last_commit"] = last_commit

        # Commits since last tag
        commits_since_tag = self.run_command(
            "git rev-list $(git describe --tags --abbrev=0)..HEAD --count"
        )
        git_status["commits_since_last_tag"] = commits_since_tag

        self.report["git_status"] = git_status

        if self.verbose:
            print(f"  Branch: {branch}")
            print(f"  Last commit: {last_commit}")
            if git_status.get("modified_files"):
                print(f"  Modified files: {len(git_status['modified_files'])}")
            if git_status.get("untracked_files"):
                print(f"  Untracked files: {len(git_status['untracked_files'])}")

    def check_project_structure(self):
        """Analyze project structure and file counts."""
        print("📁 Analyzing Project Structure...")

        structure = {}
        important_files = [
            "PROJECT_STATE.md",
            "CHANGELOG.md",
            "README.md",
            "requirements.txt",
            "setup.py",
            "pytest.ini",
        ]

        for file in important_files:
            file_path = self.repo_path / file
            structure[file] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "modified": (
                    datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    if file_path.exists()
                    else None
                ),
            }

        # Count files by directory
        directories = ["hyena_glt", "docs", "tests", "examples", "notebooks", "scripts"]
        for dir_name in directories:
            dir_path = self.repo_path / dir_name
            if dir_path.exists():
                py_files = list(dir_path.rglob("*.py"))
                md_files = list(dir_path.rglob("*.md"))
                ipynb_files = list(dir_path.rglob("*.ipynb"))

                structure[dir_name] = {
                    "python_files": len(py_files),
                    "markdown_files": len(md_files),
                    "notebook_files": len(ipynb_files),
                    "total_files": len(list(dir_path.rglob("*")))
                    - len(list(dir_path.rglob("*/"))),
                }

        self.report["project_structure"] = structure

        if self.verbose:
            for dir_name, stats in structure.items():
                if isinstance(stats, dict) and "total_files" in stats:
                    print(
                        f"  {dir_name}: {stats['total_files']} files "
                        f"({stats['python_files']} .py, {stats['markdown_files']} .md)"
                    )

    def check_version_info(self):
        """Extract version information from key files."""
        print("📋 Checking Version Information...")

        version_info = {}

        # Check setup.py version
        setup_py = self.repo_path / "setup.py"
        if setup_py.exists():
            content = setup_py.read_text()
            version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if version_match:
                version_info["setup_py"] = version_match.group(1)

        # Check CHANGELOG.md for latest version
        changelog = self.repo_path / "CHANGELOG.md"
        if changelog.exists():
            content = changelog.read_text()
            version_match = re.search(r"##\s*\[([^\]]+)\]", content)
            if version_match:
                version_info["changelog_latest"] = version_match.group(1)

        # Check package __init__.py
        init_py = self.repo_path / "hyena_glt" / "__init__.py"
        if init_py.exists():
            content = init_py.read_text()
            version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if version_match:
                version_info["package_version"] = version_match.group(1)

        self.report["version_info"] = version_info

        if self.verbose:
            for source, version in version_info.items():
                print(f"  {source}: {version}")

    def check_test_status(self):
        """Check test configuration and recent results."""
        print("🧪 Checking Test Configuration...")

        test_info = {}

        # Check if pytest is configured
        pytest_ini = self.repo_path / "pytest.ini"
        test_info["pytest_configured"] = pytest_ini.exists()

        # Count test files
        tests_dir = self.repo_path / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.rglob("test_*.py"))
            test_info["test_files_count"] = len(test_files)

            # Check for different test types
            unit_tests = (
                list((tests_dir / "unit").rglob("test_*.py"))
                if (tests_dir / "unit").exists()
                else []
            )
            integration_tests = (
                list((tests_dir / "integration").rglob("test_*.py"))
                if (tests_dir / "integration").exists()
                else []
            )

            test_info["unit_tests"] = len(unit_tests)
            test_info["integration_tests"] = len(integration_tests)

        # Try to run a quick test check (dry run)
        test_dry_run = self.run_command("python -m pytest --collect-only -q")
        if test_dry_run:
            lines = test_dry_run.split("\n")
            for line in lines:
                if "collected" in line:
                    test_info["total_tests_found"] = line.strip()
                    break

        self.report["test_status"] = test_info

        if self.verbose:
            print(f"  Pytest configured: {test_info.get('pytest_configured', False)}")
            print(f"  Test files: {test_info.get('test_files_count', 0)}")
            if "total_tests_found" in test_info:
                print(f"  {test_info['total_tests_found']}")

    def check_documentation_status(self):
        """Check documentation completeness."""
        print("📚 Checking Documentation Status...")

        docs_info = {}

        docs_dir = self.repo_path / "docs"
        if docs_dir.exists():
            doc_files = list(docs_dir.glob("*.md"))
            docs_info["documentation_files"] = len(doc_files)
            docs_info["doc_file_names"] = [f.name for f in doc_files]

        # Check for key documentation files
        key_docs = [
            "PROJECT_STATE.md",
            "README.md",
            "docs/ARCHITECTURE.md",
            "docs/USER_GUIDE.md",
            "docs/API.md",
        ]

        docs_info["key_docs_status"] = {}
        for doc in key_docs:
            doc_path = self.repo_path / doc
            docs_info["key_docs_status"][doc] = {
                "exists": doc_path.exists(),
                "size": doc_path.stat().st_size if doc_path.exists() else 0,
            }

        self.report["documentation_status"] = docs_info

        if self.verbose:
            print(f"  Documentation files: {docs_info.get('documentation_files', 0)}")
            for doc, status in docs_info.get("key_docs_status", {}).items():
                status_icon = "✅" if status["exists"] else "❌"
                print(f"  {status_icon} {doc}")

    def check_dependencies(self):
        """Check dependency status."""
        print("📦 Checking Dependencies...")

        deps_info = {}

        # Check requirements.txt
        req_file = self.repo_path / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text()
            deps = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.startswith("#")
            ]
            deps_info["requirements_count"] = len(deps)
            deps_info["requirements"] = deps[:10]  # First 10 for brevity

        # Check if virtual environment is active
        deps_info["venv_active"] = "VIRTUAL_ENV" in os.environ

        # Try to check if package is installed in development mode
        pip_show = self.run_command("pip show hyena-glt")
        deps_info["package_installed"] = pip_show is not None

        self.report["dependencies"] = deps_info

        if self.verbose:
            print(f"  Requirements defined: {deps_info.get('requirements_count', 0)}")
            print(f"  Virtual env active: {deps_info.get('venv_active', False)}")
            print(f"  Package installed: {deps_info.get('package_installed', False)}")

    def detect_development_stage(self):
        """Try to detect current development stage."""
        print("🎯 Detecting Development Stage...")

        stage_info = {}

        # Read PROJECT_STATE.md to get stage info
        state_file = self.repo_path / "PROJECT_STATE.md"
        if state_file.exists():
            content = state_file.read_text()

            # Look for current stage indicators
            stage_patterns = [
                (r"Stage\s+(\d+)[^:]*:\s*([^-\n]+)\s*-\s*🔄", "current"),
                (r"Stage\s+(\d+)[^:]*:\s*([^-\n]+)\s*-\s*✅", "completed"),
                (r"Stage\s+(\d+)[^:]*:\s*([^-\n]+)\s*-\s*⏳", "pending"),
            ]

            stages = {"completed": [], "current": [], "pending": []}

            for pattern, status in stage_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    stage_num = match.group(1)
                    stage_name = match.group(2).strip()
                    stages[status].append(f"Stage {stage_num}: {stage_name}")

            stage_info["stages"] = stages

        self.report["development_stage"] = stage_info

        if self.verbose:
            if "stages" in stage_info:
                for status, stage_list in stage_info["stages"].items():
                    if stage_list:
                        print(f"  {status.title()}: {', '.join(stage_list)}")

    def generate_summary(self):
        """Generate a human-readable summary."""
        print("\n" + "=" * 60)
        print("📊 CONTEXT RECOVERY SUMMARY")
        print("=" * 60)

        # Git status summary
        git = self.report.get("git_status", {})
        print(f"🔗 Git: Branch '{git.get('current_branch', 'unknown')}'")
        if git.get("clean"):
            print("   ✅ Working directory clean")
        else:
            modified = len(git.get("modified_files", []))
            untracked = len(git.get("untracked_files", []))
            if modified:
                print(f"   🔄 {modified} modified files")
            if untracked:
                print(f"   ➕ {untracked} untracked files")

        # Version info
        version = self.report.get("version_info", {})
        if version:
            print(f"📋 Version: {version.get('package_version', 'unknown')}")

        # Development stage
        stages = self.report.get("development_stage", {}).get("stages", {})
        current_stages = stages.get("current", [])
        if current_stages:
            print(f"🎯 Current Stage: {current_stages[0]}")

        # Project health
        structure = self.report.get("project_structure", {})
        if "hyena_glt" in structure:
            py_files = structure["hyena_glt"]["python_files"]
            print(f"🏗️  Code: {py_files} Python files in core framework")

        test_info = self.report.get("test_status", {})
        if test_info.get("test_files_count"):
            print(f"🧪 Tests: {test_info['test_files_count']} test files")

        docs_info = self.report.get("documentation_status", {})
        if docs_info.get("documentation_files"):
            print(f"📚 Docs: {docs_info['documentation_files']} documentation files")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")

        # Check if virtual env is active
        if not self.report.get("dependencies", {}).get("venv_active"):
            print("   ⚠️  Activate virtual environment")

        # Check if package is installed
        if not self.report.get("dependencies", {}).get("package_installed"):
            print("   ⚠️  Install package in development mode: pip install -e .")

        # Check if there are uncommitted changes
        if not git.get("clean"):
            print("   ⚠️  Commit or stash uncommitted changes")

        # Check for PROJECT_STATE.md
        if (
            not self.report.get("project_structure", {})
            .get("PROJECT_STATE.md", {})
            .get("exists")
        ):
            print("   ⚠️  PROJECT_STATE.md not found - consider creating it")

        print("\n✅ Ready to start development session!")

    def save_report(self, output_file):
        """Save detailed report to JSON file."""
        self.report["generated_at"] = datetime.now().isoformat()
        self.report["repo_path"] = str(self.repo_path)

        with open(output_file, "w") as f:
            json.dump(self.report, f, indent=2)

        print(f"\n📄 Detailed report saved to: {output_file}")

    def run_full_assessment(self, save_report=False):
        """Run complete context recovery assessment."""
        print("🚀 Running Context Recovery Assessment...")
        print("=" * 60)

        self.check_git_status()
        self.check_project_structure()
        self.check_version_info()
        self.check_test_status()
        self.check_documentation_status()
        self.check_dependencies()
        self.detect_development_stage()

        self.generate_summary()

        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.repo_path / f"context_recovery_report_{timestamp}.json"
            self.save_report(report_file)


def main():
    parser = argparse.ArgumentParser(
        description="Context Recovery Script for Hyena-GLT Framework"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--full-report",
        "-f",
        action="store_true",
        help="Save full detailed report to JSON file",
    )
    parser.add_argument(
        "--repo-path",
        "-r",
        type=str,
        help="Path to repository (default: current directory)",
    )

    args = parser.parse_args()

    recovery = ContextRecovery(repo_path=args.repo_path, verbose=args.verbose)

    recovery.run_full_assessment(save_report=args.full_report)


if __name__ == "__main__":
    main()

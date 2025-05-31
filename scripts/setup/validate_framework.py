#!/usr/bin/env python3
"""
Final validation script for Hyena-GLT framework.
Validates framework completeness, structure, and quality.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_directory_structure() -> tuple[bool, list[str]]:
    """Check if all required directories and files exist."""
    required_structure = {
        "hyena_glt/": {
            "__init__.py": True,
            "config/": {"__init__.py": True},
            "data/": {"__init__.py": True},
            "model/": {"__init__.py": True},
            "training/": {"__init__.py": True},
            "evaluation/": {"__init__.py": True},
            "optimization/": {"__init__.py": True},
            "cli/": {"__init__.py": True},
            "utils/": {"__init__.py": True},
        },
        "examples/": {
            "basic_usage.py": True,
            "fine_tuning.py": True,
            "evaluation.py": True,
            "generation.py": True,
        },
        "notebooks/": {
            "01_introduction.ipynb": True,
            "02_tokenization.ipynb": True,
            "03_model_architecture.ipynb": True,
            "04_training.ipynb": True,
            "05_evaluation.ipynb": True,
            "06_fine_tuning.ipynb": True,
            "07_generation.ipynb": True,
        },
        "tests/": {
            "__init__.py": True,
            "unit/": {"__init__.py": True},
            "integration/": {"__init__.py": True},
        },
        "docs/": {
            "API.md": True,
            "ARCHITECTURE.md": True,
            "USER_GUIDE.md": True,
            "testing.md": True,
        },
        "scripts/": {"run_tests.py": True},
        "": {  # Root files
            "README.md": True,
            "requirements.txt": True,
            "setup.py": True,
            "pytest.ini": True,
            "conftest.py": True,
            ".github/": {"workflows/": {"ci-cd.yml": True}},
        },
    }

    missing = []

    def check_structure(structure: dict, base_path: Path = Path(".")):
        for item, value in structure.items():
            if item == "":  # Root level
                continue

            item_path = base_path / item

            if isinstance(value, dict):
                if not item_path.is_dir():
                    missing.append(f"Directory: {item_path}")
                else:
                    check_structure(value, item_path)
            else:
                if not item_path.is_file():
                    missing.append(f"File: {item_path}")

    # Check root files
    root_files = required_structure[""]
    for file, required in root_files.items():
        if file.endswith("/"):
            check_structure({file[:-1]: root_files[file]})
        else:
            if required and not Path(file).is_file():
                missing.append(f"File: {file}")

    # Check other directories
    for dir_name, contents in required_structure.items():
        if dir_name != "":
            check_structure({dir_name: contents})

    return len(missing) == 0, missing


def check_python_syntax() -> tuple[bool, list[str]]:
    """Check Python files for syntax errors."""
    errors = []

    for root, dirs, files in os.walk("."):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                try:
                    with open(file_path, encoding="utf-8") as f:
                        compile(f.read(), file_path, "exec")
                except SyntaxError as e:
                    errors.append(f"Syntax error in {file_path}: {e}")
                except Exception as e:
                    errors.append(f"Error reading {file_path}: {e}")

    return len(errors) == 0, errors


def check_imports() -> tuple[bool, list[str]]:
    """Check if core modules can be imported (syntax check)."""
    core_modules = [
        "hyena_glt.config",
        "hyena_glt.data",
        "hyena_glt.model",
        "hyena_glt.training",
        "hyena_glt.evaluation",
        "hyena_glt.optimization",
        "hyena_glt.cli",
        "hyena_glt.utils",
    ]

    import_errors = []

    for module in core_modules:
        try:
            # Just check if we can compile the module files
            module_path = Path(module.replace(".", "/"))
            init_path = module_path / "__init__.py"

            if init_path.exists():
                with open(init_path) as f:
                    compile(f.read(), str(init_path), "exec")
        except Exception as e:
            import_errors.append(f"Module {module}: {e}")

    return len(import_errors) == 0, import_errors


def check_documentation() -> tuple[bool, list[str]]:
    """Check documentation completeness."""
    doc_issues = []

    # Check main documentation files
    doc_files = [
        "README.md",
        "docs/API.md",
        "docs/ARCHITECTURE.md",
        "docs/USER_GUIDE.md",
        "docs/testing.md",
    ]

    for doc_file in doc_files:
        if not Path(doc_file).exists():
            doc_issues.append(f"Missing documentation: {doc_file}")
        else:
            # Check if file has substantial content (> 1KB)
            size = Path(doc_file).stat().st_size
            if size < 1000:
                doc_issues.append(f"Documentation too brief: {doc_file} ({size} bytes)")

    return len(doc_issues) == 0, doc_issues


def check_test_coverage() -> tuple[bool, list[str]]:
    """Check test file coverage."""
    test_issues = []

    # Core modules that should have tests
    core_modules = [
        "config",
        "data",
        "model",
        "training",
        "evaluation",
        "optimization",
        "utils",
    ]

    for module in core_modules:
        test_file = Path(f"tests/unit/test_{module}.py")
        if not test_file.exists():
            test_issues.append(f"Missing unit tests for {module}")

    # Check for integration tests
    integration_tests = [
        "tests/integration/test_complete_workflows.py",
        "tests/integration/test_performance_benchmarks.py",
    ]

    for test_file in integration_tests:
        if not Path(test_file).exists():
            test_issues.append(f"Missing integration test: {test_file}")

    return len(test_issues) == 0, test_issues


def check_examples() -> tuple[bool, list[str]]:
    """Check example completeness."""
    example_issues = []

    required_examples = [
        "examples/basic_usage.py",
        "examples/fine_tuning.py",
        "examples/evaluation.py",
        "examples/generation.py",
    ]

    for example in required_examples:
        if not Path(example).exists():
            example_issues.append(f"Missing example: {example}")

    return len(example_issues) == 0, example_issues


def generate_framework_report() -> dict:
    """Generate comprehensive framework validation report."""
    report = {
        "framework_name": "Hyena-GLT",
        "version": "0.1.0",
        "validation_results": {},
    }

    # Run all checks
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Python Syntax", check_python_syntax),
        ("Module Imports", check_imports),
        ("Documentation", check_documentation),
        ("Test Coverage", check_test_coverage),
        ("Examples", check_examples),
    ]

    total_passed = 0
    total_checks = len(checks)

    for check_name, check_func in checks:
        logger.info(f"Running check: {check_name}")
        passed, issues = check_func()

        report["validation_results"][check_name.lower().replace(" ", "_")] = {
            "passed": passed,
            "issues": issues,
            "issue_count": len(issues),
        }

        if passed:
            total_passed += 1
            logger.info(f"✅ {check_name}: PASSED")
        else:
            logger.warning(f"❌ {check_name}: FAILED ({len(issues)} issues)")
            for issue in issues[:5]:  # Show first 5 issues
                logger.warning(f"  - {issue}")
            if len(issues) > 5:
                logger.warning(f"  ... and {len(issues) - 5} more issues")

    report["summary"] = {
        "total_checks": total_checks,
        "passed_checks": total_passed,
        "completion_percentage": round((total_passed / total_checks) * 100, 1),
    }

    return report


def main():
    """Main validation function."""
    logger.info("🔍 Starting Hyena-GLT Framework Validation")
    logger.info("=" * 60)

    # Change to project root if needed
    if not Path("hyena_glt").exists():
        logger.error("Not in Hyena-GLT project root directory")
        sys.exit(1)

    # Generate validation report
    report = generate_framework_report()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 60)

    summary = report["summary"]
    logger.info(f"Framework: {report['framework_name']} v{report['version']}")
    logger.info(f"Checks Passed: {summary['passed_checks']}/{summary['total_checks']}")
    logger.info(f"Completion: {summary['completion_percentage']}%")

    # Save detailed report
    report_file = "framework_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"💾 Detailed report saved to: {report_file}")

    # Framework status
    if summary["completion_percentage"] >= 90:
        logger.info("🎉 Framework is PRODUCTION READY!")
        status = "PRODUCTION_READY"
    elif summary["completion_percentage"] >= 80:
        logger.info("✅ Framework is RELEASE CANDIDATE")
        status = "RELEASE_CANDIDATE"
    elif summary["completion_percentage"] >= 70:
        logger.info("⚠️  Framework is in BETA stage")
        status = "BETA"
    else:
        logger.info("🚧 Framework is in DEVELOPMENT stage")
        status = "DEVELOPMENT"

    # Exit code based on completion
    exit_code = 0 if summary["completion_percentage"] >= 80 else 1

    logger.info(f"🏷️  Framework Status: {status}")
    logger.info("=" * 60)

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

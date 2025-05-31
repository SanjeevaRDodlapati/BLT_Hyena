#!/usr/bin/env python3
"""
Test runner script for Hyena-GLT framework.

This script provides various test running options and generates test reports.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description, capture_output=False):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        else:
            result = subprocess.run(cmd, check=True)

        end_time = time.time()
        print(
            f"\n✅ {description} completed successfully in {end_time - start_time:.2f}s"
        )
        return True

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"\n❌ {description} failed after {end_time - start_time:.2f}s")
        if capture_output and hasattr(e, "stdout"):
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
        return False


def run_unit_tests(args):
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]

    if args.verbose:
        cmd.append("-v")
    if args.coverage:
        cmd.extend(["--cov=hyena_glt", "--cov-report=html", "--cov-report=term"])
    if args.parallel:
        cmd.extend(["-n", "auto"])
    if args.markers:
        cmd.extend(["-m", args.markers])

    return run_command(cmd, "Unit Tests")


def run_integration_tests(args):
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/"]

    if args.verbose:
        cmd.append("-v")
    if args.timeout:
        cmd.extend(["--timeout", str(args.timeout)])
    if args.markers:
        cmd.extend(["-m", args.markers])
    else:
        # Skip slow tests by default in integration
        cmd.extend(["-m", "not slow"])

    return run_command(cmd, "Integration Tests")


def run_benchmark_tests(args):
    """Run benchmark tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/test_performance_benchmarks.py"]

    if args.verbose:
        cmd.append("-v")
    cmd.extend(["-m", "benchmark"])
    cmd.extend(["--timeout", "1800"])  # 30 minutes for benchmarks

    return run_command(cmd, "Benchmark Tests")


def run_gpu_tests(args):
    """Run GPU tests."""
    # Check if CUDA is available
    try:
        import torch

        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping GPU tests")
            return True
    except ImportError:
        print("⚠️  PyTorch not available, skipping GPU tests")
        return True

    cmd = ["python", "-m", "pytest", "tests/"]
    cmd.extend(["-m", "gpu"])
    cmd.extend(["--timeout", "900"])  # 15 minutes for GPU tests

    if args.verbose:
        cmd.append("-v")

    return run_command(cmd, "GPU Tests")


def run_smoke_tests(args):
    """Run quick smoke tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    cmd.extend(["-m", "smoke or (unit and not slow)"])
    cmd.extend(["--timeout", "60"])

    if args.verbose:
        cmd.append("-v")

    return run_command(cmd, "Smoke Tests")


def run_memory_tests(args):
    """Run memory-intensive tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    cmd.extend(["-m", "memory_intensive"])
    cmd.extend(["--timeout", "600"])

    if args.verbose:
        cmd.append("-v")

    return run_command(cmd, "Memory Tests")


def run_regression_tests(args):
    """Run regression tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    cmd.extend(["-m", "regression"])

    if args.verbose:
        cmd.append("-v")

    return run_command(cmd, "Regression Tests")


def run_lint_checks(args):
    """Run code quality checks."""
    checks = []

    # Flake8
    cmd = [
        "flake8",
        "hyena_glt",
        "tests",
        "--max-line-length=88",
        "--extend-ignore=E203,W503",
    ]
    checks.append((cmd, "Flake8 Lint Check"))

    # Black formatting check
    cmd = ["black", "--check", "--diff", "hyena_glt", "tests"]
    checks.append((cmd, "Black Format Check"))

    # isort import sorting check
    cmd = ["isort", "--check-only", "--diff", "hyena_glt", "tests"]
    checks.append((cmd, "isort Import Check"))

    # MyPy type checking
    cmd = ["mypy", "hyena_glt", "--ignore-missing-imports"]
    checks.append((cmd, "MyPy Type Check"))

    all_passed = True
    for cmd, description in checks:
        if not run_command(cmd, description):
            all_passed = False

    return all_passed


def run_security_checks(args):
    """Run security checks."""
    checks = []

    # Safety check for known vulnerabilities
    cmd = ["safety", "check"]
    checks.append((cmd, "Safety Security Check"))

    # Bandit security linter
    cmd = ["bandit", "-r", "hyena_glt"]
    checks.append((cmd, "Bandit Security Check"))

    all_passed = True
    for cmd, description in checks:
        try:
            if not run_command(cmd, description):
                all_passed = False
        except FileNotFoundError:
            print(f"⚠️  {description} skipped - tool not installed")

    return all_passed


def generate_test_report(args, results):
    """Generate a test report."""
    report_file = args.report_file or "test_report.html"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Hyena-GLT Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #ccc; }}
        .pass {{ border-left-color: #4CAF50; }}
        .fail {{ border-left-color: #f44336; }}
        .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Hyena-GLT Test Report</h1>
        <p class="timestamp">Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <p>Total Tests Run: {len(results)}</p>
        <p>Passed: {sum(1 for r in results.values() if r)}</p>
        <p>Failed: {sum(1 for r in results.values() if not r)}</p>
    </div>
"""

    for test_name, passed in results.items():
        status_class = "pass" if passed else "fail"
        status_text = "✅ PASSED" if passed else "❌ FAILED"

        html_content += f"""
    <div class="section {status_class}">
        <h3>{test_name}</h3>
        <p><strong>Status:</strong> {status_text}</p>
    </div>
"""

    html_content += """
</body>
</html>
"""

    with open(report_file, "w") as f:
        f.write(html_content)

    print(f"\n📊 Test report generated: {report_file}")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Hyena-GLT Test Runner")

    # Test selection
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--gpu", action="store_true", help="Run GPU tests")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    parser.add_argument("--memory", action="store_true", help="Run memory tests")
    parser.add_argument(
        "--regression", action="store_true", help="Run regression tests"
    )
    parser.add_argument("--lint", action="store_true", help="Run lint checks")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--ci", action="store_true", help="Run CI test suite")

    # Test options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument(
        "--timeout", type=int, default=300, help="Test timeout in seconds"
    )
    parser.add_argument("--markers", help="Pytest markers to filter tests")

    # Reporting
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--report-file", help="Report file name")

    args = parser.parse_args()

    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    print(f"Running tests from: {os.getcwd()}")

    results = {}

    # Determine which tests to run
    if args.ci:
        # CI test suite
        test_functions = [
            ("Lint Checks", run_lint_checks),
            ("Unit Tests", run_unit_tests),
            ("Integration Tests", run_integration_tests),
            ("Security Checks", run_security_checks),
        ]
    elif args.all:
        # All tests
        test_functions = [
            ("Lint Checks", run_lint_checks),
            ("Unit Tests", run_unit_tests),
            ("Integration Tests", run_integration_tests),
            ("Benchmark Tests", run_benchmark_tests),
            ("GPU Tests", run_gpu_tests),
            ("Memory Tests", run_memory_tests),
            ("Regression Tests", run_regression_tests),
            ("Security Checks", run_security_checks),
        ]
    else:
        # Individual test selections
        test_functions = []
        if args.lint:
            test_functions.append(("Lint Checks", run_lint_checks))
        if args.unit:
            test_functions.append(("Unit Tests", run_unit_tests))
        if args.integration:
            test_functions.append(("Integration Tests", run_integration_tests))
        if args.benchmark:
            test_functions.append(("Benchmark Tests", run_benchmark_tests))
        if args.gpu:
            test_functions.append(("GPU Tests", run_gpu_tests))
        if args.smoke:
            test_functions.append(("Smoke Tests", run_smoke_tests))
        if args.memory:
            test_functions.append(("Memory Tests", run_memory_tests))
        if args.regression:
            test_functions.append(("Regression Tests", run_regression_tests))
        if args.security:
            test_functions.append(("Security Checks", run_security_checks))

        # Default to smoke tests if nothing specified
        if not test_functions:
            test_functions = [("Smoke Tests", run_smoke_tests)]

    # Run selected tests
    start_time = time.time()

    for test_name, test_function in test_functions:
        results[test_name] = test_function(args)

    end_time = time.time()
    total_time = end_time - start_time

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Tests run: {len(results)}")
    print(f"Passed: {sum(1 for r in results.values() if r)}")
    print(f"Failed: {sum(1 for r in results.values() if not r)}")

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")

    # Generate report if requested
    if args.report:
        generate_test_report(args, results)

    # Exit with appropriate code
    if all(results.values()):
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

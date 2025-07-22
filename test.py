# FILE: test.py
# ehsanasgharzde - COMPREHENSIVE TEST RUNNER SCRIPT
# hosseinsolymanzadeh - PROPER COMMENTING

import os
import sys
import subprocess
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
import smtplib
from email.mime.text import MimeText #type: ignore 
from email.mime.multipart import MimeMultipart #type: ignore 

logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    #Configuration for test execution
    test_directory: str = "tests"
    output_directory: str = "test_results"
    parallel_workers: int = 4
    timeout: int = 300  # 5 minutes per test file
    coverage_enabled: bool = True
    html_report: bool = True
    json_report: bool = True
    verbose: bool = True
    fail_fast: bool = False
    include_patterns: List[str] = field(default_factory=lambda: ["test_*.py", "*_test.py"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc"])
    email_notifications: bool = False
    email_config: Dict[str, str] = field(default_factory=dict)

@dataclass
class TestResult:
    # Test result data structure
    file_path: str
    framework: str
    status: str
    duration: float
    tests_run: int
    failures: int
    errors: int
    skipped: int
    output: str
    coverage_data: Optional[Dict] = None
    memory_usage: Optional[Dict] = None

class TestDiscovery:
    # Test discovery and classification
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_directory = Path(config.test_directory)
        
    def discover_tests(self) -> Dict[str, List[Path]]:
        """Discover all test files and classify by framework"""
        test_files = {"pytest": [], "unittest": [], "utility": []}
        
        if not self.test_directory.exists():
            logger.error(f"Test directory {self.test_directory} does not exist")
            return test_files
        
        for pattern in self.config.include_patterns:
            for test_file in self.test_directory.glob(pattern):
                if self._should_include_file(test_file):
                    framework = self._detect_framework(test_file)
                    test_files[framework].append(test_file)
        
        logger.info(f"Discovered tests: {len(test_files['pytest'])} pytest, "
                   f"{len(test_files['unittest'])} unittest, "
                   f"{len(test_files['utility'])} utility")
        
        return test_files
    
    def _should_include_file(self, file_path: Path) -> bool:
        # Check if file should be included based on patterns
        for pattern in self.config.exclude_patterns:
            if file_path.match(pattern):
                return False
        return True
    
    def _detect_framework(self, file_path: Path) -> str:
        # Detect test framework based on file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for unittest
            if ("import unittest" in content or 
                "from unittest" in content or 
                "unittest.TestCase" in content):
                return "unittest"
            
            # Check for pytest
            if ("import pytest" in content or 
                "from pytest" in content or 
                "@pytest" in content):
                return "pytest"
            
            # Check for utility functions (no formal test framework)
            if ("def test_" in content or 
                "def analyze_" in content or 
                "def benchmark_" in content):
                return "utility"
            
            # Default to pytest if contains test functions
            if "def test_" in content:
                return "pytest"
            
            return "utility"
            
        except Exception as e:
            logger.warning(f"Could not analyze {file_path}: {e}")
            return "pytest"  # Default fallback

class TestExecutor:
    #Test execution engine
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def run_all_tests(self, test_files: Dict[str, List[Path]]) -> List[TestResult]:
        # Run all discovered tests
        logger.info("Starting comprehensive test execution")
        
        # Prepare output directory
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(exist_ok=True)
        
        # Execute tests by framework
        if self.config.parallel_workers > 1:
            self._run_parallel_tests(test_files)
        else:
            self._run_sequential_tests(test_files)
        
        # Generate reports
        self._generate_reports()
        
        # Send notifications if configured
        if self.config.email_notifications:
            self._send_email_notification()
        
        return self.results
    
    def _run_parallel_tests(self, test_files: Dict[str, List[Path]]) -> None:
        # Run tests in parallel
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = []
            
            for framework, files in test_files.items():
                for test_file in files:
                    future = executor.submit(self._run_single_test, test_file, framework)
                    futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.config.timeout)
                    self.results.append(result)
                    
                    if result.status == "FAILED" and self.config.fail_fast:
                        logger.error(f"Test failed: {result.file_path}")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                        
                except Exception as e:
                    logger.error(f"Test execution error: {e}")
    
    def _run_sequential_tests(self, test_files: Dict[str, List[Path]]) -> None:
        # Run tests sequentially
        for framework, files in test_files.items():
            for test_file in files:
                try:
                    result = self._run_single_test(test_file, framework)
                    self.results.append(result)
                    
                    if result.status == "FAILED" and self.config.fail_fast:
                        logger.error(f"Test failed: {result.file_path}")
                        break
                        
                except Exception as e:
                    logger.error(f"Test execution error: {e}")
    
    def _run_single_test(self, test_file: Path, framework: str) -> TestResult:
        # Run a single test file
        logger.info(f"Running {framework} test: {test_file}")
        start_time = time.time()
        
        try:
            if framework == "pytest":
                result = self._run_pytest(test_file)
            elif framework == "unittest":
                result = self._run_unittest(test_file)
            else:
                result = self._run_utility_test(test_file)
            
            result.duration = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Error running {test_file}: {e}")
            return TestResult(
                file_path=str(test_file),
                framework=framework,
                status="ERROR",
                duration=time.time() - start_time,
                tests_run=0,
                failures=0,
                errors=1,
                skipped=0,
                output=str(e)
            )
    
    def _run_pytest(self, test_file: Path) -> TestResult:
        # Run pytest test file
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--junit-xml", f"{self.config.output_directory}/pytest_{test_file.stem}.xml"
        ]
        
        if self.config.coverage_enabled:
            cmd.extend(["--cov", "--cov-report", "json"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=os.getcwd()
            )
            
            return self._parse_pytest_output(test_file, result)
            
        except subprocess.TimeoutExpired:
            return TestResult(
                file_path=str(test_file),
                framework="pytest",
                status="TIMEOUT",
                duration=self.config.timeout,
                tests_run=0,
                failures=0,
                errors=1,
                skipped=0,
                output="Test execution timed out"
            )
    
    def _run_unittest(self, test_file: Path) -> TestResult:
        # Run unittest test file
        cmd = [
            sys.executable, "-m", "unittest",
            f"{test_file.stem}",
            "-v"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=str(test_file.parent)
            )
            
            return self._parse_unittest_output(test_file, result)
            
        except subprocess.TimeoutExpired:
            return TestResult(
                file_path=str(test_file),
                framework="unittest",
                status="TIMEOUT",
                duration=self.config.timeout,
                tests_run=0,
                failures=0,
                errors=1,
                skipped=0,
                output="Test execution timed out"
            )
    
    def _run_utility_test(self, test_file: Path) -> TestResult:
        # Run utility test file
        cmd = [sys.executable, str(test_file)]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=os.getcwd()
            )
            
            return TestResult(
                file_path=str(test_file),
                framework="utility",
                status="PASSED" if result.returncode == 0 else "FAILED",
                duration=0,
                tests_run=1,
                failures=0 if result.returncode == 0 else 1,
                errors=0,
                skipped=0,
                output=result.stdout + result.stderr
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                file_path=str(test_file),
                framework="utility",
                status="TIMEOUT",
                duration=self.config.timeout,
                tests_run=0,
                failures=0,
                errors=1,
                skipped=0,
                output="Test execution timed out"
            )
    
    def _parse_pytest_output(self, test_file: Path, result: subprocess.CompletedProcess) -> TestResult:
        # Parse pytest output and XML report
        xml_file = Path(f"{self.config.output_directory}/pytest_{test_file.stem}.xml")
        
        tests_run = 0
        failures = 0
        errors = 0
        skipped = 0
        
        if xml_file.exists():
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                tests_run = int(root.get('tests', 0))
                failures = int(root.get('failures', 0))
                errors = int(root.get('errors', 0))
                skipped = int(root.get('skipped', 0))
                
            except Exception as e:
                logger.warning(f"Could not parse XML report: {e}")
        
        status = "PASSED" if result.returncode == 0 else "FAILED"
        
        return TestResult(
            file_path=str(test_file),
            framework="pytest",
            status=status,
            duration=0,
            tests_run=tests_run,
            failures=failures,
            errors=errors,
            skipped=skipped,
            output=result.stdout + result.stderr
        )
    
    def _parse_unittest_output(self, test_file: Path, result: subprocess.CompletedProcess) -> TestResult:
        # Parse unittest output
        output = result.stdout + result.stderr
        
        # Simple parsing of unittest output
        tests_run = output.count('test_')
        failures = output.count('FAIL')
        errors = output.count('ERROR')
        skipped = output.count('SKIP')
        
        status = "PASSED" if result.returncode == 0 else "FAILED"
        
        return TestResult(
            file_path=str(test_file),
            framework="unittest",
            status=status,
            duration=0,
            tests_run=tests_run,
            failures=failures,
            errors=errors,
            skipped=skipped,
            output=output
        )
    
    def _generate_reports(self) -> None:
        # Generate comprehensive test reports
        if self.config.html_report:
            self._generate_html_report()
        
        if self.config.json_report:
            self._generate_json_report()
        
        self._generate_summary_report()
    
    def _generate_html_report(self) -> None:
        """Generate HTML test report"""
        html_content = self._create_html_report()
        
        report_file = Path(f"{self.config.output_directory}/test_report.html")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_file}")
    
    def _generate_json_report(self) -> None:
        # Generate JSON test report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": time.time() - self.start_time,
            "summary": self._get_summary_stats(),
            "results": [
                {
                    "file_path": r.file_path,
                    "framework": r.framework,
                    "status": r.status,
                    "duration": r.duration,
                    "tests_run": r.tests_run,
                    "failures": r.failures,
                    "errors": r.errors,
                    "skipped": r.skipped
                }
                for r in self.results
            ]
        }
        
        report_file = Path(f"{self.config.output_directory}/test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"JSON report generated: {report_file}")
    
    def _generate_summary_report(self) -> None:
        # Generate summary report to console
        stats = self._get_summary_stats()
        
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        print(f"Total Duration: {stats['total_duration']:.2f}s")
        print(f"Files Tested: {stats['files_tested']}")
        print(f"Tests Run: {stats['tests_run']}")
        print(f"Passed: {stats['passed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Errors: {stats['errors']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print("="*60)
        
        if stats['failed'] > 0 or stats['errors'] > 0:
            print("\nFAILED TESTS:")
            for result in self.results:
                if result.status in ["FAILED", "ERROR", "TIMEOUT"]:
                    print(f"  - {result.file_path} ({result.framework})")
    
    def _get_summary_stats(self) -> Dict[str, Any]:
        # Calculate summary statistics
        total_tests = sum(r.tests_run for r in self.results)
        total_failures = sum(r.failures for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        
        passed = total_tests - total_failures - total_errors - total_skipped
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        return {
            "total_duration": time.time() - self.start_time,
            "files_tested": len(self.results),
            "tests_run": total_tests,
            "passed": passed,
            "failed": total_failures,
            "errors": total_errors,
            "skipped": total_skipped,
            "success_rate": success_rate
        }
    
    def _create_html_report(self) -> str:
        # Create HTML report content
        stats = self._get_summary_stats()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Execution Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                .skipped {{ color: blue; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Test Execution Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Duration: {stats['total_duration']:.2f}s</p>
                <p>Files Tested: {stats['files_tested']}</p>
                <p>Tests Run: {stats['tests_run']}</p>
                <p>Success Rate: {stats['success_rate']:.1f}%</p>
            </div>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>File</th>
                    <th>Framework</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Tests</th>
                    <th>Failures</th>
                    <th>Errors</th>
                    <th>Skipped</th>
                </tr>
        """
        
        for result in self.results:
            status_class = result.status.lower()
            html += f"""
                <tr>
                    <td>{result.file_path}</td>
                    <td>{result.framework}</td>
                    <td class="{status_class}">{result.status}</td>
                    <td>{result.duration:.2f}s</td>
                    <td>{result.tests_run}</td>
                    <td>{result.failures}</td>
                    <td>{result.errors}</td>
                    <td>{result.skipped}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
    
    def _send_email_notification(self) -> None:
        # Send email notification with test results
        if not self.config.email_config:
            return
        
        try:
            stats = self._get_summary_stats()
            
            msg = MimeMultipart()
            msg['From'] = self.config.email_config.get('sender')
            msg['To'] = self.config.email_config.get('recipient')
            msg['Subject'] = f"Test Results - {stats['success_rate']:.1f}% Pass Rate"
            
            body = f"""
            Test Execution Summary:
            
            Total Duration: {stats['total_duration']:.2f}s
            Files Tested: {stats['files_tested']}
            Tests Run: {stats['tests_run']}
            Passed: {stats['passed']}
            Failed: {stats['failed']}
            Errors: {stats['errors']}
            Skipped: {stats['skipped']}
            Success Rate: {stats['success_rate']:.1f}%
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.email_config.get('smtp_server')) #type: ignore 
            server.starttls()
            server.login(
                self.config.email_config.get('username'), #type: ignore 
                self.config.email_config.get('password') #type: ignore 
            )
            server.send_message(msg)
            server.quit()
            
            logger.info("Email notification sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

def main():
    # Main entry point
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    parser.add_argument("--test-dir", default="tests", help="Test directory")
    parser.add_argument("--output-dir", default="test_results", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage")
    parser.add_argument("--no-html", action="store_true", help="Disable HTML report")
    parser.add_argument("--no-json", action="store_true", help="Disable JSON report")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--email", action="store_true", help="Enable email notifications")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = TestConfig(
        test_directory=args.test_dir,
        output_directory=args.output_dir,
        parallel_workers=args.workers,
        timeout=args.timeout,
        coverage_enabled=not args.no_coverage,
        html_report=not args.no_html,
        json_report=not args.no_json,
        fail_fast=args.fail_fast,
        email_notifications=args.email
    )
    
    if args.config:
        # Load configuration from file
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                # Update config with loaded data
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            sys.exit(1)
    
    # Initialize components
    discovery = TestDiscovery(config)
    executor = TestExecutor(config)
    
    # Discover and run tests
    test_files = discovery.discover_tests()
    results = executor.run_all_tests(test_files)
    
    # Exit with appropriate code
    failed_tests = sum(1 for r in results if r.status in ["FAILED", "ERROR", "TIMEOUT"])
    sys.exit(1 if failed_tests > 0 else 0)

if __name__ == "__main__":
    main()
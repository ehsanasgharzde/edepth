# FILE: test.py
# ehsanasgharzde - COMPREHENSIVE TEST RUNNER SCRIPT

import os
import sys
import subprocess
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback
import signal

from configs.config import TestConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
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
    performance_data: Optional[Dict] = None
    retry_count: int = 0
    exception_info: Optional[str] = None

class ComponentTestDiscovery:
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_directory = Path(config.test_directory)
        
    def discover_tests(self) -> Dict[str, List[Path]]:
        test_files = {"pytest": [], "unittest": [], "mixed": [], "utility": []}
        
        if not self.test_directory.exists():
            logger.error(f"Test directory {self.test_directory} does not exist")
            return test_files
        
        self._discover_main_tests(test_files)
        self._discover_component_tests(test_files)
        
        logger.info(f"Discovered tests: {len(test_files['pytest'])} pytest, "
                   f"{len(test_files['unittest'])} unittest, "
                   f"{len(test_files['mixed'])} mixed, "
                   f"{len(test_files['utility'])} utility")
        
        return test_files
    
    def _discover_main_tests(self, test_files: Dict[str, List[Path]]):
        for pattern in self.config.include_patterns:
            for test_file in self.test_directory.glob(pattern):
                if self._should_include_file(test_file):
                    framework = self._detect_framework(test_file)
                    test_files[framework].append(test_file)
    
    def _discover_component_tests(self, test_files: Dict[str, List[Path]]):
        for component_dir in self.config.component_dirs:
            component_path = self.test_directory / component_dir
            if component_path.exists():
                for pattern in self.config.include_patterns:
                    for test_file in component_path.glob(pattern):
                        if self._should_include_file(test_file):
                            framework = self._detect_framework(test_file)
                            test_files[framework].append(test_file)
    
    def _should_include_file(self, file_path: Path) -> bool:
        for pattern in self.config.exclude_patterns:
            if file_path.match(pattern):
                return False
        return file_path.is_file() and file_path.suffix == '.py'
    
    def _detect_framework(self, file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            has_unittest = any(marker in content for marker in [
                "import unittest", "from unittest", "unittest.TestCase", "TestCase"
            ])
            
            has_pytest = any(marker in content for marker in [
                "import pytest", "from pytest", "@pytest", "pytest.mark"
            ])
            
            if has_unittest and has_pytest:
                return "mixed"
            elif has_unittest:
                return "unittest"
            elif has_pytest or "def test_" in content:
                return "pytest"
            else:
                return "utility"
                
        except Exception as e:
            logger.warning(f"Could not analyze {file_path}: {e}")
            return "pytest"

class TestExecutor:
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.start_time = time.time()
        self.failed_tests: List[Path] = []
        
    def run_all_tests(self, test_files: Dict[str, List[Path]]) -> List[TestResult]:
        logger.info("Starting comprehensive test execution")
        
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(exist_ok=True)
        
        try:
            if self.config.parallel_workers > 1:
                self._run_parallel_tests(test_files)
            else:
                self._run_sequential_tests(test_files)
            
            if self.config.retry_failed > 0 and self.failed_tests:
                self._retry_failed_tests()
            
            self._generate_reports()
            
            if self.config.email_notifications:
                self._send_email_notification()
        finally:
            pass
        
        return self.results
    
    def _run_parallel_tests(self, test_files: Dict[str, List[Path]]):
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = []
            
            for framework, files in test_files.items():
                for test_file in files:
                    if self.config.test_isolation:
                        future = executor.submit(self._run_isolated_test, test_file, framework)
                    else:
                        future = executor.submit(self._run_single_test, test_file, framework)
                    futures.append((future, test_file))
            
            for future, test_file in futures:
                try:
                    result = future.result(timeout=self.config.timeout)
                    self.results.append(result)
                    
                    if result.status in ["FAILED", "ERROR"]:
                        self.failed_tests.append(test_file)
                    
                    if result.status == "FAILED" and self.config.fail_fast:
                        logger.error(f"Test failed: {result.file_path}")
                        for f, _ in futures:
                            f.cancel()
                        break
                        
                except Exception as e:
                    logger.error(f"Test execution error for {test_file}: {e}")
                    self._create_error_result(test_file, str(e))
    
    def _run_sequential_tests(self, test_files: Dict[str, List[Path]]):
        for framework, files in test_files.items():
            for test_file in files:
                try:
                    if self.config.test_isolation:
                        result = self._run_isolated_test(test_file, framework)
                    else:
                        result = self._run_single_test(test_file, framework)
                    
                    self.results.append(result)
                    
                    if result.status in ["FAILED", "ERROR"]:
                        self.failed_tests.append(test_file)
                    
                    if result.status == "FAILED" and self.config.fail_fast:
                        logger.error(f"Test failed: {result.file_path}")
                        break
                        
                except Exception as e:
                    logger.error(f"Test execution error for {test_file}: {e}")
                    self._create_error_result(test_file, str(e))
    
    def _run_isolated_test(self, test_file: Path, framework: str) -> TestResult:
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        if framework == "pytest":
            return self._run_pytest_isolated(test_file, env)
        elif framework == "unittest":
            return self._run_unittest_isolated(test_file, env)
        elif framework == "mixed":
            return self._run_mixed_test(test_file, env)
        else:
            return self._run_utility_test(test_file, env)
    
    def _run_single_test(self, test_file: Path, framework: str) -> TestResult:
        logger.info(f"Running {framework} test: {test_file}")
        start_time = time.time()
        
        try:
            if framework == "pytest":
                result = self._run_pytest(test_file)
            elif framework == "unittest":
                result = self._run_unittest(test_file)
            elif framework == "mixed":
                result = self._run_mixed_test(test_file)
            else:
                result = self._run_utility_test(test_file)
            
            result.duration = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Error running {test_file}: {e}")
            return self._create_error_result(test_file, str(e), time.time() - start_time)
    
    def _run_pytest_isolated(self, test_file: Path, env: Dict[str, str]) -> TestResult:
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v", "--tb=short",
            f"--junit-xml={self.config.output_directory}/pytest_{test_file.stem}.xml"
        ]
        
        if self.config.coverage_enabled:
            cmd.extend([
                "--cov", "--cov-report", "json",
                f"--cov-report=json:{self.config.output_directory}/coverage_{test_file.stem}.json"
            ])
        
        return self._execute_subprocess(cmd, test_file, "pytest", env)
    
    def _run_pytest(self, test_file: Path) -> TestResult:
        return self._run_pytest_isolated(test_file, os.environ.copy())
    
    def _run_unittest_isolated(self, test_file: Path, env: Dict[str, str]) -> TestResult:
        cmd = [
            sys.executable, "-m", "unittest",
            f"{test_file.parent.name}.{test_file.stem}",
            "-v"
        ]
        
        return self._execute_subprocess(cmd, test_file, "unittest", env)
    
    def _run_unittest(self, test_file: Path) -> TestResult:
        return self._run_unittest_isolated(test_file, os.environ.copy())
    
    def _run_mixed_test(self, test_file: Path, env: Optional[Dict[str, str]] = None) -> TestResult:
        if env is None:
            env = os.environ.copy()
            
        pytest_result = self._run_pytest_isolated(test_file, env)
        
        if pytest_result.status == "PASSED":
            return pytest_result
        
        unittest_result = self._run_unittest_isolated(test_file, env)
        
        if unittest_result.status == "PASSED":
            return unittest_result
        
        return pytest_result if pytest_result.tests_run > unittest_result.tests_run else unittest_result
    
    def _run_utility_test(self, test_file: Path, env: Optional[Dict[str, str]] = None) -> TestResult:
        if env is None:
            env = os.environ.copy()
            
        cmd = [sys.executable, str(test_file)]
        return self._execute_subprocess(cmd, test_file, "utility", env)
    
    def _execute_subprocess(self, cmd: List[str], test_file: Path, framework: str, env: Dict[str, str]) -> TestResult:
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                env=env,
                cwd=os.getcwd()
            )
            
            duration = time.time() - start_time
            
            if framework == "pytest":
                return self._parse_pytest_output(test_file, result, duration)
            elif framework == "unittest":
                return self._parse_unittest_output(test_file, result, duration)
            else:
                return self._parse_utility_output(test_file, result, duration)
                
        except subprocess.TimeoutExpired:
            return TestResult(
                file_path=str(test_file),
                framework=framework,
                status="TIMEOUT",
                duration=self.config.timeout,
                tests_run=0,
                failures=0,
                errors=1,
                skipped=0,
                output="Test execution timed out"
            )
        except Exception as e:
            return self._create_error_result(test_file, str(e), time.time() - start_time)
    
    def _parse_pytest_output(self, test_file: Path, result: subprocess.CompletedProcess, duration: float) -> TestResult:
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
            duration=duration,
            tests_run=tests_run,
            failures=failures,
            errors=errors,
            skipped=skipped,
            output=result.stdout + result.stderr
        )
    
    def _parse_unittest_output(self, test_file: Path, result: subprocess.CompletedProcess, duration: float) -> TestResult:
        output = result.stdout + result.stderr
        
        import re
        test_pattern = r'test_\w+'
        tests_run = len(re.findall(test_pattern, output))
        
        failures = output.count('FAIL')
        errors = output.count('ERROR')
        skipped = output.count('SKIP')
        
        status = "PASSED" if result.returncode == 0 else "FAILED"
        
        return TestResult(
            file_path=str(test_file),
            framework="unittest",
            status=status,
            duration=duration,
            tests_run=tests_run,
            failures=failures,
            errors=errors,
            skipped=skipped,
            output=output
        )
    
    def _parse_utility_output(self, test_file: Path, result: subprocess.CompletedProcess, duration: float) -> TestResult:
        status = "PASSED" if result.returncode == 0 else "FAILED"
        
        return TestResult(
            file_path=str(test_file),
            framework="utility",
            status=status,
            duration=duration,
            tests_run=1,
            failures=0 if result.returncode == 0 else 1,
            errors=0,
            skipped=0,
            output=result.stdout + result.stderr
        )
    
    def _create_error_result(self, test_file: Path, error: str, duration: float = 0) -> TestResult:
        return TestResult(
            file_path=str(test_file),
            framework="unknown",
            status="ERROR",
            duration=duration,
            tests_run=0,
            failures=0,
            errors=1,
            skipped=0,
            output=error,
            exception_info=error
        )
    
    def _retry_failed_tests(self):
        logger.info(f"Retrying {len(self.failed_tests)} failed tests")
        
        retry_results = []
        for test_file in self.failed_tests:
            for attempt in range(self.config.retry_failed):
                logger.info(f"Retry {attempt + 1}/{self.config.retry_failed} for {test_file}")
                
                original_result = next((r for r in self.results if Path(r.file_path) == test_file), None)
                if not original_result:
                    continue
                
                retry_result = self._run_single_test(test_file, original_result.framework)
                retry_result.retry_count = attempt + 1
                retry_results.append(retry_result)
                
                if retry_result.status == "PASSED":
                    logger.info(f"Test {test_file} passed on retry {attempt + 1}")
                    break
        
        self.results.extend(retry_results)
    
    def _generate_reports(self):
        if self.config.html_report:
            self._generate_html_report()
        
        if self.config.json_report:
            self._generate_json_report()
        
        self._generate_summary_report()
        self._generate_component_report()
    
    def _generate_component_report(self):
        component_stats = {}
        
        for result in self.results:
            file_path = Path(result.file_path)
            if len(file_path.parts) > 1 and file_path.parts[-2] in self.config.component_dirs:
                component = file_path.parts[-2]
            else:
                component = "main"
            
            if component not in component_stats:
                component_stats[component] = {
                    'tests_run': 0, 'failures': 0, 'errors': 0, 'skipped': 0,
                    'duration': 0, 'files': 0
                }
            
            stats = component_stats[component]
            stats['tests_run'] += result.tests_run
            stats['failures'] += result.failures
            stats['errors'] += result.errors
            stats['skipped'] += result.skipped
            stats['duration'] += result.duration
            stats['files'] += 1
        
        report_file = Path(f"{self.config.output_directory}/component_report.json")
        with open(report_file, 'w') as f:
            json.dump(component_stats, f, indent=2)
        
        logger.info(f"Component report generated: {report_file}")
    
    def _generate_html_report(self):
        html_content = self._create_html_report()
        
        report_file = Path(f"{self.config.output_directory}/test_report.html")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_file}")
    
    def _generate_json_report(self):

        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": time.time() - self.start_time,
            "summary": self._get_summary_stats(),
            "memory_stats": None,
            "results": [
                {
                    "file_path": r.file_path,
                    "framework": r.framework,
                    "status": r.status,
                    "duration": r.duration,
                    "tests_run": r.tests_run,
                    "failures": r.failures,
                    "errors": r.errors,
                    "skipped": r.skipped,
                    "retry_count": r.retry_count
                }
                for r in self.results
            ]
        }
        
        report_file = Path(f"{self.config.output_directory}/test_report.json")
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"JSON report generated: {report_file}")
    
    def _generate_summary_report(self):
        stats = self._get_summary_stats()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST EXECUTION SUMMARY")
        print("="*80)
        print(f"Total Duration: {stats['total_duration']:.2f}s")
        print(f"Files Tested: {stats['files_tested']}")
        print(f"Tests Run: {stats['tests_run']}")
        print(f"Passed: {stats['passed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Errors: {stats['errors']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        
        print("\nFramework Breakdown:")
        for framework, count in stats['framework_breakdown'].items():
            print(f"  {framework}: {count} files")
        
        print("="*80)
        
        if stats['failed'] > 0 or stats['errors'] > 0:
            print("\nFAILED/ERROR TESTS:")
            for result in self.results:
                if result.status in ["FAILED", "ERROR", "TIMEOUT"]:
                    print(f"  - {result.file_path} ({result.framework}) - {result.status}")
                    if result.retry_count > 0:
                        print(f"    (Retried {result.retry_count} times)")
    
    def _get_summary_stats(self) -> Dict[str, Any]:
        total_tests = sum(r.tests_run for r in self.results)
        total_failures = sum(r.failures for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        
        passed = total_tests - total_failures - total_errors - total_skipped
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        framework_breakdown = {}
        for result in self.results:
            framework = result.framework
            framework_breakdown[framework] = framework_breakdown.get(framework, 0) + 1
        
        return {
            "total_duration": time.time() - self.start_time,
            "files_tested": len(self.results),
            "tests_run": total_tests,
            "passed": passed,
            "failed": total_failures,
            "errors": total_errors,
            "skipped": total_skipped,
            "success_rate": success_rate,
            "framework_breakdown": framework_breakdown
        }
    
    def _create_html_report(self) -> str:
        stats = self._get_summary_stats()
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background: #f0f0f0; padding: 20px; margin-bottom: 20px; border-radius: 5px; }
                .passed { color: green; font-weight: bold; }
                .failed { color: red; font-weight: bold; }
                .error { color: orange; font-weight: bold; }
                .timeout { color: purple; font-weight: bold; }
                .skipped { color: blue; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
                .stat-box { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
            </style>
        </head>
        <body>
            <h1>Comprehensive Test Execution Report</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <h3>Duration</h3>
                        <p>{total_duration:.2f}s</p>
                    </div>
                    <div class="stat-box">
                        <h3>Files Tested</h3>
                        <p>{files_tested}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Tests Run</h3>
                        <p>{tests_run}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Success Rate</h3>
                        <p>{success_rate:.1f}%</p>
                    </div>
                </div>
                {memory_section}
            </div>
            
            <h2>Framework Breakdown</h2>
            <table>
                <tr><th>Framework</th><th>Files</th></tr>
                {framework_rows}
            </table>
            
            <h2>Detailed Test Results</h2>
            <table>
                <tr>
                    <th>File</th><th>Framework</th><th>Status</th><th>Duration</th>
                    <th>Tests</th><th>Failures</th><th>Errors</th><th>Skipped</th><th>Retries</th>
                </tr>
                {result_rows}
            </table>
        </body>
        </html>
        """
        
        framework_rows = ""
        for framework, count in stats['framework_breakdown'].items():
            framework_rows += f"<tr><td>{framework}</td><td>{count}</td></tr>"
        
        result_rows = ""
        for result in self.results:
            status_class = result.status.lower()
            result_rows += f"""
                <tr>
                    <td>{result.file_path}</td>
                    <td>{result.framework}</td>
                    <td class="{status_class}">{result.status}</td>
                    <td>{result.duration:.2f}s</td>
                    <td>{result.tests_run}</td>
                    <td>{result.failures}</td>
                    <td>{result.errors}</td>
                    <td>{result.skipped}</td>
                    <td>{result.retry_count}</td>
                </tr>
            """
        
        return html_template.format(
            total_duration=stats['total_duration'],
            files_tested=stats['files_tested'],
            tests_run=stats['tests_run'],
            success_rate=stats['success_rate'],
            framework_rows=framework_rows,
            result_rows=result_rows
        )
    
    def _send_email_notification(self):
        if not self.config.email_config:
            return
        
        try:
            stats = self._get_summary_stats()
            
            msg = MIMEMultipart()
            msg['From'] = self.config.email_config.get('sender') # type: ignore
            msg['To'] = self.config.email_config.get('recipient') # type: ignore
            msg['Subject'] = f"Test Results - {stats['success_rate']:.1f}% Pass Rate"
            
            body = f"""
            Comprehensive Test Execution Summary:
            
            Total Duration: {stats['total_duration']:.2f}s
            Files Tested: {stats['files_tested']}
            Tests Run: {stats['tests_run']}
            Passed: {stats['passed']}
            Failed: {stats['failed']}
            Errors: {stats['errors']}
            Skipped: {stats['skipped']}
            Success Rate: {stats['success_rate']:.1f}%
            
            Framework Breakdown:
            {chr(10).join(f'  {k}: {v}' for k, v in stats['framework_breakdown'].items())}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.email_config.get('smtp_server')) # type: ignore
            server.starttls()
            server.login(
                self.config.email_config.get('username'), # type: ignore
                self.config.email_config.get('password') # type: ignore
            )
            server.send_message(msg)
            server.quit()
            
            logger.info("Email notification sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

def signal_handler(signum, frame):
    logger.info("Received interrupt signal, stopping tests...")
    sys.exit(1)
# FILE: tests/test.py
# ehsanasgharzde - COMPLETE TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from logger.logger import setup_logging 

# Setup logger for factory operations
logger = setup_logging(__file__)

def discover_pytest_tests(test_dir: str, include_patterns: List[str], exclude_patterns: List[str], component_dirs: List[str]) -> List[Path]:
    test_files = []
    test_directory = Path(test_dir)

    if not test_directory.exists():
        logger.error(f"Test directory {test_directory} does not exist")
        return test_files

    def should_include(file_path: Path) -> bool:
        return all(not file_path.match(pat) for pat in exclude_patterns) and file_path.is_file() and file_path.suffix == '.py'

    for pattern in include_patterns:
        test_files.extend([f for f in test_directory.glob(pattern) if should_include(f)])

    for component_dir in component_dirs:
        component_path = test_directory / component_dir
        if component_path.exists():
            for pattern in include_patterns:
                test_files.extend([f for f in component_path.glob(pattern) if should_include(f)])

    logger.info(f"Discovered {len(test_files)} pytest test files")
    return test_files

def run_pytest_tests(test_files: List[Path], output_dir: str, parallel: int = 1, timeout: int = 300, coverage: bool = False) -> List[Dict[str, Any]]:
    results = []
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pytest", "-v",
        f"--junit-xml={output_dir}/pytest_results.xml",
    ]

    if parallel > 1:
        cmd.extend(["-n", str(parallel)])

    if coverage:
        cmd.extend([
            "--cov", ".",
            "--cov-report", f"html:{output_dir}/coverage_html",
            f"--cov-report=xml:{output_dir}/coverage.xml"
        ])

    cmd.extend([str(f) for f in test_files])

    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=os.getcwd())
        xml_path = Path(f"{output_dir}/pytest_results.xml")
        test_data = parse_pytest_xml_results(xml_path) if xml_path.exists() else []
        results.append({
            'framework': 'pytest',
            'execution_time': time.time() - start_time,
            'return_code': result.returncode,
            'output': result.stdout + result.stderr,
            'tests': test_data
        })
    except subprocess.TimeoutExpired:
        results.append({
            'framework': 'pytest',
            'execution_time': timeout,
            'return_code': -1,
            'output': "Test execution timed out",
            'tests': []
        })

    return results

def parse_pytest_xml_results(xml_path: Path) -> List[Dict[str, Any]]:
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        results = []
        for testcase in root.findall('.//testcase'):
            test = {
                'name': testcase.get('name', ''),
                'classname': testcase.get('classname', ''),
                'file': testcase.get('file', ''),
                'line': testcase.get('line', ''),
                'time': float(testcase.get('time', 0)),
                'status': 'passed'
            }
            if testcase.find('failure') is not None:
                test['status'] = 'failed'
                test['message'] = testcase.find('failure').get('message', '')
                test['traceback'] = testcase.find('failure').text
            elif testcase.find('error') is not None:
                test['status'] = 'error'
                test['message'] = testcase.find('error').get('message', '')
                test['traceback'] = testcase.find('error').text
            elif testcase.find('skipped') is not None:
                test['status'] = 'skipped'
                test['message'] = testcase.find('skipped').get('message', '')
            results.append(test)
        return results
    except Exception as e:
        logger.error(f"Failed to parse XML results: {e}")
        return []

def print_summary(test_results: List[Dict[str, Any]], start_time: float) -> None:
    stats = {'total': 0, 'passed': 0, 'failed': 0, 'error': 0, 'skipped': 0}
    for group in test_results:
        for test in group.get('tests', []):
            stats['total'] += 1
            stats[test['status']] += 1

    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)
    print(f"Duration: {time.time() - start_time:.2f}s")
    for key in ['total', 'passed', 'failed', 'error', 'skipped']:
        print(f"{key.capitalize()}: {stats[key]}")
    print("="*80)

def run_tests(args: argparse.Namespace) -> None:
    start_time = time.time()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    include_patterns = ["test_*.py"]
    exclude_patterns = ["*_helper.py", "*_utilities.py"]
    component_dirs = ["models", "datasets", "training", "utils"]

    test_files = discover_pytest_tests(args.test_dir, include_patterns, exclude_patterns, component_dirs)

    if args.test_file:
        test_files = [f for f in test_files if args.test_file in str(f)]

    if args.component:
        test_files = [f for f in test_files if args.component in str(f)]

    test_results = run_pytest_tests(test_files, output_dir, parallel=args.parallel, timeout=args.timeout, coverage=args.coverage)
    print_summary(test_results, start_time)

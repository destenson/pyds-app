#!/usr/bin/env python3
"""
Coverage analysis and dead code correlation script for PyDS application.

This script combines test coverage data with dead code detection to identify
high-confidence dead code, track coverage trends over time, and provide
actionable recommendations for code cleanup.

Usage:
    python scripts/coverage-analysis.py --run-tests
    python scripts/coverage-analysis.py --correlate-dead-code
    python scripts/coverage-analysis.py --trend-analysis
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import sqlite3
import xml.etree.ElementTree as ET

# Import PyDS modules if available
try:
    from src.utils.logging import get_logger
    from src.utils.errors import PyDSError
    HAS_PYDS_MODULES = True
except ImportError:
    HAS_PYDS_MODULES = False
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


@dataclass
class CoverageFile:
    """File coverage information."""
    file_path: str
    statements: int
    missing: int
    excluded: int
    coverage_percent: float
    missing_lines: List[int] = field(default_factory=list)
    branches_total: Optional[int] = None
    branches_missing: Optional[int] = None
    branch_coverage_percent: Optional[float] = None


@dataclass
class CoverageReport:
    """Complete coverage analysis report."""
    timestamp: str
    total_statements: int
    total_missing: int
    overall_coverage: float
    branch_coverage: Optional[float]
    files: List[CoverageFile]
    uncovered_functions: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeadCodeCandidate:
    """Dead code candidate information."""
    file_path: str
    line: int
    function_name: Optional[str]
    confidence: float
    reason: str
    coverage_data: Optional[Dict[str, Any]] = None


@dataclass
class CoverageCorrelationReport:
    """Report correlating coverage with dead code detection."""
    timestamp: str
    coverage_report: CoverageReport
    dead_code_candidates: List[DeadCodeCandidate]
    high_confidence_dead_code: List[DeadCodeCandidate]
    recommendations: List[str] = field(default_factory=list)
    cleanup_estimate: Dict[str, Any] = field(default_factory=dict)


class CoverageDatabase:
    """SQLite database for tracking coverage trends over time."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the coverage database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS coverage_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    overall_coverage REAL NOT NULL,
                    branch_coverage REAL,
                    total_statements INTEGER NOT NULL,
                    total_missing INTEGER NOT NULL,
                    commit_hash TEXT,
                    branch_name TEXT
                );
                
                CREATE TABLE IF NOT EXISTS file_coverage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    statements INTEGER NOT NULL,
                    missing INTEGER NOT NULL,
                    coverage_percent REAL NOT NULL,
                    missing_lines TEXT,
                    FOREIGN KEY (run_id) REFERENCES coverage_runs (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_coverage_runs_timestamp 
                ON coverage_runs(timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_file_coverage_file_path 
                ON file_coverage(file_path);
            """)
    
    def store_coverage_run(self, report: CoverageReport) -> int:
        """Store a coverage run in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get git info if available
            commit_hash, branch_name = self._get_git_info()
            
            # Insert coverage run
            cursor.execute("""
                INSERT INTO coverage_runs 
                (timestamp, overall_coverage, branch_coverage, total_statements, 
                 total_missing, commit_hash, branch_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                report.timestamp,
                report.overall_coverage,
                report.branch_coverage,
                report.total_statements,
                report.total_missing,
                commit_hash,
                branch_name
            ))
            
            run_id = cursor.lastrowid
            
            # Insert file coverage data
            for file_cov in report.files:
                cursor.execute("""
                    INSERT INTO file_coverage 
                    (run_id, file_path, statements, missing, coverage_percent, missing_lines)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    file_cov.file_path,
                    file_cov.statements,
                    file_cov.missing,
                    file_cov.coverage_percent,
                    json.dumps(file_cov.missing_lines)
                ))
            
            return run_id
    
    def get_coverage_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get coverage trend over the specified number of days."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT timestamp, overall_coverage, branch_coverage, 
                       total_statements, total_missing, commit_hash
                FROM coverage_runs 
                WHERE timestamp >= ?
                ORDER BY timestamp
            """, (cutoff_date,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_file_coverage_history(self, file_path: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get coverage history for a specific file."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT cr.timestamp, fc.coverage_percent, fc.statements, 
                       fc.missing, fc.missing_lines
                FROM file_coverage fc
                JOIN coverage_runs cr ON fc.run_id = cr.id
                WHERE fc.file_path = ? AND cr.timestamp >= ?
                ORDER BY cr.timestamp
            """, (file_path, cutoff_date))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def _get_git_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Get current git commit hash and branch name."""
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10
            )
            commit_hash = result.stdout.strip() if result.returncode == 0 else None
            
            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10
            )
            branch_name = result.stdout.strip() if result.returncode == 0 else None
            
            return commit_hash, branch_name
        except Exception:
            return None, None


class CoverageAnalyzer:
    """Main coverage analysis orchestrator."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = working_dir or Path.cwd()
        self.logger = get_logger(__name__) if HAS_PYDS_MODULES else logger
        self.coverage_db = CoverageDatabase(self.working_dir / "analysis" / "coverage" / "coverage.db")
    
    async def run_tests_with_coverage(self) -> CoverageReport:
        """Run tests with coverage analysis."""
        self.logger.info("Running tests with coverage analysis...")
        
        try:
            # Run pytest with coverage
            cmd = [
                "pytest",
                "--cov=src",
                "--cov-report=xml:analysis/coverage/reports/coverage.xml",
                "--cov-report=html:analysis/coverage/reports/htmlcov",
                "--cov-report=json:analysis/coverage/reports/coverage.json",
                "--cov-branch",
                "tests/"
            ]
            
            result = await self._run_command(cmd)
            
            if result["returncode"] != 0:
                self.logger.warning(f"Tests failed with exit code {result['returncode']}")
            
            # Parse coverage results
            coverage_report = self._parse_coverage_results()
            
            # Store in database
            self.coverage_db.store_coverage_run(coverage_report)
            
            return coverage_report
            
        except Exception as e:
            self.logger.error(f"Coverage analysis failed: {e}")
            raise
    
    def _parse_coverage_results(self) -> CoverageReport:
        """Parse coverage results from JSON and XML reports."""
        json_path = self.working_dir / "analysis" / "coverage" / "reports" / "coverage.json"
        xml_path = self.working_dir / "analysis" / "coverage" / "reports" / "coverage.xml"
        
        # Parse JSON coverage data
        coverage_data = {}
        if json_path.exists():
            with open(json_path, 'r') as f:
                coverage_data = json.load(f)
        
        # Parse XML for additional branch coverage data
        branch_data = {}
        if xml_path.exists():
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for package in root.findall('.//package'):
                for class_elem in package.findall('classes/class'):
                    filename = class_elem.get('filename')
                    if filename:
                        lines = class_elem.find('lines')
                        if lines is not None:
                            branch_info = self._extract_branch_info(lines)
                            if branch_info:
                                branch_data[filename] = branch_info
        
        # Create coverage report
        files = []
        total_statements = 0
        total_missing = 0
        
        for file_path, file_data in coverage_data.get('files', {}).items():
            summary = file_data.get('summary', {})
            statements = summary.get('num_statements', 0)
            missing = summary.get('missing_lines', 0)
            coverage_percent = summary.get('percent_covered', 0.0)
            
            missing_lines = []
            if isinstance(missing, list):
                missing_lines = missing
                missing = len(missing_lines)
            
            # Get branch coverage if available
            branch_info = branch_data.get(file_path, {})
            
            file_coverage = CoverageFile(
                file_path=file_path,
                statements=statements,
                missing=missing,
                excluded=summary.get('excluded_lines', 0),
                coverage_percent=coverage_percent,
                missing_lines=missing_lines,
                branches_total=branch_info.get('total_branches'),
                branches_missing=branch_info.get('missing_branches'),
                branch_coverage_percent=branch_info.get('branch_coverage')
            )
            
            files.append(file_coverage)
            total_statements += statements
            total_missing += missing
        
        overall_coverage = 0.0
        if total_statements > 0:
            overall_coverage = ((total_statements - total_missing) / total_statements) * 100
        
        # Get overall branch coverage
        totals = coverage_data.get('totals', {})
        branch_coverage = None
        if 'percent_covered_display' in totals:
            # Try to extract branch coverage from display string
            display = totals['percent_covered_display']
            if '/' in display:
                parts = display.split('/')
                if len(parts) == 2:
                    try:
                        branch_coverage = float(parts[1].strip('%'))
                    except ValueError:
                        pass
        
        return CoverageReport(
            timestamp=datetime.now().isoformat(),
            total_statements=total_statements,
            total_missing=total_missing,
            overall_coverage=overall_coverage,
            branch_coverage=branch_coverage,
            files=files,
            summary={
                "files_with_issues": len([f for f in files if f.coverage_percent < 90]),
                "fully_covered_files": len([f for f in files if f.coverage_percent == 100]),
                "average_coverage": sum(f.coverage_percent for f in files) / len(files) if files else 0
            }
        )
    
    def _extract_branch_info(self, lines_elem) -> Dict[str, Any]:
        """Extract branch coverage information from XML."""
        total_branches = 0
        covered_branches = 0
        
        for line in lines_elem.findall('line'):
            branch = line.get('branch')
            if branch == 'true':
                condition_coverage = line.get('condition-coverage', '')
                if condition_coverage:
                    # Parse condition coverage like "50% (1/2)"
                    import re
                    match = re.search(r'\((\d+)/(\d+)\)', condition_coverage)
                    if match:
                        covered = int(match.group(1))
                        total = int(match.group(2))
                        covered_branches += covered
                        total_branches += total
        
        if total_branches > 0:
            branch_coverage = (covered_branches / total_branches) * 100
            return {
                'total_branches': total_branches,
                'missing_branches': total_branches - covered_branches,
                'branch_coverage': branch_coverage
            }
        
        return {}
    
    async def run_vulture_analysis(self) -> List[DeadCodeCandidate]:
        """Run vulture dead code detection."""
        self.logger.info("Running dead code detection...")
        
        candidates = []
        
        try:
            cmd = ["vulture", "src/", "--min-confidence=60", "--sort-by-size"]
            result = await self._run_command(cmd)
            
            for line in result["stdout"].splitlines():
                if ":" in line and ("unused" in line.lower() or "dead" in line.lower()):
                    parts = line.split(":", 3)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = int(parts[1]) if parts[1].isdigit() else 0
                        message = parts[2].strip() if len(parts) > 2 else ""
                        
                        # Extract function name if present
                        function_name = None
                        if "function" in message or "method" in message:
                            import re
                            match = re.search(r"'(\w+)'", message)
                            if match:
                                function_name = match.group(1)
                        
                        # Estimate confidence based on vulture output
                        confidence = 0.6  # Default minimum
                        if "unused" in message.lower():
                            confidence = 0.7
                        if "never used" in message.lower():
                            confidence = 0.8
                        
                        candidates.append(DeadCodeCandidate(
                            file_path=file_path,
                            line=line_num,
                            function_name=function_name,
                            confidence=confidence,
                            reason=message
                        ))
        
        except Exception as e:
            self.logger.error(f"Vulture analysis failed: {e}")
        
        return candidates
    
    async def correlate_coverage_and_dead_code(self) -> CoverageCorrelationReport:
        """Correlate coverage data with dead code detection."""
        self.logger.info("Correlating coverage data with dead code detection...")
        
        # Get coverage data
        coverage_report = self._parse_coverage_results()
        
        # Get dead code candidates
        dead_code_candidates = await self.run_vulture_analysis()
        
        # Correlate the data
        high_confidence_dead_code = []
        
        # Create a mapping of file coverage
        file_coverage_map = {f.file_path: f for f in coverage_report.files}
        
        for candidate in dead_code_candidates:
            # Normalize file path for comparison
            normalized_path = candidate.file_path.replace('\\', '/')
            
            # Find matching coverage file
            coverage_file = None
            for path, cov_file in file_coverage_map.items():
                if normalized_path in path or path in normalized_path:
                    coverage_file = cov_file
                    break
            
            if coverage_file:
                # Check if the line is uncovered
                is_line_uncovered = candidate.line in coverage_file.missing_lines
                
                # Increase confidence if line is uncovered
                adjusted_confidence = candidate.confidence
                if is_line_uncovered:
                    adjusted_confidence += 0.2
                
                # Add coverage context
                candidate.coverage_data = {
                    "file_coverage_percent": coverage_file.coverage_percent,
                    "line_is_uncovered": is_line_uncovered,
                    "file_missing_lines": len(coverage_file.missing_lines)
                }
                
                # High confidence if uncovered AND flagged by vulture
                if is_line_uncovered and adjusted_confidence >= 0.8:
                    candidate.confidence = adjusted_confidence
                    high_confidence_dead_code.append(candidate)
        
        # Generate recommendations
        recommendations = self._generate_cleanup_recommendations(
            coverage_report, dead_code_candidates, high_confidence_dead_code
        )
        
        # Estimate cleanup impact
        cleanup_estimate = self._estimate_cleanup_impact(high_confidence_dead_code)
        
        return CoverageCorrelationReport(
            timestamp=datetime.now().isoformat(),
            coverage_report=coverage_report,
            dead_code_candidates=dead_code_candidates,
            high_confidence_dead_code=high_confidence_dead_code,
            recommendations=recommendations,
            cleanup_estimate=cleanup_estimate
        )
    
    def _generate_cleanup_recommendations(
        self,
        coverage: CoverageReport,
        dead_code: List[DeadCodeCandidate],
        high_confidence: List[DeadCodeCandidate]
    ) -> List[str]:
        """Generate actionable cleanup recommendations."""
        recommendations = []
        
        # Coverage-based recommendations
        low_coverage_files = [f for f in coverage.files if f.coverage_percent < 70]
        if low_coverage_files:
            recommendations.append(
                f"Increase test coverage for {len(low_coverage_files)} files with <70% coverage"
            )
        
        # Dead code recommendations
        if high_confidence:
            recommendations.append(
                f"Remove {len(high_confidence)} high-confidence dead code items"
            )
        
        # File-specific recommendations
        file_issues = {}
        for candidate in high_confidence:
            file_path = candidate.file_path
            if file_path not in file_issues:
                file_issues[file_path] = []
            file_issues[file_path].append(candidate)
        
        for file_path, issues in file_issues.items():
            if len(issues) >= 3:
                recommendations.append(
                    f"Review {file_path}: {len(issues)} dead code candidates found"
                )
        
        # Overall recommendations
        if coverage.overall_coverage < 90:
            recommendations.append(
                f"Increase overall test coverage from {coverage.overall_coverage:.1f}% to 90%+"
            )
        
        return recommendations
    
    def _estimate_cleanup_impact(self, high_confidence: List[DeadCodeCandidate]) -> Dict[str, Any]:
        """Estimate the impact of cleaning up dead code."""
        files_affected = set(c.file_path for c in high_confidence)
        functions_to_remove = len([c for c in high_confidence if c.function_name])
        
        # Estimate lines of code that could be removed
        # This is a rough estimate - would need actual AST analysis for precision
        estimated_loc_removed = len(high_confidence) * 5  # Assume 5 lines per item on average
        
        return {
            "files_affected": len(files_affected),
            "functions_to_remove": functions_to_remove,
            "estimated_loc_removed": estimated_loc_removed,
            "maintenance_burden_reduction": f"{estimated_loc_removed * 0.1:.0f} hours saved annually"
        }
    
    def generate_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Generate coverage trend analysis."""
        self.logger.info(f"Generating coverage trend analysis for last {days} days...")
        
        trend_data = self.coverage_db.get_coverage_trend(days)
        
        if not trend_data:
            return {"error": "No historical coverage data available"}
        
        # Calculate trend metrics
        coverages = [run["overall_coverage"] for run in trend_data]
        
        trend_analysis = {
            "period_days": days,
            "runs_analyzed": len(trend_data),
            "current_coverage": coverages[-1] if coverages else 0,
            "initial_coverage": coverages[0] if coverages else 0,
            "coverage_change": (coverages[-1] - coverages[0]) if len(coverages) > 1 else 0,
            "max_coverage": max(coverages) if coverages else 0,
            "min_coverage": min(coverages) if coverages else 0,
            "average_coverage": sum(coverages) / len(coverages) if coverages else 0,
            "trend_direction": "improving" if len(coverages) > 1 and coverages[-1] > coverages[0] else "declining",
            "data_points": trend_data
        }
        
        return trend_analysis
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a command and return the result."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)
            
            return {
                "stdout": stdout.decode('utf-8'),
                "stderr": stderr.decode('utf-8'),
                "returncode": process.returncode
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def save_report(self, report: CoverageCorrelationReport, output_path: Path) -> None:
        """Save coverage correlation report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        self.logger.info(f"Coverage correlation report saved to {output_path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Coverage analysis and dead code correlation for PyDS application",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Analysis options
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run tests with coverage analysis"
    )
    
    parser.add_argument(
        "--correlate-dead-code",
        action="store_true",
        help="Correlate coverage data with dead code detection"
    )
    
    parser.add_argument(
        "--trend-analysis",
        action="store_true",
        help="Generate coverage trend analysis"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all analysis types"
    )
    
    # Options
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days for trend analysis (default: 30)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/reports/coverage-analysis.json"),
        help="Output file path for analysis report"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    return parser


async def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    
    analyzer = CoverageAnalyzer()
    
    try:
        if args.run_tests or args.all:
            coverage_report = await analyzer.run_tests_with_coverage()
            if not args.quiet:
                print(f"Coverage: {coverage_report.overall_coverage:.1f}%")
        
        if args.correlate_dead_code or args.all:
            correlation_report = await analyzer.correlate_coverage_and_dead_code()
            analyzer.save_report(correlation_report, args.output)
            
            if not args.quiet:
                print(f"\nCorrelation Analysis:")
                print(f"  Dead code candidates: {len(correlation_report.dead_code_candidates)}")
                print(f"  High confidence: {len(correlation_report.high_confidence_dead_code)}")
                print(f"  Cleanup estimate: {correlation_report.cleanup_estimate}")
        
        if args.trend_analysis or args.all:
            trend_analysis = analyzer.generate_trend_analysis(args.days)
            
            if not args.quiet and "error" not in trend_analysis:
                print(f"\nTrend Analysis ({args.days} days):")
                print(f"  Current coverage: {trend_analysis['current_coverage']:.1f}%")
                print(f"  Change: {trend_analysis['coverage_change']:+.1f}%")
                print(f"  Direction: {trend_analysis['trend_direction']}")
        
        return 0
        
    except Exception as e:
        if not args.quiet:
            print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
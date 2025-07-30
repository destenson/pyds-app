#!/usr/bin/env python3
"""
Unified static analysis runner for PyDS application.

This script orchestrates multiple static analysis tools to provide comprehensive
code quality, security, and maintainability checks. It supports parallel
execution, result aggregation, and various output formats.

Usage:
    python scripts/static-analysis.py --all-tools
    python scripts/static-analysis.py --security-only 
    python scripts/static-analysis.py --format json --output analysis/reports/
"""

import argparse
import asyncio
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import subprocess
import tempfile
import shutil
import logging

# Import configuration and utilities if available
try:
    from src.config import get_config, ConfigurationError
    from src.utils.logging import get_logger
    from src.utils.errors import PyDSError
    HAS_PYDS_MODULES = True
except ImportError:
    HAS_PYDS_MODULES = False
    # Fallback logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class AnalysisStatus(str, Enum):
    """Status of an analysis tool execution."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class IssueSeverity(str, Enum):
    """Severity levels for analysis issues."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Issue:
    """Individual analysis finding."""
    tool: str
    file_path: str
    line: Optional[int] = None
    column: Optional[int] = None
    severity: IssueSeverity = IssueSeverity.MEDIUM
    category: str = "unknown"
    message: str = ""
    rule_id: Optional[str] = None
    fix_suggestion: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result from a single analysis tool."""
    tool: str
    status: AnalysisStatus
    duration: float
    issues: List[Issue] = field(default_factory=list)
    files_analyzed: int = 0
    command: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class AnalysisReport:
    """Complete analysis session report."""
    timestamp: str
    total_duration: float
    tools_run: List[str]
    results: List[AnalysisResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Optional[Dict[str, Any]] = None


class AnalysisTool:
    """Base class for analysis tool configuration."""
    
    def __init__(self, name: str, command: List[str], working_dir: Optional[Path] = None):
        self.name = name
        self.command = command
        self.working_dir = working_dir or Path.cwd()
        self.timeout = 300  # 5 minutes default
        self.enabled = True
        
    def is_available(self) -> bool:
        """Check if the tool is available in the system."""
        try:
            result = subprocess.run(
                [self.command[0], "--version"],
                capture_output=True,
                timeout=10,
                cwd=self.working_dir
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
    
    async def run(self) -> AnalysisResult:
        """Execute the analysis tool."""
        if not self.enabled:
            return AnalysisResult(
                tool=self.name,
                status=AnalysisStatus.SKIPPED,
                duration=0.0
            )
        
        if not self.is_available():
            return AnalysisResult(
                tool=self.name,
                status=AnalysisStatus.ERROR,
                duration=0.0,
                error_message=f"Tool {self.name} not available"
            )
        
        start_time = time.time()
        
        try:
            # Run the tool
            process = await asyncio.create_subprocess_exec(
                *self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
                exit_code = process.returncode
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return AnalysisResult(
                    tool=self.name,
                    status=AnalysisStatus.TIMEOUT,
                    duration=time.time() - start_time,
                    error_message=f"Tool {self.name} timed out after {self.timeout}s"
                )
            
            duration = time.time() - start_time
            
            # Parse output
            issues = self.parse_output(stdout.decode('utf-8'), stderr.decode('utf-8'))
            
            # Determine status
            if exit_code == 0:
                status = AnalysisStatus.SUCCESS if not issues else AnalysisStatus.WARNING
            else:
                status = AnalysisStatus.ERROR
            
            return AnalysisResult(
                tool=self.name,
                status=status,
                duration=duration,
                issues=issues,
                command=" ".join(self.command),
                stdout=stdout.decode('utf-8'),
                stderr=stderr.decode('utf-8'),
                exit_code=exit_code
            )
            
        except Exception as e:
            return AnalysisResult(
                tool=self.name,
                status=AnalysisStatus.ERROR,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    def parse_output(self, stdout: str, stderr: str) -> List[Issue]:
        """Parse tool output into structured issues. Override in subclasses."""
        return []


class RuffTool(AnalysisTool):
    """Ruff linting tool."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        super().__init__(
            name="ruff",
            command=["ruff", "check", "src/", "--format", "json"],
            working_dir=working_dir
        )
    
    def parse_output(self, stdout: str, stderr: str) -> List[Issue]:
        """Parse Ruff JSON output."""
        issues = []
        try:
            if stdout.strip():
                data = json.loads(stdout)
                for item in data:
                    issues.append(Issue(
                        tool=self.name,
                        file_path=item.get("filename", ""),
                        line=item.get("location", {}).get("row"),
                        column=item.get("location", {}).get("column"),
                        severity=self._map_severity(item.get("code", "")),
                        category="linting",
                        message=item.get("message", ""),
                        rule_id=item.get("code")
                    ))
        except json.JSONDecodeError:
            pass
        return issues
    
    def _map_severity(self, code: str) -> IssueSeverity:
        """Map Ruff error codes to severity levels."""
        if code.startswith(("E9", "F8")):
            return IssueSeverity.CRITICAL
        elif code.startswith(("E", "W")):
            return IssueSeverity.MEDIUM
        elif code.startswith("S"):
            return IssueSeverity.HIGH
        return IssueSeverity.LOW


class MyPyTool(AnalysisTool):
    """MyPy type checking tool."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        super().__init__(
            name="mypy",
            command=["mypy", "src/", "--config-file", "pyproject.toml"],
            working_dir=working_dir
        )
    
    def parse_output(self, stdout: str, stderr: str) -> List[Issue]:
        """Parse MyPy output."""
        issues = []
        for line in stdout.splitlines():
            if ":" in line and "error:" in line:
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    issues.append(Issue(
                        tool=self.name,
                        file_path=parts[0],
                        line=int(parts[1]) if parts[1].isdigit() else None,
                        severity=IssueSeverity.HIGH if "error" in line else IssueSeverity.MEDIUM,
                        category="typing",
                        message=parts[3].strip() if len(parts) > 3 else ""
                    ))
        return issues


class BanditTool(AnalysisTool):
    """Bandit security scanning tool."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        super().__init__(
            name="bandit",
            command=["bandit", "-r", "src/", "-f", "json", "-c", "pyproject.toml"],
            working_dir=working_dir
        )
    
    def parse_output(self, stdout: str, stderr: str) -> List[Issue]:
        """Parse Bandit JSON output."""
        issues = []
        try:
            if stdout.strip():
                data = json.loads(stdout)
                for result in data.get("results", []):
                    issues.append(Issue(
                        tool=self.name,
                        file_path=result.get("filename", ""),
                        line=result.get("line_number"),
                        severity=self._map_severity(result.get("issue_severity", "")),
                        category="security",
                        message=result.get("issue_text", ""),
                        rule_id=result.get("test_id"),
                        context={"confidence": result.get("issue_confidence")}
                    ))
        except json.JSONDecodeError:
            pass
        return issues
    
    def _map_severity(self, severity: str) -> IssueSeverity:
        """Map Bandit severity to standard levels."""
        mapping = {
            "LOW": IssueSeverity.LOW,
            "MEDIUM": IssueSeverity.MEDIUM,
            "HIGH": IssueSeverity.HIGH
        }
        return mapping.get(severity.upper(), IssueSeverity.MEDIUM)


class VultureTool(AnalysisTool):
    """Vulture dead code detection tool."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        super().__init__(
            name="vulture",
            command=["vulture", "src/", "--min-confidence=80"],
            working_dir=working_dir
        )
    
    def parse_output(self, stdout: str, stderr: str) -> List[Issue]:
        """Parse Vulture output."""
        issues = []
        for line in stdout.splitlines():
            if ":" in line and ("unused" in line.lower() or "dead" in line.lower()):
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    issues.append(Issue(
                        tool=self.name,
                        file_path=parts[0],
                        line=int(parts[1]) if parts[1].isdigit() else None,
                        severity=IssueSeverity.LOW,
                        category="dead-code",
                        message=parts[2].strip() if len(parts) > 2 else ""
                    ))
        return issues


class SemgrepTool(AnalysisTool):
    """Semgrep advanced security scanning tool."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        super().__init__(
            name="semgrep",
            command=["semgrep", "--config=analysis/rules/", "src/", "--json"],
            working_dir=working_dir
        )
    
    def parse_output(self, stdout: str, stderr: str) -> List[Issue]:
        """Parse Semgrep JSON output."""
        issues = []
        try:
            if stdout.strip():
                data = json.loads(stdout)
                for result in data.get("results", []):
                    issues.append(Issue(
                        tool=self.name,
                        file_path=result.get("path", ""),
                        line=result.get("start", {}).get("line"),
                        column=result.get("start", {}).get("col"),
                        severity=self._map_severity(result.get("extra", {}).get("severity", "")),
                        category="security",
                        message=result.get("extra", {}).get("message", ""),
                        rule_id=result.get("check_id"),
                        fix_suggestion=result.get("extra", {}).get("fix")
                    ))
        except json.JSONDecodeError:
            pass
        return issues
    
    def _map_severity(self, severity: str) -> IssueSeverity:
        """Map Semgrep severity to standard levels."""
        mapping = {
            "INFO": IssueSeverity.INFO,
            "WARNING": IssueSeverity.MEDIUM,
            "ERROR": IssueSeverity.HIGH
        }
        return mapping.get(severity.upper(), IssueSeverity.MEDIUM)


class AnalysisRunner:
    """Main analysis orchestrator."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = working_dir or Path.cwd()
        self.logger = get_logger(__name__) if HAS_PYDS_MODULES else logger
        
        # Initialize tools
        self.tools = {
            "ruff": RuffTool(working_dir),
            "mypy": MyPyTool(working_dir),
            "bandit": BanditTool(working_dir),
            "vulture": VultureTool(working_dir),
            "semgrep": SemgrepTool(working_dir)
        }
    
    async def run_analysis(
        self,
        tools: Optional[List[str]] = None,
        parallel: bool = True,
        fail_fast: bool = False
    ) -> AnalysisReport:
        """Run static analysis with specified tools."""
        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine which tools to run
        if tools is None:
            tools_to_run = list(self.tools.keys())
        else:
            tools_to_run = [t for t in tools if t in self.tools]
        
        self.logger.info(f"Starting analysis with tools: {tools_to_run}")
        
        # Run tools
        if parallel:
            results = await self._run_parallel(tools_to_run, fail_fast)
        else:
            results = await self._run_sequential(tools_to_run, fail_fast)
        
        total_duration = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(results)
        
        report = AnalysisReport(
            timestamp=timestamp,
            total_duration=total_duration,
            tools_run=tools_to_run,
            results=results,
            summary=summary
        )
        
        self.logger.info(f"Analysis completed in {total_duration:.2f}s")
        return report
    
    async def _run_parallel(self, tools: List[str], fail_fast: bool) -> List[AnalysisResult]:
        """Run tools in parallel."""
        tasks = []
        for tool_name in tools:
            if tool_name in self.tools:
                task = asyncio.create_task(
                    self.tools[tool_name].run(),
                    name=f"analysis-{tool_name}"
                )
                tasks.append(task)
        
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            
            if fail_fast and result.status == AnalysisStatus.ERROR:
                # Cancel remaining tasks
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break
        
        return results
    
    async def _run_sequential(self, tools: List[str], fail_fast: bool) -> List[AnalysisResult]:
        """Run tools sequentially."""
        results = []
        for tool_name in tools:
            if tool_name in self.tools:
                result = await self.tools[tool_name].run()
                results.append(result)
                
                if fail_fast and result.status == AnalysisStatus.ERROR:
                    break
        
        return results
    
    def _generate_summary(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Generate analysis summary statistics."""
        total_issues = sum(len(r.issues) for r in results)
        issues_by_severity = {}
        issues_by_category = {}
        tools_status = {}
        
        for result in results:
            tools_status[result.tool] = result.status.value
            
            for issue in result.issues:
                # Count by severity
                severity = issue.severity.value
                issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
                
                # Count by category
                category = issue.category
                issues_by_category[category] = issues_by_category.get(category, 0) + 1
        
        return {
            "total_issues": total_issues,
            "issues_by_severity": issues_by_severity,
            "issues_by_category": issues_by_category,
            "tools_status": tools_status,
            "success_rate": len([r for r in results if r.status == AnalysisStatus.SUCCESS]) / len(results) if results else 0
        }
    
    def save_report(self, report: AnalysisReport, output_path: Path, format: str = "json") -> None:
        """Save analysis report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Report saved to {output_path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified static analysis runner for PyDS application",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Tool selection
    parser.add_argument(
        "--all-tools",
        action="store_true",
        help="Run all available analysis tools"
    )
    
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=["ruff", "mypy", "bandit", "vulture", "semgrep"],
        help="Specific tools to run"
    )
    
    parser.add_argument(
        "--security-only",
        action="store_true",
        help="Run only security-focused tools (bandit, semgrep)"
    )
    
    # Execution options
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run tools sequentially instead of in parallel"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop analysis on first tool failure"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/reports/static-analysis.json"),
        help="Output file path for analysis report"
    )
    
    parser.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Output format for analysis report"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any issues found"
    )
    
    return parser


async def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    
    # Determine tools to run
    if args.all_tools:
        tools = None  # Run all tools
    elif args.security_only:
        tools = ["bandit", "semgrep"]
    elif args.tools:
        tools = args.tools
    else:
        tools = None  # Default to all tools
    
    # Run analysis
    runner = AnalysisRunner()
    report = await runner.run_analysis(
        tools=tools,
        parallel=not args.sequential,
        fail_fast=args.fail_fast
    )
    
    # Save report
    runner.save_report(report, args.output, args.format)
    
    # Print summary
    if not args.quiet:
        print(f"\nAnalysis Summary:")
        print(f"  Total issues: {report.summary['total_issues']}")
        print(f"  Duration: {report.total_duration:.2f}s")
        print(f"  Tools run: {', '.join(report.tools_run)}")
        
        if report.summary['issues_by_severity']:
            print(f"  Issues by severity:")
            for severity, count in report.summary['issues_by_severity'].items():
                print(f"    {severity}: {count}")
    
    # Exit code logic
    if args.strict and report.summary['total_issues'] > 0:
        return 1
    
    # Check for tool failures
    failed_tools = [r for r in report.results if r.status == AnalysisStatus.ERROR]
    if failed_tools:
        return 2
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
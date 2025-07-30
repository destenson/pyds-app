#!/usr/bin/env python3
"""
Security-focused analysis script for PyDS application.

This script combines multiple security scanning tools to provide comprehensive
security analysis including code vulnerabilities, dependency scanning, and
secrets detection.

Usage:
    python scripts/security-scan.py --all
    python scripts/security-scan.py --baseline analysis/baselines/
    python scripts/security-scan.py --dependency-scan-only
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import re

# Import PyDS modules if available
try:
    from src.utils.logging import get_logger
    from src.utils.errors import PyDSError
    HAS_PYDS_MODULES = True
except ImportError:
    HAS_PYDS_MODULES = False
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class SecuritySeverity(str, Enum):
    """Security issue severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityCategory(str, Enum):
    """Security issue categories."""
    VULNERABILITY = "vulnerability"
    DEPENDENCY = "dependency"
    SECRETS = "secrets"
    INJECTION = "injection"
    CRYPTO = "crypto"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    MEMORY_SAFETY = "memory_safety"
    CONFIGURATION = "configuration"


@dataclass
class SecurityIssue:
    """Security issue finding."""
    tool: str
    category: SecurityCategory
    severity: SecuritySeverity
    title: str
    description: str
    file_path: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    rule_id: Optional[str] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    fix_suggestion: Optional[str] = None
    references: List[str] = field(default_factory=list)
    confidence: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityReport:
    """Comprehensive security analysis report."""
    timestamp: str
    scan_duration: float
    tools_used: List[str]
    issues: List[SecurityIssue]
    summary: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)


class SecretsScanner:
    """Custom secrets detection for common patterns."""
    
    def __init__(self):
        self.patterns = [
            # API Keys and tokens
            (r'(?i)api[_-]?key\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', 'API Key'),
            (r'(?i)secret[_-]?key\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', 'Secret Key'),
            (r'(?i)access[_-]?token\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', 'Access Token'),
            (r'(?i)auth[_-]?token\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', 'Auth Token'),
            
            # Database credentials
            (r'(?i)password\s*[:=]\s*["\']([^"\'\\s]{8,})["\']', 'Hardcoded Password'),
            (r'(?i)db[_-]?pass\s*[:=]\s*["\']([^"\'\\s]{8,})["\']', 'Database Password'),
            (r'(?i)mysql[_-]?pass\s*[:=]\s*["\']([^"\'\\s]{8,})["\']', 'MySQL Password'),
            (r'(?i)postgres[_-]?pass\s*[:=]\s*["\']([^"\'\\s]{8,})["\']', 'PostgreSQL Password'),
            
            # Cloud provider keys
            (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
            (r'(?i)aws[_-]?secret[_-]?access[_-]?key', 'AWS Secret Access Key'),
            (r'(?i)gcp[_-]?service[_-]?account', 'GCP Service Account'),
            (r'(?i)azure[_-]?tenant[_-]?id', 'Azure Tenant ID'),
            
            # Private keys
            (r'-----BEGIN [A-Z ]+PRIVATE KEY-----', 'Private Key'),
            (r'-----BEGIN RSA PRIVATE KEY-----', 'RSA Private Key'),
            (r'-----BEGIN EC PRIVATE KEY-----', 'EC Private Key'),
            
            # JWT tokens
            (r'eyJ[A-Za-z0-9_/+-]*\.eyJ[A-Za-z0-9_/+-]*\.[A-Za-z0-9_/+-]*', 'JWT Token'),
            
            # URLs with credentials
            (r'[a-zA-Z][a-zA-Z0-9+.-]*://[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+', 'URL with credentials'),
        ]
    
    def scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """Scan a file for secrets."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for line_num, line in enumerate(content.splitlines(), 1):
                for pattern, secret_type in self.patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        # Skip common false positives
                        if self._is_false_positive(match.group(0), file_path):
                            continue
                        
                        issues.append(SecurityIssue(
                            tool="secrets-scanner",
                            category=SecurityCategory.SECRETS,
                            severity=SecuritySeverity.HIGH,
                            title=f"Potential {secret_type} detected",
                            description=f"Found potential {secret_type} in source code",
                            file_path=str(file_path),
                            line=line_num,
                            column=match.start(),
                            confidence="medium"
                        ))
        except Exception:
            pass  # Skip files that can't be read
        
        return issues
    
    def _is_false_positive(self, match: str, file_path: Path) -> bool:
        """Check if a match is likely a false positive."""
        # Skip test files with dummy data
        if 'test' in str(file_path).lower() or 'example' in str(file_path).lower():
            return True
        
        # Skip common placeholder patterns
        placeholders = ['your_key_here', 'insert_key', 'api_key_here', 'password123', 'secret123']
        if any(placeholder in match.lower() for placeholder in placeholders):
            return True
        
        # Skip very short matches
        if len(match.strip('"\'')) < 8:
            return True
        
        return False
    
    def scan_directory(self, directory: Path) -> List[SecurityIssue]:
        """Scan all files in a directory for secrets."""
        issues = []
        
        for file_path in directory.rglob("*.py"):
            if file_path.is_file():
                issues.extend(self.scan_file(file_path))
        
        return issues


class SecurityAnalyzer:
    """Main security analysis orchestrator."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = working_dir or Path.cwd()
        self.logger = get_logger(__name__) if HAS_PYDS_MODULES else logger
        self.secrets_scanner = SecretsScanner()
    
    async def run_bandit_scan(self) -> List[SecurityIssue]:
        """Run Bandit security scanner."""
        issues = []
        
        try:
            cmd = ["bandit", "-r", "src/", "-f", "json", "-c", "pyproject.toml"]
            result = await self._run_command(cmd)
            
            if result["stdout"]:
                data = json.loads(result["stdout"])
                for finding in data.get("results", []):
                    issues.append(SecurityIssue(
                        tool="bandit",
                        category=self._map_bandit_category(finding.get("test_name", "")),
                        severity=self._map_bandit_severity(finding.get("issue_severity", "")),
                        title=finding.get("issue_text", ""),
                        description=finding.get("issue_text", ""),
                        file_path=finding.get("filename", ""),
                        line=finding.get("line_number"),
                        rule_id=finding.get("test_id"),
                        confidence=finding.get("issue_confidence", "").lower(),
                        context={"more_info": finding.get("more_info", "")}
                    ))
        except Exception as e:
            self.logger.error(f"Bandit scan failed: {e}")
        
        return issues
    
    async def run_semgrep_scan(self) -> List[SecurityIssue]:
        """Run Semgrep security scanner."""
        issues = []
        
        try:
            cmd = ["semgrep", "--config=analysis/rules/", "src/", "--json"]
            result = await self._run_command(cmd)
            
            if result["stdout"]:
                data = json.loads(result["stdout"])
                for finding in data.get("results", []):
                    extra = finding.get("extra", {})
                    metadata = extra.get("metadata", {})
                    
                    issues.append(SecurityIssue(
                        tool="semgrep",
                        category=self._map_semgrep_category(metadata.get("category", "")),
                        severity=self._map_semgrep_severity(extra.get("severity", "")),
                        title=extra.get("message", ""),
                        description=extra.get("message", ""),
                        file_path=finding.get("path", ""),
                        line=finding.get("start", {}).get("line"),
                        column=finding.get("start", {}).get("col"),
                        rule_id=finding.get("check_id"),
                        cwe_id=metadata.get("cwe"),
                        fix_suggestion=extra.get("fix"),
                        references=metadata.get("references", [])
                    ))
        except Exception as e:
            self.logger.error(f"Semgrep scan failed: {e}")
        
        return issues
    
    async def run_dependency_scan(self) -> List[SecurityIssue]:
        """Run dependency vulnerability scanning with safety and pip-audit."""
        issues = []
        
        # Run safety
        try:
            cmd = ["safety", "check", "--json"]
            result = await self._run_command(cmd)
            
            if result["stdout"]:
                data = json.loads(result["stdout"])
                for vuln in data:
                    issues.append(SecurityIssue(
                        tool="safety",
                        category=SecurityCategory.DEPENDENCY,
                        severity=self._map_cvss_to_severity(vuln.get("advisory", "")),
                        title=f"Vulnerable dependency: {vuln.get('package_name', '')}",
                        description=vuln.get("advisory", ""),
                        rule_id=vuln.get("vulnerability_id"),
                        context={
                            "package_name": vuln.get("package_name"),
                            "installed_version": vuln.get("installed_version"),
                            "affected_versions": vuln.get("affected_versions")
                        }
                    ))
        except Exception as e:
            self.logger.warning(f"Safety scan failed: {e}")
        
        # Run pip-audit
        try:
            cmd = ["pip-audit", "--format=json"]
            result = await self._run_command(cmd)
            
            if result["stdout"]:
                data = json.loads(result["stdout"])
                for vuln in data.get("vulnerabilities", []):
                    issues.append(SecurityIssue(
                        tool="pip-audit",
                        category=SecurityCategory.DEPENDENCY,
                        severity=self._map_cvss_to_severity(vuln.get("description", "")),
                        title=f"Vulnerable dependency: {vuln.get('package', '')}",
                        description=vuln.get("description", ""),
                        rule_id=vuln.get("id"),
                        context={
                            "package_name": vuln.get("package"),
                            "installed_version": vuln.get("installed_version"),
                            "fixed_versions": vuln.get("fixed_versions")
                        }
                    ))
        except Exception as e:
            self.logger.warning(f"pip-audit scan failed: {e}")
        
        return issues
    
    async def run_secrets_scan(self) -> List[SecurityIssue]:
        """Run secrets detection scan."""
        self.logger.info("Running secrets detection scan...")
        return self.secrets_scanner.scan_directory(self.working_dir / "src")
    
    async def run_comprehensive_scan(
        self,
        include_bandit: bool = True,
        include_semgrep: bool = True,
        include_dependencies: bool = True,
        include_secrets: bool = True
    ) -> SecurityReport:
        """Run comprehensive security analysis."""
        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        tools_used = []
        all_issues = []
        
        self.logger.info("Starting comprehensive security scan...")
        
        # Run scans
        if include_bandit:
            tools_used.append("bandit")
            bandit_issues = await self.run_bandit_scan()
            all_issues.extend(bandit_issues)
            self.logger.info(f"Bandit found {len(bandit_issues)} issues")
        
        if include_semgrep:
            tools_used.append("semgrep")
            semgrep_issues = await self.run_semgrep_scan()
            all_issues.extend(semgrep_issues)
            self.logger.info(f"Semgrep found {len(semgrep_issues)} issues")
        
        if include_dependencies:
            tools_used.extend(["safety", "pip-audit"])
            dep_issues = await self.run_dependency_scan()
            all_issues.extend(dep_issues)
            self.logger.info(f"Dependency scan found {len(dep_issues)} issues")
        
        if include_secrets:
            tools_used.append("secrets-scanner")
            secrets_issues = await self.run_secrets_scan()
            all_issues.extend(secrets_issues)
            self.logger.info(f"Secrets scan found {len(secrets_issues)} issues")
        
        # Deduplicate issues
        all_issues = self._deduplicate_issues(all_issues)
        
        # Generate summary
        summary = self._generate_summary(all_issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues)
        
        scan_duration = time.time() - start_time
        
        report = SecurityReport(
            timestamp=timestamp,
            scan_duration=scan_duration,
            tools_used=tools_used,
            issues=all_issues,
            summary=summary,
            recommendations=recommendations
        )
        
        self.logger.info(f"Security scan completed in {scan_duration:.2f}s with {len(all_issues)} issues")
        return report
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a command and return the result."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            
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
    
    def _deduplicate_issues(self, issues: List[SecurityIssue]) -> List[SecurityIssue]:
        """Remove duplicate issues based on file, line, and message."""
        seen = set()
        unique_issues = []
        
        for issue in issues:
            key = (issue.file_path, issue.line, issue.title)
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)
        
        return unique_issues
    
    def _generate_summary(self, issues: List[SecurityIssue]) -> Dict[str, Any]:
        """Generate security scan summary."""
        summary = {
            "total_issues": len(issues),
            "by_severity": {},
            "by_category": {},
            "by_tool": {},
            "critical_files": []
        }
        
        file_issue_count = {}
        
        for issue in issues:
            # Count by severity
            severity = issue.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            
            # Count by category
            category = issue.category.value
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
            
            # Count by tool
            tool = issue.tool
            summary["by_tool"][tool] = summary["by_tool"].get(tool, 0) + 1
            
            # Track files with issues
            if issue.file_path:
                file_issue_count[issue.file_path] = file_issue_count.get(issue.file_path, 0) + 1
        
        # Find files with most issues
        summary["critical_files"] = sorted(
            file_issue_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return summary
    
    def _generate_recommendations(self, issues: List[SecurityIssue]) -> List[str]:
        """Generate security recommendations based on found issues."""
        recommendations = []
        
        # Count issues by category
        category_counts = {}
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        # Generate category-specific recommendations
        if category_counts.get(SecurityCategory.SECRETS, 0) > 0:
            recommendations.append(
                "Use environment variables or secure credential storage instead of hardcoded secrets"
            )
        
        if category_counts.get(SecurityCategory.DEPENDENCY, 0) > 0:
            recommendations.append(
                "Update vulnerable dependencies to their latest secure versions"
            )
        
        if category_counts.get(SecurityCategory.INJECTION, 0) > 0:
            recommendations.append(
                "Implement proper input validation and parameterized queries"
            )
        
        if category_counts.get(SecurityCategory.CRYPTO, 0) > 0:
            recommendations.append(
                "Use well-established cryptographic libraries and avoid custom implementations"
            )
        
        # Critical severity recommendations
        critical_issues = [i for i in issues if i.severity == SecuritySeverity.CRITICAL]
        if critical_issues:
            recommendations.append(
                f"Address {len(critical_issues)} critical security issues immediately"
            )
        
        return recommendations
    
    def _map_bandit_severity(self, severity: str) -> SecuritySeverity:
        """Map Bandit severity to standard levels."""
        mapping = {
            "LOW": SecuritySeverity.LOW,
            "MEDIUM": SecuritySeverity.MEDIUM,
            "HIGH": SecuritySeverity.HIGH
        }
        return mapping.get(severity.upper(), SecuritySeverity.MEDIUM)
    
    def _map_bandit_category(self, test_name: str) -> SecurityCategory:
        """Map Bandit test names to security categories."""
        if "sql" in test_name.lower() or "injection" in test_name.lower():
            return SecurityCategory.INJECTION
        elif "crypto" in test_name.lower() or "hash" in test_name.lower():
            return SecurityCategory.CRYPTO
        elif "password" in test_name.lower() or "hardcoded" in test_name.lower():
            return SecurityCategory.SECRETS
        else:
            return SecurityCategory.VULNERABILITY
    
    def _map_semgrep_severity(self, severity: str) -> SecuritySeverity:
        """Map Semgrep severity to standard levels."""
        mapping = {
            "INFO": SecuritySeverity.INFO,
            "WARNING": SecuritySeverity.MEDIUM,
            "ERROR": SecuritySeverity.HIGH
        }
        return mapping.get(severity.upper(), SecuritySeverity.MEDIUM)
    
    def _map_semgrep_category(self, category: str) -> SecurityCategory:
        """Map Semgrep category to security categories."""
        mapping = {
            "security": SecurityCategory.VULNERABILITY,
            "injection": SecurityCategory.INJECTION,
            "crypto": SecurityCategory.CRYPTO,
            "secrets": SecurityCategory.SECRETS
        }
        return mapping.get(category.lower(), SecurityCategory.VULNERABILITY)
    
    def _map_cvss_to_severity(self, description: str) -> SecuritySeverity:
        """Map CVSS score or description to severity."""
        description_lower = description.lower()
        
        if "critical" in description_lower:
            return SecuritySeverity.CRITICAL
        elif "high" in description_lower:
            return SecuritySeverity.HIGH
        elif "medium" in description_lower:
            return SecuritySeverity.MEDIUM
        elif "low" in description_lower:
            return SecuritySeverity.LOW
        else:
            return SecuritySeverity.MEDIUM
    
    def save_report(self, report: SecurityReport, output_path: Path) -> None:
        """Save security report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        self.logger.info(f"Security report saved to {output_path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Security-focused analysis for PyDS application",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Scan options
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all security scans (default)"
    )
    
    parser.add_argument(
        "--code-scan-only",
        action="store_true",
        help="Run only code security scans (bandit, semgrep, secrets)"
    )
    
    parser.add_argument(
        "--dependency-scan-only",
        action="store_true",
        help="Run only dependency vulnerability scans"
    )
    
    parser.add_argument(
        "--secrets-only",
        action="store_true",
        help="Run only secrets detection"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/reports/security-scan.json"),
        help="Output file path for security report"
    )
    
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to baseline directory for comparison"
    )
    
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with non-zero code if security issues found"
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
    
    # Create analyzer
    analyzer = SecurityAnalyzer()
    
    # Determine scan options
    if args.code_scan_only:
        scan_opts = {
            "include_bandit": True,
            "include_semgrep": True,
            "include_dependencies": False,
            "include_secrets": True
        }
    elif args.dependency_scan_only:
        scan_opts = {
            "include_bandit": False,
            "include_semgrep": False,
            "include_dependencies": True,
            "include_secrets": False
        }
    elif args.secrets_only:
        scan_opts = {
            "include_bandit": False,
            "include_semgrep": False,
            "include_dependencies": False,
            "include_secrets": True
        }
    else:
        # Default: run all scans
        scan_opts = {
            "include_bandit": True,
            "include_semgrep": True,
            "include_dependencies": True,
            "include_secrets": True
        }
    
    # Run security scan
    report = await analyzer.run_comprehensive_scan(**scan_opts)
    
    # Save report
    analyzer.save_report(report, args.output)
    
    # Print summary
    if not args.quiet:
        print(f"\nSecurity Scan Summary:")
        print(f"  Total issues: {report.summary['total_issues']}")
        print(f"  Duration: {report.scan_duration:.2f}s")
        print(f"  Tools used: {', '.join(report.tools_used)}")
        
        if report.summary['by_severity']:
            print(f"  Issues by severity:")
            for severity, count in report.summary['by_severity'].items():
                print(f"    {severity}: {count}")
        
        if report.recommendations:
            print(f"\n  Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"    {i}. {rec}")
    
    # Exit code logic
    if args.exit_code:
        critical_count = report.summary.get('by_severity', {}).get('critical', 0)
        high_count = report.summary.get('by_severity', {}).get('high', 0)
        
        if critical_count > 0:
            return 3  # Critical issues found
        elif high_count > 0:
            return 2  # High severity issues found
        elif report.summary['total_issues'] > 0:
            return 1  # Any issues found
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
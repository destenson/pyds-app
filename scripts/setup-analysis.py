#!/usr/bin/env python3
"""
Development environment setup script for static analysis system.

This script installs and configures all analysis tools, sets up pre-commit hooks,
generates initial baselines, and validates tool configurations for the PyDS
static analysis system.

Usage:
    python scripts/setup-analysis.py --install-all
    python scripts/setup-analysis.py --setup-hooks
    python scripts/setup-analysis.py --generate-baselines
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AnalysisSetup:
    """Main setup orchestrator for static analysis system."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = working_dir or Path.cwd()
        self.logger = logger
    
    async def install_tools(self) -> bool:
        """Install all required analysis tools."""
        self.logger.info("Installing static analysis tools...")
        
        success = True
        
        # Check if uv is available (preferred)
        if await self._check_command_available("uv"):
            self.logger.info("Using uv for package installation")
            success &= await self._run_command([
                "uv", "sync", "--extra", "dev"
            ])
        else:
            self.logger.info("uv not available, using pip")
            success &= await self._run_command([
                "pip", "install", "-e", ".[dev]"
            ])
        
        # Verify tool installations
        tools_to_check = [
            ("ruff", ["ruff", "--version"]),
            ("mypy", ["mypy", "--version"]),
            ("bandit", ["bandit", "--version"]),
            ("vulture", ["vulture", "--version"]),
            ("semgrep", ["semgrep", "--version"]),
            ("safety", ["safety", "--version"]),
            ("pip-audit", ["pip-audit", "--version"]),
            ("pre-commit", ["pre-commit", "--version"]),
            ("pytest", ["pytest", "--version"]),
            ("coverage", ["coverage", "--version"])
        ]
        
        self.logger.info("Verifying tool installations...")
        for tool_name, check_cmd in tools_to_check:
            if await self._check_command_available(check_cmd[0]):
                result = await self._run_command(check_cmd, capture_output=True)
                if result["success"]:
                    version = result["stdout"].strip().split('\n')[0]
                    self.logger.info(f"✓ {tool_name}: {version}")
                else:
                    self.logger.warning(f"✗ {tool_name}: Failed to get version")
                    success = False
            else:
                self.logger.error(f"✗ {tool_name}: Not found")
                success = False
        
        return success
    
    async def setup_pre_commit_hooks(self) -> bool:
        """Setup pre-commit hooks."""
        self.logger.info("Setting up pre-commit hooks...")
        
        # Install pre-commit hooks
        success = await self._run_command([
            "pre-commit", "install"
        ])
        
        if success:
            self.logger.info("✓ Pre-commit hooks installed")
            
            # Install commit-msg hook for conventional commits
            success &= await self._run_command([
                "pre-commit", "install", "--hook-type", "commit-msg"
            ])
            
            if success:
                self.logger.info("✓ Commit message hooks installed")
        
        return success
    
    async def generate_baselines(self) -> bool:
        """Generate initial baselines for analysis tools."""
        self.logger.info("Generating analysis baselines...")
        
        baselines_dir = self.working_dir / "analysis" / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)
        
        success = True
        
        # Generate Bandit baseline
        self.logger.info("Generating Bandit baseline...")
        bandit_baseline = baselines_dir / "bandit-baseline.json"
        result = await self._run_command([
            "bandit", "-r", "src/", "-f", "json", "-o", str(bandit_baseline)
        ], capture_output=True)
        
        if result["success"]:
            self.logger.info(f"✓ Bandit baseline saved to {bandit_baseline}")
        else:
            self.logger.warning("✗ Failed to generate Bandit baseline")
            success = False
        
        # Generate initial coverage baseline
        self.logger.info("Generating coverage baseline...")
        coverage_baseline = baselines_dir / "coverage-baseline.json"
        
        # Run a quick test to get initial coverage
        result = await self._run_command([
            "pytest", "--cov=src", "--cov-report=json", "tests/", "-x"
        ], capture_output=True)
        
        if result["success"]:
            # Move coverage.json to baseline
            coverage_json = self.working_dir / "coverage.json"
            if coverage_json.exists():
                coverage_json.rename(coverage_baseline)
                self.logger.info(f"✓ Coverage baseline saved to {coverage_baseline}")
            else:
                self.logger.warning("✗ Coverage report not found")
        else:
            self.logger.warning("✗ Failed to generate coverage baseline")
        
        return success
    
    async def validate_configurations(self) -> Dict[str, bool]:
        """Validate all tool configurations."""
        self.logger.info("Validating tool configurations...")
        
        validation_results = {}
        
        # Test Ruff configuration
        self.logger.info("Validating Ruff configuration...")
        result = await self._run_command([
            "ruff", "check", "--config", "pyproject.toml", "--show-source", "src/"
        ], capture_output=True)
        validation_results["ruff"] = result["success"] or result["returncode"] == 1  # 1 is expected for issues found
        
        # Test MyPy configuration
        self.logger.info("Validating MyPy configuration...")
        result = await self._run_command([
            "mypy", "--config-file", "pyproject.toml", "src/"
        ], capture_output=True)
        validation_results["mypy"] = result["success"] or result["returncode"] == 1
        
        # Test Bandit configuration
        self.logger.info("Validating Bandit configuration...")
        result = await self._run_command([
            "bandit", "-c", "pyproject.toml", "-r", "src/", "-f", "json"
        ], capture_output=True)
        validation_results["bandit"] = result["success"] or result["returncode"] == 1
        
        # Test Vulture configuration
        self.logger.info("Validating Vulture configuration...")
        result = await self._run_command([
            "vulture", "src/", "--config", ".vulture"
        ], capture_output=True)
        validation_results["vulture"] = result["success"] or result["returncode"] == 1
        
        # Test Semgrep configuration
        self.logger.info("Validating Semgrep configuration...")
        rules_dir = self.working_dir / "analysis" / "rules"
        if rules_dir.exists():
            result = await self._run_command([
                "semgrep", "--config", str(rules_dir), "src/", "--dry-run"
            ], capture_output=True)
            validation_results["semgrep"] = result["success"] or result["returncode"] == 1
        else:
            self.logger.warning("Semgrep rules directory not found")
            validation_results["semgrep"] = False
        
        # Test pre-commit configuration
        self.logger.info("Validating pre-commit configuration...")
        result = await self._run_command([
            "pre-commit", "run", "--all-files", "--show-diff-on-failure"
        ], capture_output=True)
        validation_results["pre-commit"] = result["success"] or result["returncode"] == 1
        
        # Report results
        for tool, valid in validation_results.items():
            status = "✓" if valid else "✗"
            self.logger.info(f"{status} {tool} configuration")
        
        return validation_results
    
    async def setup_github_directories(self) -> bool:
        """Setup GitHub Actions workflow directories."""
        self.logger.info("Setting up GitHub Actions directories...")
        
        github_dir = self.working_dir / ".github"
        workflows_dir = github_dir / "workflows"
        
        # Create directories
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # Create issue templates directory
        issue_templates_dir = github_dir / "ISSUE_TEMPLATE"
        issue_templates_dir.mkdir(exist_ok=True)
        
        self.logger.info("✓ GitHub directories created")
        return True
    
    async def create_development_documentation(self) -> bool:
        """Create developer documentation for the analysis system."""
        self.logger.info("Creating development documentation...")
        
        docs_dir = self.working_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Create static analysis guide
        analysis_guide = docs_dir / "static-analysis-guide.md"
        
        guide_content = """# Static Analysis Guide

This document provides guidance for using the comprehensive static analysis system in the PyDS project.

## Quick Start

1. **Setup the analysis environment:**
   ```bash
   python scripts/setup-analysis.py --install-all
   ```

2. **Run all analysis tools:**
   ```bash
   python scripts/static-analysis.py --all-tools
   ```

3. **Run security-focused scan:**
   ```bash
   python scripts/security-scan.py --all
   ```

4. **Run coverage analysis:**
   ```bash
   python scripts/coverage-analysis.py --run-tests --correlate-dead-code
   ```

## Tools Overview

### Code Quality Tools
- **Ruff**: Fast Python linter and formatter
- **MyPy**: Static type checker
- **Black**: Code formatter (via Ruff)

### Security Tools
- **Bandit**: Security issues detection
- **Semgrep**: Advanced security patterns
- **Safety**: Dependency vulnerability scanning
- **pip-audit**: Alternative dependency scanner
- **Custom secrets scanner**: Detects hardcoded secrets

### Dead Code Detection
- **Vulture**: Identifies unused code
- **Coverage correlation**: Finds untested + unused code

### Pre-commit Integration
All tools are integrated into pre-commit hooks for automatic execution on commit.

## Configuration Files

- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `pyproject.toml`: Tool configurations (ruff, mypy, bandit, vulture)
- `.bandit`: Bandit-specific configuration
- `.vulture`: Vulture whitelist configuration
- `analysis/rules/`: Custom Semgrep security rules

## Baseline Management

Baselines help reduce false positives:
- `analysis/baselines/bandit-baseline.json`: Known security issues
- `analysis/baselines/vulture-whitelist.py`: Legitimate "unused" code
- `analysis/baselines/coverage-baseline.json`: Coverage tracking

## CI/CD Integration

The analysis system integrates with GitHub Actions:
- Runs on pull requests and main branch commits
- Provides quality gates and failure prevention
- Uploads analysis reports as artifacts

## Troubleshooting

### Common Issues

1. **Tool not found**: Run setup script to install dependencies
2. **False positives**: Update baseline files or tool configurations
3. **Performance issues**: Adjust tool timeouts and concurrency settings
4. **Coverage issues**: Ensure tests are properly configured

### Getting Help

1. Check tool documentation links in pyproject.toml comments
2. Review baseline files for examples of excluded patterns
3. Examine GitHub Actions logs for CI-specific issues
"""
        
        with open(analysis_guide, 'w') as f:
            f.write(guide_content)
        
        self.logger.info(f"✓ Static analysis guide created: {analysis_guide}")
        return True
    
    async def run_comprehensive_setup(self) -> bool:
        """Run the complete setup process."""
        self.logger.info("Starting comprehensive static analysis setup...")
        
        success = True
        
        # Install tools
        success &= await self.install_tools()
        
        # Setup pre-commit hooks
        if success:
            success &= await self.setup_pre_commit_hooks()
        
        # Generate baselines
        if success:
            success &= await self.generate_baselines()
        
        # Setup GitHub directories
        if success:
            success &= await self.setup_github_directories()
        
        # Create documentation
        if success:
            success &= await self.create_development_documentation()
        
        # Validate configurations
        if success:
            validation_results = await self.validate_configurations()
            success &= all(validation_results.values())
        
        if success:
            self.logger.info("✅ Static analysis setup completed successfully!")
            self.logger.info("Next steps:")
            self.logger.info("  1. Run 'python scripts/static-analysis.py --all-tools' to test")
            self.logger.info("  2. Make a test commit to verify pre-commit hooks")
            self.logger.info("  3. Review docs/static-analysis-guide.md for usage")
        else:
            self.logger.error("❌ Setup completed with errors. Check logs above.")
        
        return success
    
    async def _check_command_available(self, command: str) -> bool:
        """Check if a command is available in the system."""
        try:
            result = await asyncio.create_subprocess_exec(
                command, "--version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await result.wait()
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    async def _run_command(
        self, 
        cmd: List[str], 
        capture_output: bool = False,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Run a command and return result information."""
        try:
            if capture_output:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.working_dir
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                return {
                    "success": process.returncode == 0,
                    "returncode": process.returncode,
                    "stdout": stdout.decode('utf-8'),
                    "stderr": stderr.decode('utf-8')
                }
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.working_dir
                )
                
                returncode = await asyncio.wait_for(
                    process.wait(),
                    timeout=timeout
                )
                
                return {
                    "success": returncode == 0,
                    "returncode": returncode,
                    "stdout": "",
                    "stderr": ""
                }
                
        except asyncio.TimeoutError:
            self.logger.error(f"Command timed out: {' '.join(cmd)}")
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Command timed out"
            }
        except Exception as e:
            self.logger.error(f"Command failed: {' '.join(cmd)}: {e}")
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Setup development environment for static analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete setup
  python scripts/setup-analysis.py --install-all
  
  # Install tools only
  python scripts/setup-analysis.py --install-tools
  
  # Setup hooks only
  python scripts/setup-analysis.py --setup-hooks
  
  # Generate baselines only
  python scripts/setup-analysis.py --generate-baselines
  
  # Validate configurations
  python scripts/setup-analysis.py --validate-only
        """
    )
    
    # Setup options
    parser.add_argument(
        "--install-all",
        action="store_true",
        help="Run complete setup (install tools, hooks, baselines, validation)"
    )
    
    parser.add_argument(
        "--install-tools",
        action="store_true",
        help="Install analysis tools only"
    )
    
    parser.add_argument(
        "--setup-hooks",
        action="store_true",
        help="Setup pre-commit hooks only"
    )
    
    parser.add_argument(
        "--generate-baselines",
        action="store_true",
        help="Generate analysis baselines only"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate tool configurations only"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    return parser


async def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    setup = AnalysisSetup()
    
    try:
        if args.install_all:
            success = await setup.run_comprehensive_setup()
        elif args.install_tools:
            success = await setup.install_tools()
        elif args.setup_hooks:
            success = await setup.setup_pre_commit_hooks()
        elif args.generate_baselines:
            success = await setup.generate_baselines()
        elif args.validate_only:
            validation_results = await setup.validate_configurations()
            success = all(validation_results.values())
        else:
            # Default: run comprehensive setup
            success = await setup.run_comprehensive_setup()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
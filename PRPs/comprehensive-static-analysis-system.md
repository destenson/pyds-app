name: "Comprehensive Static Analysis System"
description: |

## Purpose
Implement a robust, multi-layered static analysis system that ensures every piece of code in the PyDS application will work correctly at runtime without requiring execution. This includes security scanning, dependency vulnerability detection, dead code elimination, and comprehensive quality checks with automated workflows.

## Core Principles
1. **Prevention Over Detection**: Catch issues before they reach production
2. **Security First**: Prioritize security vulnerabilities and supply chain risks
3. **Automation**: Integrate into development workflow with pre-commit hooks and CI/CD
4. **Comprehensive Coverage**: Multiple analysis dimensions (syntax, types, security, dependencies, complexity)
5. **Developer Friendly**: Fast feedback loops with actionable reports

---

## Goal
Create a comprehensive static analysis system that provides multiple layers of code quality assurance:
- **Security scanning** for vulnerabilities and secrets
- **Dependency vulnerability scanning** for supply chain security  
- **Dead code detection** to maintain clean codebase
- **Advanced type checking** beyond basic MyPy
- **Code complexity analysis** and maintainability metrics
- **Automated enforcement** via pre-commit hooks and CI/CD
- **Unified reporting** dashboard for all analysis results

## Why
- **Risk Mitigation**: Prevent security vulnerabilities from reaching production
- **Code Quality**: Maintain high standards as codebase grows
- **Developer Productivity**: Catch issues early when they're cheaper to fix
- **Compliance**: Meet security standards for video analytics applications
- **Technical Debt**: Proactively identify and remove unused/problematic code
- **Team Consistency**: Ensure all developers follow same quality standards

## What
A multi-tool static analysis pipeline that provides:
1. **Pre-commit validation** - Immediate feedback during development
2. **Security scanning** - Vulnerability detection and secrets scanning
3. **Dependency analysis** - Third-party package vulnerability scanning
4. **Code quality metrics** - Complexity, maintainability, and technical debt analysis
5. **Dead code detection** - Identify and help remove unused code
6. **Test coverage analysis** - Track code coverage trends and identify untested code
7. **Coverage-guided dead code detection** - Correlate untested code with potentially unused code
8. **Unified reporting** - Single dashboard for all analysis results with coverage integration
9. **CI/CD integration** - Automated quality gates and failure prevention

### Success Criteria
- [ ] All commits automatically validated with 5+ analysis tools
- [ ] Security vulnerabilities detected before code review
- [ ] Dependency vulnerabilities caught within 24 hours of disclosure
- [ ] Dead code identified and removal process automated
- [ ] Test coverage maintained above 90% with trend tracking
- [ ] Coverage-guided dead code detection identifies untested+unused code
- [ ] <10 second feedback loop for common issues
- [ ] 100% of critical paths covered by static analysis
- [ ] Zero false positives for security-critical checks
- [ ] Unified HTML/web dashboard showing all analysis results with coverage metrics

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://pre-commit.com/hooks.html
  why: Pre-commit framework and hook configuration patterns
  critical: Understanding hook execution order and failure handling

- url: https://bandit.readthedocs.io/en/latest/
  why: Security scanning configuration and rule customization
  section: Configuration files and baseline creation
  critical: Avoiding false positives while maintaining security coverage

- url: https://semgrep.dev/docs/writing-rules/overview/
  why: Custom security rule creation for PyDS-specific patterns
  critical: AST-based pattern matching for video analytics security concerns

- url: https://vulture.readthedocs.io/
  why: Dead code detection configuration and whitelist management
  critical: Handling dynamic imports and GStreamer plugin loading

- url: https://coverage.readthedocs.io/
  why: Advanced test coverage analysis and reporting
  section: Coverage combining and branch analysis
  critical: Correlating coverage data with dead code detection

- url: https://mypy.readthedocs.io/en/stable/config_file.html
  why: Advanced MyPy configuration for strict type checking
  section: Incremental mode and performance optimization

- url: https://ruff.rs/docs/configuration/
  why: Ruff rule selection and custom rule development
  critical: Performance tuning for large codebases

- file: src/utils/deepstream.py
  why: Pattern for handling optional dependencies and fallback modes
  critical: Static analysis must handle mock vs real GStreamer imports

- file: src/pipeline/manager.py
  why: Complex async code patterns that need careful type checking
  critical: Handling GStreamer callback types and thread safety

- file: pyproject.toml
  why: Current tool configuration to extend, not replace
  critical: Maintaining compatibility with existing Ruff/MyPy setup
```

### Current Codebase Tree
```bash
pyds-app/
├── src/
│   ├── alerts/           # Alert handling system
│   ├── detection/        # Detection engine with strategies
│   ├── monitoring/       # Health and performance monitoring
│   ├── pipeline/         # GStreamer pipeline management
│   └── utils/           # Core utilities and error handling
├── tests/               # Test suite with pytest
├── scripts/             # Utility scripts
├── examples/            # Example implementations
├── configs/             # YAML configuration files
├── docs/               # User documentation
└── pyproject.toml      # Project configuration with Ruff/MyPy/Black
```

### Desired Codebase Tree
```bash
pyds-app/
├── src/ (unchanged)
├── tests/ (unchanged)
├── .pre-commit-config.yaml          # Pre-commit hook configuration
├── .bandit                          # Bandit security scanner config
├── .semgrep.yml                     # Semgrep custom rules
├── .vulture                         # Dead code detection whitelist
├── scripts/
│   ├── static-analysis.py           # Unified analysis runner
│   ├── security-scan.py             # Security-focused analysis
│   ├── coverage-analysis.py         # Coverage analysis and dead code correlation
│   ├── quality-report.py            # Generate unified reports
│   └── setup-analysis.py            # Development environment setup
├── analysis/
│   ├── rules/
│   │   ├── deepstream-security.yml  # Custom security rules for video analytics
│   │   ├── async-patterns.yml       # AsyncIO and threading analysis
│   │   └── gstreamer-safety.yml     # GStreamer-specific safety checks
│   ├── baselines/
│   │   ├── bandit-baseline.json     # Known security issues baseline
│   │   ├── vulture-whitelist.py     # Approved "dead" code patterns
│   │   └── coverage-baseline.json   # Coverage trends and thresholds
│   ├── coverage/
│   │   ├── history/                 # Coverage trend data over time
│   │   ├── reports/                 # HTML/XML coverage reports
│   │   └── combined/                # Dead code + coverage correlation
│   └── reports/                     # Generated analysis reports
├── .github/
│   └── workflows/
│       ├── static-analysis.yml      # CI pipeline for analysis
│       └── security-scan.yml        # Scheduled security scanning
└── pyproject.toml (extended)        # Additional tool configurations
```

### Known Gotchas of Our Codebase & Library Quirks
```python
# CRITICAL: GStreamer imports are conditionally mocked
# Static analysis must handle both real and mock imports
# See src/pipeline/manager.py lines 15-25 for pattern

# CRITICAL: Pydantic v2 syntax throughout codebase
# Use @field_validator with @classmethod, not @validator
# Pattern in src/config.py and src/detection/models.py

# CRITICAL: AsyncIO patterns with GStreamer callbacks
# Type checkers struggle with GStreamer callback signatures
# Need custom type stubs or ignores for gi.repository

# CRITICAL: Dynamic plugin loading for detection strategies
# Vulture will flag legitimate dynamic imports as dead code
# Need whitelist for entry_points and plugin discovery

# CRITICAL: Optional dependencies (DeepStream, CUDA)
# Analysis must work in environments without GPU/DeepStream
# Pattern: try/except ImportError with AVAILABLE flags

# CRITICAL: Rich console output with ANSI codes
# Security scanners may flag format strings incorrectly
# Need baseline for Rich library usage patterns

# CRITICAL: Windows/Linux compatibility
# Some tools (resource module) are Unix-only
# Analysis setup must handle platform differences
```

## Implementation Blueprint

### Data Models and Structure
```python
# Analysis result aggregation models
@dataclass
class AnalysisResult:
    tool: str
    status: str  # "pass", "fail", "warning"
    issues: List[Issue]
    execution_time: float
    file_count: int

@dataclass 
class Issue:
    severity: str  # "error", "warning", "info"
    category: str  # "security", "quality", "type", "deadcode"
    file_path: str
    line_number: int
    description: str
    rule_id: str
    suggested_fix: Optional[str]

@dataclass
class AnalysisReport:
    timestamp: datetime
    results: List[AnalysisResult]
    summary: AnalysisSummary
    baseline_diff: Optional[BaselineDiff]
```

### List of Tasks to Complete (in order)

```yaml
Task 1: "Enhance development dependencies and tool configuration"
MODIFY pyproject.toml:
  - ADD to [project.optional-dependencies.dev]:
    - "bandit[toml]>=1.7.5"     # Security scanning
    - "semgrep>=1.45.0"         # Advanced security rules  
    - "vulture>=2.10"           # Dead code detection
    - "safety>=2.3.5"           # Dependency vulnerability scanning
    - "pip-audit>=2.6.1"        # Alternative dependency scanner
    - "pre-commit>=3.6.0"       # Git hooks framework
    - "prospector>=1.10.3"      # Analysis aggregator
    - "coverage[toml]>=7.3.0"   # Enhanced coverage analysis
    - "diff-cover>=8.0.0"       # Coverage diff analysis
    - "coverage-badge>=1.1.0"   # Generate coverage badges
  - ADD [tool.bandit] section with exclude patterns
  - ADD [tool.vulture] section with whitelist configuration

Task 2: "Create pre-commit hook configuration"
CREATE .pre-commit-config.yaml:
  - CONFIGURE repos for: ruff, mypy, bandit, vulture, semgrep
  - SET execution order: syntax → types → security → deadcode
  - ENABLE auto-fixing where safe (ruff --fix)
  - CONFIGURE to fail fast on critical security issues

Task 3: "Setup security scanning configuration"
CREATE .bandit:
  - CONFIGURE to scan src/ directory recursively  
  - EXCLUDE tests/ and examples/ from security checks
  - SET confidence level to HIGH for CI, MEDIUM for local
  - CREATE baseline for existing issues

CREATE analysis/rules/deepstream-security.yml:
  - DEFINE rules for GStreamer pipeline injection
  - ADD patterns for unsafe DeepStream element creation
  - INCLUDE video file processing security checks
  - ADD rules for RTSP authentication handling

Task 4: "Configure dead code detection"
CREATE .vulture:
  - WHITELIST dynamic imports from entry_points
  - EXCLUDE GStreamer mock classes (GSTREAMER_AVAILABLE=False)
  - WHITELIST test fixtures and pytest markers
  - INCLUDE custom patterns for plugin discovery

CREATE analysis/baselines/vulture-whitelist.py:
  - DEFINE legitimate "unused" code patterns
  - INCLUDE GStreamer callback function signatures
  - ADD DeepStream version compatibility shims
  - WHITELIST CLI command functions

Task 5: "Create unified analysis runner script"
CREATE scripts/static-analysis.py:
  - IMPLEMENT AnalysisRunner class with tool orchestration
  - ADD parallel execution for independent tools
  - IMPLEMENT result aggregation and deduplication
  - CONFIGURE output formatting (JSON, HTML, terminal)
  - ADD baseline comparison and diff generation

Task 6: "Implement security-focused analysis script"  
CREATE scripts/security-scan.py:
  - COMBINE bandit + semgrep + secrets scanning
  - ADD dependency vulnerability scanning (safety + pip-audit)
  - IMPLEMENT severity-based reporting
  - CONFIGURE for both CI and developer use
  - ADD integration with security baseline

Task 7: "Create quality reporting dashboard"
CREATE scripts/quality-report.py:
  - GENERATE HTML dashboard with all tool results
  - IMPLEMENT trend analysis over time
  - ADD code coverage integration with pytest-cov
  - CREATE maintainability metrics visualization
  - IMPLEMENT email/Slack notifications for critical issues

Task 8: "Setup GitHub Actions CI integration"
CREATE .github/workflows/static-analysis.yml:
  - TRIGGER on pull requests and main branch pushes
  - RUN full analysis suite with caching
  - IMPLEMENT quality gates (fail on security/type errors)
  - UPLOAD analysis reports as artifacts
  - ADD PR comments with analysis summary

CREATE .github/workflows/security-scan.yml:
  - SCHEDULE daily dependency vulnerability scans
  - IMPLEMENT automated security baseline updates
  - ADD alerts for new critical vulnerabilities
  - INTEGRATE with GitHub Security tab

Task 9: "Create development environment setup"
CREATE scripts/setup-analysis.py:
  - INSTALL and configure all analysis tools
  - SETUP pre-commit hooks automatically
  - GENERATE initial baselines for new projects
  - VALIDATE tool configurations
  - CREATE developer onboarding documentation

Task 10: "Implement advanced type checking enhancements"
MODIFY pyproject.toml [tool.mypy]:
  - ADD strict mode configurations
  - CONFIGURE incremental mode for performance
  - ADD custom type stubs for GStreamer/DeepStream
  - IMPLEMENT namespace packages support
  - ADD plugin configurations for pydantic and pytest
```

### Per Task Pseudocode

```python
# Task 5: Unified Analysis Runner
class AnalysisRunner:
    def __init__(self, config_path: str = "pyproject.toml"):
        # PATTERN: Load configuration using existing config patterns
        self.config = load_analysis_config(config_path)
        self.tools = self._initialize_tools()
    
    async def run_all_analysis(self, paths: List[str]) -> AnalysisReport:
        """Run all configured analysis tools in parallel"""
        # PATTERN: Use async task management from src/utils/async_utils.py
        tasks = []
        for tool in self.tools:
            task = asyncio.create_task(self._run_tool(tool, paths))
            tasks.append(task)
        
        # CRITICAL: Handle tool failures gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # PATTERN: Use error handling from src/utils/errors.py
        return self._aggregate_results(results)

# Task 6: Security Analysis
class SecurityScanner:
    def __init__(self):
        self.bandit = BanditScanner()
        self.semgrep = SemgrepScanner() 
        self.secrets = SecretsScanner()
        self.deps = DependencyScanner()
    
    async def comprehensive_scan(self, paths: List[str]) -> SecurityReport:
        """Run all security tools and combine results"""
        # CRITICAL: Security tools must run in isolation
        # Some tools modify file system (semgrep --autofix)
        
        # PATTERN: Use performance context from src/utils/logging.py
        with performance_context("security_scan"):
            security_results = await asyncio.gather(
                self.bandit.scan(paths),
                self.semgrep.scan(paths, rules=self.custom_rules),
                self.secrets.scan(paths),
                self.deps.scan_dependencies()
            )
        
        return self._prioritize_by_severity(security_results)

# Task 7: Quality Report Generation  
class QualityReporter:
    def generate_html_report(self, analysis_report: AnalysisReport) -> str:
        """Generate comprehensive HTML dashboard"""
        # PATTERN: Use Rich console for beautiful output (see main.py)
        
        # CRITICAL: HTML output must be XSS-safe
        # Escape all file paths and code snippets
        template = self._load_template("dashboard.html")
        
        context = {
            "summary": analysis_report.summary,
            "issues_by_severity": self._group_by_severity(analysis_report),
            "trends": self._calculate_trends(analysis_report),
            "coverage": self._get_coverage_data()
        }
        
        return template.render(**context)
```

### Integration Points
```yaml
PYPROJECT_TOML:
  - extend: [tool.bandit] configuration section
  - add: [tool.vulture] exclude patterns
  - modify: [tool.mypy] for stricter checking
  - add: [tool.semgrep] custom rule paths

PRE_COMMIT_HOOKS:
  - integrate: with existing git workflow
  - order: ruff → mypy → bandit → vulture → semgrep
  - performance: cache tool results between runs
  - developer_experience: fast feedback for common issues

CI_PIPELINE:
  - github_actions: parallel job execution
  - caching: tool installations and analysis results  
  - artifacts: HTML reports and baseline files
  - notifications: Slack/email for critical security issues

DEVELOPER_WORKFLOW:
  - vscode: integrate with Python language server
  - cli: unified command for all analysis tools
  - reporting: real-time feedback during development
  - documentation: clear guidance for fixing issues
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Install development dependencies
uv sync --extra dev

# Verify tool installations
bandit --version
semgrep --version  
vulture --version
pre-commit --version

# Run basic tool validation
ruff check scripts/static-analysis.py --fix
mypy scripts/static-analysis.py
black scripts/static-analysis.py

# Expected: No errors, tools properly installed
```

### Level 2: Tool Configuration Tests
```bash
# Test pre-commit hook configuration
pre-commit install
pre-commit run --all-files

# Test security scanning
bandit -r src/ -f json -o analysis/reports/bandit-test.json
semgrep --config=analysis/rules/ src/

# Test dead code detection  
vulture src/ --exclude src/pipeline/manager.py

# Expected: Tools run without configuration errors
```

### Level 3: Analysis Pipeline Integration
```bash
# Run unified analysis
python scripts/static-analysis.py --all-tools --output analysis/reports/

# Generate quality report
python scripts/quality-report.py --input analysis/reports/ --output docs/quality.html

# Test security-focused scan
python scripts/security-scan.py --baseline analysis/baselines/

# Expected: Reports generated, no critical security issues
```

### Level 4: CI/CD Pipeline Test
```bash
# Simulate GitHub Actions locally
act pull_request

# Test pre-commit hooks with sample changes
echo "# test change" >> src/app.py
git add src/app.py
git commit -m "test pre-commit"

# Expected: Pre-commit hooks run, analysis passes
```

## Final Validation Checklist
- [ ] All analysis tools pass: `python scripts/static-analysis.py --strict`
- [ ] Pre-commit hooks installed: `pre-commit run --all-files`
- [ ] Security baseline established: `bandit -r src/ -f json -o baseline.json`
- [ ] Dead code detection tuned: `vulture src/ --min-confidence 80`
- [ ] Quality dashboard generated: `python scripts/quality-report.py`
- [ ] CI pipeline validates: GitHub Actions run successfully
- [ ] Developer workflow smooth: <10 second feedback for common issues
- [ ] Documentation updated: README.md includes analysis commands

---

## Anti-Patterns to Avoid
- ❌ Don't ignore security baselines - keep them current
- ❌ Don't make pre-commit hooks too slow (>30 seconds)
- ❌ Don't create false positive noise - tune sensitivity
- ❌ Don't skip dependency scanning - supply chain attacks are real
- ❌ Don't make analysis optional - enforce in CI/CD
- ❌ Don't forget to handle Windows/Linux differences
- ❌ Don't overload developers with too many tools at once
- ❌ Don't ignore performance impact on large codebases

## Confidence Score: 9/10

This PRP provides comprehensive context for implementing a production-grade static analysis system. The high confidence comes from:
- **Rich Context**: Detailed tool documentation and codebase-specific patterns
- **Incremental Approach**: Builds on existing Ruff/MyPy foundation
- **Validation Gates**: Multiple levels of testing ensure working implementation
- **Real-world Focus**: Addresses actual PyDS security and quality concerns
- **Developer Experience**: Balances thoroughness with usability

Risk areas: Tool integration complexity and potential performance impact, but these are mitigated through careful configuration and phased rollout.
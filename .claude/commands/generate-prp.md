# Create PRP

## Usage
- With feature file: `/generate-prp PRPs/feature.md`
- With description: `/generate-prp "Add error handling for API rate limits"`
- Combined: `/generate-prp PRPs/feature.md "Additional context about the feature"`

## Input: $ARGUMENTS

Generate a complete PRP for the specified feature implementation with thorough research. If a file path is provided, read it first to understand requirements. If a description is provided, use it as the feature specification. Ensure context is passed to the AI agent to enable self-validation and iterative refinement.

The AI agent only gets the context you are appending to the PRP and training data. Assuma the AI agent has access to the codebase and the same knowledge cutoff as you, so its important that your research findings are included or referenced in the PRP. The Agent has Websearch capabilities, so pass urls to documentation and examples.

## Research Process

1. **Codebase Analysis**
   - Search for similar features/patterns in the codebase
   - Identify files to reference in PRP
   - Note existing conventions to follow
   - Check test patterns for validation approach

2. **External Research**
   - Search for similar features/patterns online
   - Library documentation (include specific URLs)
   - Implementation examples (GitHub/StackOverflow/blogs)
   - Best practices and common pitfalls

3. **User Clarification** (as needed)
   - Specific patterns to mirror and where to find them?
   - Integration requirements and where to find them?
   - Any existing documentation or examples to reference?
   - If there are any remaining ambiguities or uncertainties, ask the user for clarification before proceeding.

## PRP Generation

Using PRPs/templates/prp_base.md as template:

### Critical Context to Include and pass to the AI agent as part of the PRP
- **Documentation**: URLs with specific sections
- **Code Examples**: Real snippets from codebase
- **Gotchas**: Library quirks, version issues
- **Patterns**: Existing approaches to follow

### Implementation Blueprint
- Start with pseudocode showing approach
- Reference real files for patterns
- Include error handling strategy
- list tasks to be completed to fullfill the PRP in the order they should be completed

### Validation Gates (Must be Executable) eg for python
```bash
# Syntax/Style
ruff check --fix && mypy .

# Unit Tests
uv run pytest tests/ -v

```

eg for rust
```bash
# Syntax/Style
cargo fmt --check && cargo clippy --all-targets --all-features -- -D
# Unit Tests
cargo test --all-targets --all-features -- --nocapture
```


*** CRITICAL AFTER YOU ARE DONE RESEARCHING AND EXPLORING THE CODEBASE BEFORE YOU START WRITING THE PRP ***

### Additional Considerations

If the PRP is very large, consider breaking it into smaller PRPs that can be implemented independently. Each PRP should have its own context and validation gates.

*** ULTRATHINK ABOUT THE PRP AND PLAN YOUR APPROACH THEN START WRITING THE PRP ***

## Output
Save as: `PRPs/{feature-name}.md`

## Quality Checklist
- [ ] All necessary context included
- [ ] Validation gates are executable by AI
- [ ] References existing or idiomatic patterns, if applicable
- [ ] Clear implementation path
- [ ] Error handling documented

Score the PRP on a scale of 1-10 (confidence level to succeed in one-pass implementation using claude codes)

Remember: The goal is one-pass implementation success through comprehensive context.

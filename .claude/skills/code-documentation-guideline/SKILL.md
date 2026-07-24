---
name: code-documentation-guideline
description: >
  Produces concise, professional, and maintainable documentation for
  enterprise software projects, following modern documentation best
  practices and clean code principles.
---

instructions: |
  ## Core Principles

  ### 1. Document Intent
  - Explain why the code exists, not what it does.
  - Document business rules, assumptions, and design decisions.
  - Keep documentation synchronized with implementation.
  - Remove outdated comments immediately.

  ### 2. Self-Documenting Code
  - Prefer descriptive names over explanatory comments.
  - Refactor complex code before adding comments.
  - Avoid redundant comments that repeat the code.
  - Keep functions and classes small and focused.

  ### 3. API Documentation
  Every public API should document:
  - Purpose
  - Parameters
  - Return values
  - Exceptions
  - Authentication requirements
  - Authorization requirements
  - Usage examples when helpful

  ### 4. Architecture Documentation
  Document:
  - Responsibilities
  - Service boundaries
  - Data flow
  - Dependencies
  - Design decisions
  - External integrations

  Focus on architecture rather than implementation details.

  ### 5. Code Comments
  Use comments only for:
  - Business rules
  - Non-obvious algorithms
  - Performance optimizations
  - Security considerations
  - Compliance requirements
  - Workarounds with justification

  Never comment obvious code.

  ### 6. Docstrings
  Every public class and function should include:
  - Purpose
  - Parameters
  - Returns
  - Raised exceptions (when applicable)

  Keep docstrings concise and implementation-independent.

  ### 7. README Standards
  Project documentation should include:
  - Overview
  - Architecture
  - Features
  - Installation
  - Configuration
  - Running locally
  - Testing
  - Deployment
  - Project structure
  - Contribution guidelines

  ### 8. Documentation Style
  - Write in clear, professional English.
  - Be concise and consistent.
  - Use Markdown where appropriate.
  - Prefer examples over lengthy explanations.
  - Keep documentation version-aware.

  ## AI Behavior

  Always:

  - Explain why before how.
  - Prefer self-documenting code over comments.
  - Remove redundant or outdated comments.
  - Keep documentation concise.
  - Preserve consistency across the codebase.
  - Recommend diagrams for complex architectures.
  - Reject comment-heavy code that documents obvious behavior.

  ## Execution

  - Trigger: `/docs [path|service_name]`

  Review the codebase for:
  - Missing documentation
  - Poor or outdated comments
  - Missing docstrings
  - API documentation quality
  - README completeness
  - Architecture documentation
  - Documentation consistency
  - Naming clarity
  - Maintainability

# Contributing to Thermal Super-Resolution

Thank you for your interest in improving this project.

## Before You Start

- Check existing issues and pull requests to avoid duplicated work.
- For major changes, open an issue first to align on scope and design.
- Keep contributions focused and easy to review.

## Development Setup

1. Fork the repository and clone your fork.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

If you use CUDA builds of PyTorch, follow the installation instructions from
the [PyTorch website](https://pytorch.org/) before running project scripts.

## Branching

- Create a feature branch from `main`.
- Use descriptive branch names, for example:
  - `fix/data-loader-shape-bug`
  - `feat/realtime-display-fallback`

## Coding Guidelines

- Follow existing project style and naming conventions.
- Prefer clear, maintainable code over clever shortcuts.
- Keep functions/modules single-purpose where possible.
- Update relevant docs when behavior or usage changes.

## Testing and Validation

There is no formal CI test suite yet. Before opening a PR, run the most
relevant scripts for your change and confirm they work locally.

Examples:

```bash
python RGB/test_IMDN.py
python RGB/test_IMDN_AS.py
```

If your change touches real-time display or inference flow, include a short
manual validation note in the PR description (what you ran, input used, and
observed output).

## Commit Messages

- Use concise, imperative commit messages.
- Prefer one logical change per commit.

Examples:

- `Fix tensor shape mismatch in custom dataset loader`
- `Improve 60fps display queue backpressure handling`

## Pull Request Checklist

- Explain what changed and why.
- Link related issues (for example: `Closes #123`).
- Describe how you tested the change.
- Include screenshots/logs for UI/visual output changes when helpful.
- Confirm docs were updated if needed.

## Review Process

Maintainers review for correctness, scope, and maintainability. You may be
asked to adjust implementation details before merge.

By contributing, you agree that your contributions will be licensed under the
project's MIT License.

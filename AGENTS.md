# AGENTS.md

## Code Quality

- All code changes MUST follow the `clean-code` skill.
- All code writing and refactoring MUST use the `code-simplification` skill to review and simplify the result.
- Read applicable `SKILL.md` files before implementation.
- Prefer readable, simple, maintainable code.
- Avoid unnecessary abstractions, duplication, and complexity.
- Preserve existing behavior unless explicitly requested otherwise.
- Keep changes minimal and focused.

## Formatting

MUST format all modified source files using the corresponding tools:

| Language   | Extensions                                       | Formatter                               |
| ---------- | ------------------------------------------------ | --------------------------------------- |
| Shell      | `.sh`, `.bash`                               | `shfmt`                               |
| Python     | `.py`, `.pyi`                                | `black`                               |
| Fortran    | `.f90`, `.F90`, `.f95`, `.f03`, `.f08` | `fprettify`                           |
| PowerShell | `.ps1`, `.psm1`, `.psd1`                   | `Invoke-Formatter` (PSScriptAnalyzer) |

Rules:

- Format only modified files.
- Respect existing project formatting configurations.
- Review diffs after formatting.
- If a formatter is unavailable, report it explicitly.
- Do not automatically format legacy fixed-form Fortran.

## Workflow

1. Read relevant instructions and skills.
2. Implement the smallest correct change.
3. Apply `clean-code` and `code-simplification`.
4. Format modified files.
5. Run relevant tests and checks.
6. Review the final diff.

## Completion

Report changes, skills used, formatting performed, test results, and any unresolved issues. Never claim unexecuted checks passed.

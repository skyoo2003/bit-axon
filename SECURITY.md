# Security Policy

## Supported Versions

| Version | Supported | Notes |
|---------|:----------:|:-------|
| 0.1.x   | :white_check_mark: | Current release |

## Reporting a Vulnerability

**Do not report security vulnerabilities through public issues.** Use one of these channels:

1. **GitHub Private Vulnerability Reporting** (preferred): [https://github.com/skyoo2003/bit-axon/security/advisories/new](https://github.com/skyoo2003/bit-axon/security/advisories/new)
2. **Email**: Contact the maintainer directly via GitHub (see [SUPPORT.md](SUPPORT.md))

## Scope

This policy covers:
- The Bit-Axon Python package (`src/bit_axon/`)
- CLI tools (`bit-axon` command)
- Documentation and build infrastructure
- GitHub Actions workflows

This policy does **not** cover:
- Vulnerabilities in upstream dependencies (report to the respective project)
- Issues specific to Apple MLX framework (report to [apple/mlx](https://github.com/apple/mlx))
- General usage questions or non-security bugs (use [Issues](https://github.com/skyoo2003/bit-axon/issues))

## Response Timeline

| Phase | Timeframe | What to Expect |
|-------|-----------|----------------|
| Acknowledgment | Within 48 hours | Confirmation that the report was received |
| Initial Assessment | Within 7 days | Severity classification and triage |
| Fix Communication | Within 14 days | Planned fix timeline or workaround |
| Fix Delivery | Varies by severity | Patch release or advisory publication |

## Severity Classification

| Severity | Example | Timeline |
|----------|---------|----------|
| **Critical** | Remote code execution, auth bypass | 7 days |
| **High** | Privilege escalation, data exposure | 14 days |
| **Medium** | Information disclosure, DoS | 30 days |
| **Low** | Minor info leak, best practice | Next release |

## Security Measures

- **Dependency Scanning**: [Dependabot](https://docs.github.com/en/code-security/dependabot) runs weekly for known CVEs
- **Secret Scanning**: GitHub secret scanning and push protection enabled
- **Supply Chain**: All dependencies pinned with minimum version constraints in `pyproject.toml`
- **CI Isolation**: GitHub Actions run with minimal permissions (`contents: read` for CI)
- **PyPI Publishing**: Uses [Trusted Publishers](https://peps.python.org/pep-0621/) with OIDC, no stored tokens

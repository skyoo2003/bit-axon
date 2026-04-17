# Governance

## Project Governance

Bit-Axon is maintained by [skyoo2003](https://github.com/skyoo2003) as an independent open-source project under the Apache License 2.0.

## Decision Making

| Decision Type | Process |
|---------------|---------|
| Bug fixes | Open issue → triage → fix PR → merge |
| Features | Discussion → RFC (if significant) → PR → review → merge |
| Breaking changes | PR must include migration guide and label `breaking-change` |
| Dependencies | Dependabot PRs reviewed weekly; major bumps assessed for compatibility |

## Release Process

1. **Development**: Changes accumulate on `main` branch
2. **Release Drafter**: Automatically drafts release notes from merged PRs (categorized by type)
3. **Manual Review**: Maintainer reviews drafted notes, adjusts version bump if needed
4. **Publish**: GitHub release created → PyPI Trusted Publisher deploys automatically
5. **Post-Release**: Docs site rebuilt, CHANGELOG finalized

### Versioning

- **Semantic versioning** (`MAJOR.MINOR.PATCH`) following [Conventional Commits](https://www.conventionalcommits.org/)
- `breaking-change` label → major bump
- `enhancement` / `feature` label → minor bump
- Default → patch bump

### Release Cadence

No fixed schedule. Releases happen when enough meaningful changes accumulate or when upstream dependency changes require an update.

## Contribution Model

This project accepts contributions from the community. See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

| Area | Process |
|------|---------|
| Code PRs | Requires CI pass + maintainer review |
| Documentation | PR to `docs/` directory, auto-deployed on merge |
| Bug Reports | Triage within 7 days, priority labeled |
| Feature Requests | Must be discussed in Discussions before PR |

## Branch Policy

| Branch | Purpose | Protection |
|--------|---------|-----------|
| `main` | Production | Squash merge only, CI required |
| `gh-pages` | Docs site | Auto-managed, do not push directly |

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). All participants are expected to adhere to its standards.

## Security

Security vulnerabilities are handled per [SECURITY.md](SECURITY.md). Private reporting is available via GitHub Security Advisories.

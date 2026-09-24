# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report them privately through GitHub's
[private vulnerability reporting](https://github.com/drewster99/swift-llm-kit/security/advisories/new).
Include steps to reproduce and the version (git tag) you tested.

You can expect an acknowledgement within a few days. Once a fix is released, the
advisory will be published and credited to you, unless you'd prefer not to be
named.

## Supported versions

Only the latest tagged release receives fixes.

## Things worth knowing

- API keys are stored in the macOS Keychain, not in UserDefaults or on disk.
- `verboseLogging` writes full request and response bodies to `$TMPDIR/SwiftLLMKit-Logs/`, **including auth headers**. Never enable it in production builds. Redaction is tracked as an open issue.

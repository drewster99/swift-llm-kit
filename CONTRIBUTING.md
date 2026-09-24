# Contributing to SwiftLLMKit

Thanks for helping out. Bug reports, docs fixes, tests, and new features are all
welcome.

## Finding something to work on

- **First contribution?** Pick a [good first issue](https://github.com/drewster99/swift-llm-kit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). Each one names the exact files involved and says what "done" looks like.
- **Want something bigger?** Try [help wanted](https://github.com/drewster99/swift-llm-kit/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22), or the planned work in [ROADMAP.md](ROADMAP.md).
- Comment on an issue before you start, so two people don't build the same thing.
- For anything that changes public API or persisted data, please open an issue to discuss the design first.

## Building and testing

You need macOS 15+ and Xcode 16.4+ (Swift 6.1 or newer). There are no external
dependencies.

```sh
git clone https://github.com/drewster99/swift-llm-kit.git
cd swift-llm-kit
swift build
swift test                      # the full suite runs in a few seconds, no network needed
swift test --filter Gemini      # run just one area
```

You can also open `Package.swift` in Xcode and run the tests with ⌘U.

Tests use [Swift Testing](https://developer.apple.com/documentation/testing)
(`import Testing`, `@Test`, `#expect`), not XCTest. The default suite never
calls a real LLM API. Live tests are opt-in and need your own credentials.

CI builds and tests every pull request on both the oldest and the newest
supported toolchain. Some Swift features behave differently between the two, so
a green local build doesn't guarantee a green CI run.

## Code conventions

[`CLAUDE.md`](CLAUDE.md) is the detailed architecture guide. It explains *why*
the code looks the way it does. The rules that matter most:

- **No force unwraps (`!`) and no `try?`** that silently swallow errors. Handle the error or surface it.
- **No silent fallbacks.** If a value is unknown, keep it unknown (`nil`, `.unknown`) rather than inventing a default.
- **Don't break saved data.** Persisted types decode with `decodeIfPresent` so older files still load. See `ModelConfiguration.init(from:)`.
- **Adding a `ProviderAPIType` case?** Update *every* `switch` over it (`SwiftLLMKit.swift`, `ProviderAPIType.swift`, `ModelFetchService.swift`).
- **Adding a model capability?** Follow the checklist in CLAUDE.md under "Capabilities are vendor facts". It needs five edits, and a test fails if you miss one.
- **Public API gets `///` doc comments.** Comments explain *why*, not *what*.
- **Names should be clear where they're used.** Follow the [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/).
- **Every bug fix comes with a test** that fails without the fix.

## Submitting a pull request

1. Fork the repo and create a branch from `main`.
2. Make your change, with tests.
3. Run `swift test` and make sure it passes with **no new warnings**.
4. Open a PR and fill in the template. Link the issue it closes (`Closes #123`).

Keep PRs focused. A small PR that does one thing gets reviewed much faster than
a large one that does several.

## Reporting bugs

Use the [bug report form](https://github.com/drewster99/swift-llm-kit/issues/new?template=bug_report.yml).
The most useful reports include the provider, the model ID, and the request and
response JSON. Set `verboseLogging = true` on your `LLMKitManager` to write them
to `$TMPDIR/SwiftLLMKit-Logs/`.

> Headers (and so API keys) are not written to these logs, but they do contain your full prompts and responses. Remove anything private before attaching them.

Security problems go through [SECURITY.md](SECURITY.md), not public issues.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind.

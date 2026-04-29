# Security policy

## Reporting a vulnerability

Email smigolsmigol@protonmail.com. Please do not open a public issue for security
reports. Use "f3dx-cache security" in the subject line so it does not get lost.

Acknowledgement within 48 hours. For confirmed critical issues, a fix or
mitigation lands within 7 days. Lower-severity reports get a fix in the next
regular release.

If a CVE is warranted, the GHSA goes through GitHub's coordinated disclosure
flow. Reporters who want credit are credited in the release notes.

## Supported versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |

Older versions are not patched. Pin to a recent release.

## Architecture notes for reviewers

`f3dx-cache` is a Rust core (the `f3dx-cache`, `f3dx-replay`, `f3dx-cache-py`
crates) with a PyO3 bridge that ships as an abi3 wheel. The runtime surface:

- No network IO at rest. The library reads and writes a local redb file. There
  is no built-in fetch, no HTTP client, no telemetry callback.
- The on-disk store is a single redb file. redb uses fsync semantics on commit
  for durability.
- Request and response payloads are treated as opaque bytes after RFC 8785 JCS
  canonicalization. The library does not parse user-supplied JSON beyond what
  JCS requires.
- BLAKE3 is used for content addressing only. It is not a MAC and the cache
  does not authenticate writers. Treat the cache file as trusted local state.

If you find a path that violates any of the above, that is the kind of report
this policy is for.

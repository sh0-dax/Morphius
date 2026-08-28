# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Morphius, please report it
privately rather than opening a public issue:

- Use GitHub's [private vulnerability reporting](https://github.com/sh0-dax/Morphius/security/advisories/new)
  feature (Security tab → "Report a vulnerability"), or
- Open a draft security advisory directly.

Please include:
- A description of the vulnerability and its potential impact.
- Steps to reproduce (proof-of-concept code or a minimal repro is ideal).
- The version/commit you tested against.

You should expect an initial response within a few days. Please allow
time for a fix to be released before any public disclosure.

## Scope

Morphius is a fully client-side application (no backend server), so the
most relevant vulnerability classes are:

- XSS or injection issues in chat rendering, i18n string interpolation,
  or session import/export.
- Weaknesses in the AES-GCM key-vault implementation (see
  [PRIVACY.md](PRIVACY.md) for the intended design).
- Service worker / cache-poisoning issues affecting the offline app
  shell.
- Anything that could exfiltrate a stored API key or chat history to an
  unintended origin.

Vulnerabilities in third-party dependencies (Three.js, WebLLM, etc.)
should generally be reported upstream, but flagging them here is
welcome too if Morphius's usage makes the impact worse than default.

## Supported Versions

Only the latest commit on `main` is actively supported, since this
project has no separate long-term-support branches.

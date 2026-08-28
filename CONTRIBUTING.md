# Contributing to Morphius

Thanks for considering a contribution — bug reports, new locales, face
models, and code are all welcome.

## Before you start

Morphius is intentionally **zero-build**: everything is plain ES
modules loaded directly by the browser. Please keep it that way — avoid
introducing a bundler, transpiler, or framework dependency unless
discussed in an issue first.

## Development setup

```bash
git clone https://github.com/sh0-dax/Morphius.git
cd Morphius
npm install
npx serve .          # or open index.html directly (some features need a secure context)
npm test              # vitest run — must stay green
```

## Making a change

1. Fork the repo and create a branch: `git checkout -b feature/my-feature`.
2. Make your changes.
3. Run `npm test` — all suites must pass (see `tests/`).
4. If you touched face states, visemes, or the render loop, sanity-check
   visually in a real browser — the unit tests don't cover rendering.
5. Update `README.md` if you changed public behavior (a provider, a
   setting, the runtime API).
6. Open a pull request describing **what** changed and **why**. Small,
   focused PRs are much easier to review than large ones.

## Adding a new face model

- Model must be GLB with a Mixamo-compatible rig and ARKit/Oculus
  viseme blend shapes (52 morph slots) — see `models/README.md`.
- Add an entry to `models/manifest.json`.
- Test blinking, breathing, and at least 3 face states with the model
  before submitting.

## Adding a locale

- Copy `i18n/en.json` and translate all 145 keys — don't drop any
  (the parity test in `tests/i18n.test.mjs` will fail if you do).
- Keep the JSON BOM-free.

## Reporting bugs

Open a [GitHub issue](https://github.com/sh0-dax/Morphius/issues) with:
- Browser + OS + GPU (relevant for WebGPU/WebLLM issues).
- Steps to reproduce.
- Console errors, if any.

## Code of conduct

Be respectful. Disagree on code, not on people.

# AI Face — Voice & Vision Assistant Interface

![License](https://img.shields.io/badge/license-MIT-7dd3fc)
![No Build Step](https://img.shields.io/badge/build-none%20required-00d4aa)
![WebGPU](https://img.shields.io/badge/WebLLM-runs%20100%25%20in--browser-ffaa00)
![PWA](https://img.shields.io/badge/installable-PWA-2f81f7)

Voice and vision assistant interface running entirely in the browser: a 3D face (Three.js) that syncs in real-time with speech, optional SiriWave alternative, text and voice chat, and multi-provider model support — including **fully local mode without any API key** via WebLLM.

## Features

- **Live 3D face** built on Three.js + morph targets, with auto states (idle / listening / thinking / responding / alert).
- **SiriWave alternative** toggleable from settings, without changing the default face.
- **Multi-provider support**: Gemini, OpenAI, Meta AI, BazaarLink, Ollama (local), any OpenAI-compatible endpoint, and **WebLLM** (100% local in-browser inference via WebGPU, no server, no key).
- **Lip-synced voice output** via Web Speech API, Gemini native voice, or Gemini Live API.
- **Live metrics panel** (waveform, energy, frequency) based on real AudioAnalyser data, with "no input" state when disconnected.
- **Encrypted local API key storage**: AES-GCM via Web Crypto with non-extractable key in IndexedDB — never stored as plain text.
- **Installable PWA** with offline app shell via service worker.
- **Accessibility**: `aria-live` regions for chat and captions, `prefers-reduced-motion` support, visible keyboard focus (`:focus-visible`).
- **Onboarding flow** for first-time users with step-by-step setup guide.
- **Chat UX**: message timestamps, copy button, regenerate button, scroll-to-bottom indicator.

## Quick Start

No build tools required. Just serve the folder:

```bash
npx serve .
# or open index.html directly, or deploy the folder to GitHub Pages
```

> Note: Some features (microphone, WebLLM via WebGPU) require a secure context (`https://` or `localhost`) and may not work from `file://` in all browsers.

## WebLLM Mode (No API Key)

Select **WebLLM** from the provider list. The browser downloads the model weights (~1–2 GB depending on model) once via WebGPU, then caches them. All inference runs locally afterward with zero data sent to any server. Requires a modern browser with WebGPU support (latest Chrome/Edge).

## Project Structure

```
/
├── index.html            # Main HTML (minimal — references external CSS/JS)
├── css/
│   └── styles.css        # All styles (tokens, layout, components, responsive)
├── js/
│   └── app.js            # App logic (Three.js, chat, voice, settings, LLM bridge)
├── assets/
│   └── icon.svg          # App icon (SVG)
├── sw.js                 # Service worker (app-shell cache)
├── manifest.json         # PWA manifest
├── deploy.yml            # GitHub Pages deployment workflow
├── LICENSE               # MIT license
└── README.md
```

## Architecture

```
┌─────────────────────────────────────────┐
│  UI Layer (index.html + styles.css)     │
│  Chat · Settings · HUD · Onboarding     │
├─────────────────────────────────────────┤
│  App Logic (app.js)                     │
│  Face Engine · Audio Engine · Providers  │
│  Speech I/O · Settings · State Machine  │
├─────────────────────────────────────────┤
│  Providers                               │
│  Gemini · OpenAI · Ollama · WebLLM       │
│  Custom OpenAI-compatible endpoints      │
├─────────────────────────────────────────┤
│  Security (AES-GCM + IndexedDB)          │
└─────────────────────────────────────────┘
```

## Security Note

API keys are encrypted locally and never stored as plain text, but this is defense-in-depth — any code running on the same page can still call the decryption function, since decryption must happen client-side to use the key. For multi-user deployments, put the provider key behind a server-side proxy.

## Roadmap

- [ ] Optional migration to Three.js `WebGPURenderer` + TSL for deeper visual effects.
- [ ] Split `app.js` into separate ES modules (`face.js`, `audio.js`, `providers.js`).
- [ ] Automated tests (Vitest / Playwright) and CI.
- [ ] i18n via JSON translation files instead of embedded strings.

## License

MIT — see [LICENSE](./LICENSE).

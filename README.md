# Morphius — Reactive AI Face & Voice Assistant

![Version](https://img.shields.io/badge/version-6.0.0-7dd3fc)
![License](https://img.shields.io/badge/license-MIT-7dd3fc)
![No Build Step](https://img.shields.io/badge/build-none%20required-00d4aa)
![Status](https://img.shields.io/badge/status-stable-2f81f7)

![Tests](https://img.shields.io/badge/tests-47%2F47%20passing-22c55e)
![E2E](https://img.shields.io/badge/e2e-126%20checks%20passing-22c55e)
![WebLLM](https://img.shields.io/badge/WebLLM-100%25%20in--browser-ffaa00)
![i18n](https://img.shields.io/badge/i18n-6%20locales-ec4899)
![PWA](https://img.shields.io/badge/installable-PWA-2f81f7)
![Pages](https://img.shields.io/badge/deploy-GitHub%20Pages-181717?logo=github)

> A 3D voice & vision assistant that runs 100% in the browser — an expressive 3D face (Three.js) that syncs with speech, carries live state (idle / listening / thinking / responding / alert), and supports multi-provider chat with a **fully local, keyless mode** via WebLLM (WebGPU).

---

## ✨ Features

### Face & Expression Engine
- **Live 3D face** on Three.js morph targets — 52 slots resolved per-model, with autonomous **blinking** and **breathing** while idle.
- **Auto states**: `idle` / `listening` / `thinking` / `responding` / `alert` — the face reacts to every stage of a conversation.
- **Viseme playground**: type a letter and watch the mouth shape, or hold to sustain it.
- **5 lighting presets**: Blueprint · Matrix · Warm · Soft · Noir — with smooth ambient transitions.
- **7 bundled GLB models** (FaceCap, ARKit 52, robot, raccoon, android, and more) + optional webcam face-mirror.

### Chat & Providers
- **Multi-provider**: Gemini · OpenAI · Meta AI · Ollama · any OpenAI-compatible endpoint · **WebLLM** (100% local, no key, no server).
- **Multimodal**: attach images (compressed to ≤1024px) and stream vision replies.
- **Persistent sessions**: saved to IndexedDB, with restore, rename, export, and new-chat.
- **Chat UX**: timestamps, copy, regenerate, scroll-to-bottom indicator.

### Voice & Audio
- **Lip-synced TTS**: Web Speech API · Gemini native voice · Gemini Live API.
- **Master audio bus** with volume control and output-device routing (`setSinkId`).
- **Whisper tiers** (tiny / base / small) for speech-to-text.

### Platform & i18n
- **6 languages**: English · Arabic · French · German · Spanish · Japanese — auto-detected + manual override.
- **Installable PWA**: offline app shell via service worker, 192/512 icons, maskable support.
- **Security**: API keys AES-GCM encrypted (non-extractable) in IndexedDB — never plain text.
- **Accessibility**: `aria-live` regions, `prefers-reduced-motion`, visible `:focus-visible`.
- **Onboarding** flow + live metrics panel (waveform, energy, frequency).

---

## 🚀 Quick Start

No build step required — just serve the folder:

```bash
npx serve .
# or open index.html directly, or deploy the folder to GitHub Pages
```

> Some features (microphone, WebLLM via WebGPU) require a secure context (`https://` or `localhost`) and may not work from `file://`.

## 🔒 WebLLM Mode (No API Key)

Pick **WebLLM** from the provider list. The model (~1–2 GB) downloads once over WebGPU and is cached; all inference is local afterwards — **zero data leaves the browser**. Requires latest Chrome/Edge with WebGPU.

---

## 🧠 Architecture

```
┌───────────────────────────────────────────────┐
│  UI Layer (index.html + css/styles.css)        │
│  Chat · Settings · HUD · Onboarding · i18n     │
├───────────────────────────────────────────────┤
│  App Core (js/app.js)                          │
│  Face Engine · Audio Engine · Providers        │
│  State Machine · Settings · Lighting           │
├───────────────────────────────────────────────┤
│  Modules                                      │
│  pure.js · mirror.js · localSpeech.js          │
│  chatStore.js · masterBus.js · progress.js     │
├───────────────────────────────────────────────┤
│  Providers                                     │
│  Gemini · OpenAI · Ollama · WebLLM · Custom    │
├───────────────────────────────────────────────┤
│  Security (AES-GCM + IndexedDB) · PWA (sw.js)  │
└───────────────────────────────────────────────┘
```

```
/
├── index.html            # App shell (correct title branding)
├── css/styles.css        # Tokens, layout, components, responsive
├── js/
│   ├── app.js            # Core logic (face, chat, voice, LLM bridge)
│   ├── pure.js           # DOM-free helpers (feelings, visemes, utils)
│   ├── mirror.js         # Webcam → blendshape mirror
│   ├── localSpeech.js    # Whisper STT · Kokoro/MMS TTS
│   ├── chatStore.js      # IndexedDB session store
│   ├── masterBus.js      # Volume + output-device routing
│   └── progress.js       # Model-download progress
├── i18n/                 # en · ar · fr · de · es · ja (142 keys each)
├── models/               # 7 GLB models + models/manifest.json
├── assets/               # icons (192/512/svg) + screenshots
├── .github/workflows/deploy.yml   # GitHub Pages CD
├── sw.js · manifest.json          # PWA
├── tests/                # Vitest suites (47 tests)
├── package.json · LICENSE · README.md
```

---

## ✅ QA & Testing

| Suite | File | Tests |
|------|------|-------|
| Chat store (IndexedDB) | `tests/chatStore.test.mjs` | 13 |
| Pure helpers | `tests/pure.test.mjs` | 27 |
| i18n parity | `tests/i18n.test.mjs` | 7 |
| **Total (unit)** | | **47** |

Per-phase Playwright E2E gates: **126 checks** across phases A–H (chat persistence, multimodal, models, audio bus, i18n, PWA, idle-life, lighting).

```bash
npm test        # vitest run  → 47/47 passing
```

## 🔒 Security Note

Keys are encrypted locally via AES-GCM and never stored as plain text — but this is defense-in-depth: any page JS can call the decryption, since it must happen client-side. For multi-user deployments, proxy the provider key server-side.

---

## 🗺️ Roadmap

- [ ] Optional migration to Three.js `WebGPURenderer` + TSL for deeper visuals.
- [ ] Split `app.js` into `face.js` / `audio.js` / `providers.js`.
- [ ] CI running Vitest + Playwright on every push.
- [ ] Electron desktop wrapper.

## 📄 License

MIT — see [LICENSE](./LICENSE).

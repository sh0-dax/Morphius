# Morphius — Reactive AI Face & Voice Assistant

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/Morphius_Dark.gif" />
    <source media="(prefers-color-scheme: light)" srcset="assets/Morphius_Light.gif" />
    <img alt="Morphius — Reactive AI Face & Voice Assistant" src="assets/Morphius_Light.gif" width="100%" />
  </picture>
</p>

[![Three.js](https://img.shields.io/badge/Three.js-r167-000000?style=for-the-badge&logo=threedotjs&logoColor=white)](https://threejs.org/)
[![WebLLM](https://img.shields.io/badge/WebLLM-100%25_Local-ffaa00?style=for-the-badge&logo=webgpu&logoColor=white)]()
[![PWA](https://img.shields.io/badge/Installable-PWA-2f81f7?style=for-the-badge&logo=pwa&logoColor=white)]()
[![i18n](https://img.shields.io/badge/i18n-6_Locales-ec4899?style=for-the-badge&logo=google-translate&logoColor=white)]()
[![Tests](https://img.shields.io/badge/Tests-98_passing-00d4aa?style=for-the-badge&logo=vitest&logoColor=white)]()
[![CI](https://github.com/sh0-dax/Morphius/actions/workflows/deploy.yml/badge.svg)](https://github.com/sh0-dax/Morphius/actions/workflows/deploy.yml)
[![Status](https://img.shields.io/badge/Status-v6.0.0_Ready-22c55e?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)

[![Gemini](https://img.shields.io/badge/Gemini-Supported-4285F4?style=for-the-badge&logo=google&logoColor=white)]()
[![OpenAI](https://img.shields.io/badge/OpenAI-Supported-412991?style=for-the-badge&logo=openai&logoColor=white)]()
[![Ollama](https://img.shields.io/badge/Ollama-Supported-000000?style=for-the-badge&logo=ollama&logoColor=white)]()
[![WebLLM](https://img.shields.io/badge/WebLLM_%28Local%29-Supported-ffaa00?style=for-the-badge)]()
[![Whisper](https://img.shields.io/badge/Whisper-Supported-00b8d9?style=for-the-badge&logo=openai&logoColor=white)]()
[![GitHub Pages](https://img.shields.io/badge/Deploy-GitHub_Pages-181717?style=for-the-badge&logo=github&logoColor=white)](https://sh0-dax.github.io/Morphius/)

---

**Table of Contents**

[Why Morphius?](#why-morphius) · [1. System Architecture](#1-system-architecture-overview) · [2. Face & State Engine](#2-face--state-engine) · [3. Voice & Audio](#3-voice--audio) · [4. Chat & Persistence](#4-chat--persistence) · [5. i18n & Platform](#5-i18n--platform) · [7. Quick Start](#7-quick-start) · [8. Providers](#8-providers) · [9. Runtime API](#9-runtime-api--programmatic-control) · [10. Advanced Features](#10-advanced-features) · [11. Browser Support](#11-browser-support) · [12. CI/CD & QA](#12-cicd--qa) · [13. Roadmap](#13-roadmap) · [14. Contributing](#14-contributing) · [15. License](#15-license)

---

## Why Morphius?

**Morphius is a live 3D assistant persona — not another text-only chat widget.**

Typical voice/chat assistants ship a static avatar and play recorded audio. Morphius doesn't. Morphius gives the interface a **face that thinks**: a GPU-rendered head that breathes, blinks on its own, and runs a full 10-state expression machine that reacts to every stage of a conversation.

They render text. Morphius **reacts**.

| Typical widget | Morphius's answer |
|---|---|
| Static image or looping idle animation | **Autonomous idle life** — native-morph blinking + rhythmic breathing driven by real morph slots |
| Blocks of text with no affect | **10 auto states** (idle / greeting / listening / thinking / speaking / responding / alert / error / paying / processing) blended into expressions |
| Robotic, un-synced TTS | **Exact visemes** per phoneme via `visemeFor()`, lip-synced to speech in real time |
| Fixed camera / one look | **5 lighting presets** (Blueprint / Matrix / Warm / Soft / Noir) with smooth ambient lerp |
| Requires an API key to do anything | **Fully local WebLLM mode** — keyless, serverless, 100% in-browser via WebGPU |
| Single language | **6 locales** (en / ar / fr / de / es / ja), auto-detected, parity-test-enforced |
| Text-only memory | **Encrypted IndexedDB sessions** — restore, rename, export, multimodal images |
| No perception | **Real-time AI Vision** — on-device object detection (`vision.js`): YOLO26n one-to-one via WebGPU, COCO-SSD fallback on WebGL. Lives in a Vision status pill with a live detections label; optional avatar "engaged" reaction when a person is detected. Frames never leave the device. |

## 1\. System Architecture Overview

Morphius is a zero-build client-side application. All logic is plain ES modules loaded directly by the browser; a service worker provides an offline app shell and cache-first delivery for pinned dependencies.

### Runtime Request Flow

```mermaid
graph TD
    A["User message"] --> B["chatStore persist"]
    B --> C["state -> thinking"]
    C --> D["Provider router"]
    D -->|cloud| E["Gemini / OpenAI / Ollama"]
    D -->|local| F["WebLLM (WebGPU)"]
    E --> G["stream tokens"]
    F --> G
    G --> H["state -> responding"]
    H --> I["viseme + TTS synthesis"]
    I --> J["face morphs + audio render"]
```

### Core Pillars

| Pillar | Description |
|---|---|
| **Face Engine** | Three.js morph-target rig; **52 slots** resolved per model via `resolveSlotKey()` |
| **State Machine** | 10 discrete states → expression moodlets; `idle` adds autonomous blink/breath |
| **Master Audio Bus** | Pure volume gain + `OutputDevice.setSinkId` routing for every playback source |
| **Viseme Engine** | `visemeFor(ch)` maps each character to a phoneme morph key; typing burst + hold sustain |
| **Encrypted Vault** | AES-GCM via Web Crypto; non-extractable CryptoKey in IndexedDB |
| **Vision Engine** | `vision.js` — on-device object detection: YOLO26n one-to-one ONNX via onnxruntime-web (WebGPU) with COCO-SSD/TensorFlow.js (WebGL) fallback; letterbox preprocess + `parseYoloOneToOne` |
| **PWA Shell** | `sw.js` app-shell cache + installable manifest with 192/512 PNG + SVG icons |

---

## 2\. Face & State Engine

### Auto States

The avatar transitions automatically based on conversation lifecycle. Each state remixes a set of morph targets into an expression:

| State | Trigger | Face behavior |
|---|---|---|
| `idle` | no activity | **autonomous blinking** (+ breathing) |
| `greeting` | app open | eyes widen, brows rise |
| `listening` | mic / input focus | eyes forward, subtle nod |
| `thinking` | awaiting LLM | eyes up, gentle brows |
| `speaking` | TTS / playback | **lip-synced visemes** |
| `responding` | streaming reply | normal engagement |
| `alert` | new token / attention | brows raise |
| `error` | provider failure | brows down |
| `paying` / `processing` | busy | minimized movement |

### Idle Life (Autonomous)

- Blinking runs on the model's **native morph slots** (`eyeBlink_L` / `eyeBlink_R`) — canonical `STATE_TARGETS` never touch them.
- Blink cadence ~2–6 s, ~150 ms closure, clamped per-frame to survive headless rAF throttling.
- Breathing modulates `jawOpen` (0.03–0.08) and a ±0.003 `head.position.y` bob around `headBaseY`.

### Viseme Playground

Type any character and the mouth instantly shapes the corresponding phoneme; **hold** to sustain it. Included in the Advanced settings group.

### Lighting Presets

`LIGHT_PRESETS` — `blueprint` · `matrix` · `warm` · `soft` · `noir` — ambient colors lerp smoothly in the render loop.

---

## 3\. Voice & Audio

### Speech Synthesis (TTS)

- **Web Speech API** — zero-install browser voices.
- **Gemini native voice** — `gemini-2.5-flash-preview-tts`.
- **Gemini Live API** — streaming native audio.

Output routes through the **master audio bus**: a single volume control and output-device selector (`setSinkId`) apply to every source.

### Speech Recognition (STT)

Whisper tiers `tiny` / `base` / `small` via `localSpeech.js` — `setWhisperModel()`.

### Live Metrics

Real `AudioAnalyser` data drives the HUD (waveform · energy · frequency) with a "no input" state when disconnected.

---

## 4\. Chat & Persistence

- **Multimodal**: attach images → compressed to ≤1024px JPEG dataURLs → streamed as OpenAI-style parts, persisted and re-rendered.
- **Sessions**: IndexedDB store via `chatStore.js` — create, restore, rename, export, new-chat.
- **Chat UX**: timestamps, copy button, regenerate, scroll-to-bottom indicator, `aria-live` captions.
- **Programmatic API**: `window.AIFace` exposes `registerSessionMessage`, session control, and face state for embedding.

---

## 5\. i18n & Platform

- **6 locales**: English · العربية · Français · Deutsch · Español · 日本語 — 145 keys each, auto-detected from `navigator.language` with manual override.
- **BOM-free JSON**: enforced to avoid encoding corruption.
- **PWA**: installable, offline app shell, 192/512 + maskable icons, SVG favicon.
- **Accessibility**: `prefers-reduced-motion`, visible `:focus-visible`, `aria-live` regions.

---

## 7\. Quick Start

No build tools. Serve the folder:

```bash
npx serve .
# or open index.html directly, or deploy the folder to GitHub Pages
```

> Some features (microphone, WebLLM via WebGPU) require a secure context (`https://` or `localhost`) — they may not work from `file://`.

Run the test suite:

```bash
npm install
npm test        # vitest run → all suites pass (see CI/CD §12)
```

---

## 8\. Providers

| Provider | Mode | Notes |
|---|---|---|
| **WebLLM** | **Local** | 100% in-browser via WebGPU — **no key, no server**; ~1–2 GB model download once, then cached |
| **Gemini** | Cloud | `gemini` provider with native TTS + Live audio |
| **OpenAI** | Cloud | Chat Completions, streaming |
| **Ollama** | Local | Any locally-served model |
| **OpenAI-compatible** | Cloud/Local | BazaarLink, Meta, custom endpoints, LM Studio, etc. |

WebLLM requires a modern Chrome/Edge with WebGPU and downloads weights on first use.

---

## 9\. Runtime API — Programmatic Control

```js
const AIFace = window.AIFace;

// Inject messages that the face can react to
AIFace.registerSessionMessage('user', 'Hello!');
AIFace.registerSessionMessage('assistant', 'And a smiling expression!');
```

Key internal modules:

| Module | Responsibility |
|---|---|
| `pure.js` | `detectFeeling`, `visemeFor`, `VISEME_KEYS`, `contentToText`, `contentImages`, `buildUserContent`, `geminiContentParts`, `dataUrlMeta` |
| `chatStore.js` | IndexedDB session CRUD + persistence |
| `masterBus.js` | master gain, output device routing |
| `vision.js` | `Vision.init/start/stop/getBackend/describeScene` — on-device object detection (YOLO26n one-to-one via onnxruntime-web/WebGPU, COCO-SSD fallback) |
| `localSpeech.js` | `setWhisperModel`, `applyMasterSettings`, TTS drivers |

---

## 10\. Advanced Features

- **Webcam face mirror** (`mirror.js`) — copy your own expressions to the avatar.
- **AI Vision** (`vision.js`) — opt-in Settings toggle; real-time on-device object detection (YOLO26n one-to-one via WebGPU, COCO-SSD fallback). Live detections label + status pill, pause/resume button, and an optional avatar reaction when a person is detected.
- **SiriWave alternative** — toggleable without changing the default face.
- **Encrypted key vault** — AES-GCM, non-extractable, IndexedDB (never plain text). *Limits: this protects data at rest; it does not stop XSS or in-page code from using the key while a request runs (see [PRIVACY.md](PRIVACY.md)).*
- **Onboarding flow** — first-run step-by-step setup.
- **Model downloads** — progress surfaced to the HUD via `progress.js`.
- **7 bundled GLB models** including FaceCap (CDN), ARKit 52, robot, raccoon, android, and more.

### AI Vision

Toggle **Settings → Vision** and enable **Real-time object detection (AI Vision)** to start on-device object detection from your webcam. Vision is **off by default** (opt-in, to avoid a camera prompt on first launch).

- Backend is chosen **automatically** — YOLO26n one-to-one via onnxruntime-web on **WebGPU** when available, otherwise **COCO-SSD** on WebGL. A `Vision Backend` setting can force either.
- Detections appear live in the `VISION` status pill (e.g. `YOLO (WebGPU) · 1 person`), throttled by `Max detection rate (FPS)`.
- **Pause / Resume** suspends inference without releasing the camera.
- When **React when a person is detected** is on, the avatar adopts a subtle engaged expression while a person is in view.

**Vision drives feeling (sentiment).** Detections map onto the avatar's feeling state:
- A **person present** keeps a warm feeling (`love`) until they leave.
- A **new object class** appearing triggers a short `surprised` burst.
- Known classes feel specific things — animals (`cat`, `dog`, `horse`, …) → `happy`.
- Vision feeling applies whenever detection is active and clears when nothing is detected (or Vision is turned off).

**Vision drives audio (proactive commentary).** With **Speak about what I see** on (default), the avatar may spontaneously comment out loud on what it detects — e.g. "I see a person." or "I see a cat!". Rules keep it from being chatty:
- Only when the avatar is **idle** and not already speaking.
- Only on a **scene change** (a class appears that wasn't there before).
- A **25s cooldown** between comments.
- Local i18n lines (no API cost); fully disabled by unticking **Speak about what I see** or turning Vision off.

- All frames are processed locally — images never leave the device. The int8 model (`models/yolo26n_int8.onnx`) is precached by the service worker.

---

## 11\. Browser Support

| Feature | Requirement |
|---|---|
| Core app / face rendering | Any modern browser with WebGL2 (Chrome, Edge, Firefox, Safari) |
| WebLLM (local models) | Chrome or Edge with WebGPU enabled |
| AI Vision (YOLO) | Chrome or Edge with WebGPU enabled (falls back to COCO-SSD on WebGL) |
| Microphone / STT | Secure context (`https://` or `localhost`) |
| PWA install | Chromium-based browsers; Safari has partial PWA support |

---

## 12\. CI/CD & QA

Every push to `main` runs a **mandatory test gate** (`.github/workflows/deploy.yml` → `test` job): `node --check` on every JS module plus the full vitest suite. If any check fails, the **deploy is blocked** — broken code never reaches production Pages. The `deploy` job only runs after the `test` gate passes.

| Suite | File | Tests |
|------|------|-------|
| Chat store (IndexedDB) | `tests/chatStore.test.mjs` | 13 |
| Pure helpers | `tests/pure.test.mjs` | 47 |
| AI Vision logic (pure) | `tests/visionLogic.test.mjs` | 31 |
| i18n parity | `tests/i18n.test.mjs` | 7 |
| **Total (unit)** | | **98** |

CodeQL static analysis also runs on push/PR (`.github/workflows/codeql.yml`).

Per-phase Playwright E2E gates across development phases A–H (chat, multimodal, models, audio bus, i18n, PWA, idle-life, lighting): **126 checks passing**.

---

## 13\. Roadmap

- [ ] Additional GLB face models and community model contributions
- [ ] More TTS voices per locale
- [ ] Expanded Live API support for additional providers
- [ ] Plugin/extension system for custom states and gestures

> Have an idea? Open an [issue](https://github.com/sh0-dax/Morphius/issues) or start a [discussion](https://github.com/sh0-dax/Morphius/discussions).

---

## 14\. Contributing

Contributions are welcome — bug reports, new locales, face models, or code.

1. Fork the repo and create a branch: `git checkout -b feature/my-feature`
2. Make your changes (no build step — plain ES modules).
3. Run `npm test` and make sure all suites pass.
4. Open a pull request describing what changed and why.

Please keep PRs focused and add tests for new logic where possible.

---

## 15\. License

MIT — see [LICENSE](./LICENSE).

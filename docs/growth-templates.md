# Growth & Community Outreach — Drafts

Ready-to-adapt drafts for the "Growth & Community" improvement track.
Edit before posting — especially the live demo link and any claims about
performance, which you should verify on your own hardware first.

---

## Show HN (Hacker News)

**Title:**
`Show HN: Morphius – a 3D talking AI face that runs 100% in your browser, no server or API key needed`

**Body:**
```
Hi HN, I built Morphius — a client-side 3D assistant face (Three.js)
with lip-sync, a 10-state expression engine, and multi-provider LLM
support (Gemini/OpenAI/Ollama, or fully local via WebLLM + WebGPU).

What makes it different from other "AI avatar" projects I found while
building it: it's zero-build (plain ES modules), needs no backend
server at all, and the WebLLM mode needs no API key — everything runs
in-browser via WebGPU, cached after first load. It's also a real
installable PWA that works offline.

Live demo: https://sh0-dax.github.io/Morphius/
Source: https://github.com/sh0-dax/Morphius

Feedback and issues very welcome — especially on WebGPU compatibility
across different GPUs, since that's the trickiest part to test alone.
```

---

## r/LocalLLaMA

**Title:**
`Built a browser-only 3D talking face for local LLMs (WebLLM/WebGPU, no server, no key)`

**Body:**
```
Wanted a visual front-end for local models that didn't need a Python
backend, Live2D pipeline, or desktop app — so I built Morphius: a
Three.js face with lip-sync and 10 reactive states, wired to WebLLM so
inference runs entirely client-side via WebGPU. Also supports
Gemini/OpenAI/Ollama if you want a cloud model instead.

It's a zero-build PWA — open the page (or install it) and go. Repo has
47 unit tests + Playwright E2E if anyone wants to poke at the
internals.

Demo: https://sh0-dax.github.io/Morphius/
Repo: https://github.com/sh0-dax/Morphius

Curious what local models people are running through it and how the
performance compares across GPUs — that's the part I can't test alone.
```

---

## GitHub repo topics (for discoverability)

Add these under the repo's "About" gear icon → Topics:

```
ai-avatar, webllm, webgpu, three-js, pwa, offline-first, voice-assistant,
lip-sync, text-to-speech, speech-to-text, local-llm, client-side-ai,
progressive-web-app, gemini-api, openai-api
```

---

## Discord / GitHub Discussions setup checklist

- [ ] Enable **Discussions** on the repo (Settings → Features →
      Discussions) — lower friction than Discord for an early-stage
      project, and keeps everything searchable on GitHub itself.
- [ ] Create categories: `Show and tell` (community-built models/
      integrations), `Q&A`, `Ideas` (feature requests, replaces
      scattered issues).
- [ ] Pin a "Roadmap" discussion linking to the README's Roadmap
      section.
- [ ] Only set up a Discord server once Discussions activity justifies
      real-time chat — an empty Discord looks worse than no Discord.

---

## 60-second comparison video — suggested shot list

1. (0:00–0:10) Terminal: `git clone` → `npx serve .` → browser opens.
2. (0:10–0:25) Pick WebLLM in Settings, model downloads (sped up),
   avatar greets you — no key entered anywhere.
3. (0:25–0:40) Quick montage: face states changing (thinking →
   speaking), lighting preset switch, language switch to Arabic/عربي.
4. (0:40–0:55) Split-screen or cut to a competing project's install
   steps (only show their *own* public README/install docs — don't
   fabricate or misrepresent another project).
5. (0:55–1:00) End card: repo URL + "0 servers · 0 build tools".

# Privacy & Data Handling

Morphius is a **client-side application with no backend of its own**. This
document explains exactly what happens to your data and why, so you don't
have to take that on faith.

## Where your data lives

| Data                                   | Storage                          | Leaves the browser?                        |
| --------------------------------------- | ---------------------------------- | --------------------------------------------- |
| Chat messages & sessions                | IndexedDB (`chatStore.js`)         | No — never sent anywhere but your chosen provider |
| API keys (Gemini / OpenAI / etc.)       | IndexedDB, **AES-GCM encrypted**   | Only the request payload goes to the provider you picked; the stored key itself never leaves the browser |
| Uploaded images (multimodal chat)       | IndexedDB (as compressed dataURLs) | Only if you use a cloud provider and send them in a message |
| Selected model / UI settings            | `localStorage`                     | No |
| WebLLM model weights                    | Browser cache (Cache Storage API)  | No — downloaded once from the model host, then served from cache |

## How the key vault works

When you enter a cloud provider's API key in **Settings**, Morphius:

1. Generates a **non-extractable** AES-GCM 256-bit `CryptoKey` via the
   Web Crypto API and stores it in IndexedDB. "Non-extractable" means the
   raw key material can never be read back out of the browser, even by
   Morphius's own code — only used to encrypt/decrypt in place.
2. Encrypts your API key with that `CryptoKey` before writing it to
   IndexedDB. The key is never stored in plain text at rest.
3. Decrypts it in memory only when a request needs it, and only sends it
   to the provider's own API endpoint (e.g. `api.openai.com`) — never to
   any Morphius-operated server, because there isn't one.

### What this protects — and what it does not

Being honest about the limits of client-side encryption:

- **It protects *data at rest*.** Anyone who later reads your IndexedDB or
  local storage (a stolen device, another extension, a backup export, a
  forensic copy) cannot recover the raw key or your plaintext API key.
- **It does NOT protect against XSS or malicious code running in this
  page.** Morphius and any third-party script bundled into the page can
  call `crypto.subtle.decrypt()` on exactly the key it needs, at the exact
  moment it uses the key. A non-extractable `CryptoKey` is a browser-level
  guard against *exporting* the key material — it does not stop code that
  is already executing in the page from using the key. If you navigate to
  a compromised build of this site, that build can read your key while it
  makes a request.
- **It does not protect against provider-side storage.** Once your key
  travels to `api.openai.com` (or the provider you chose), it exists in
  that provider's systems and is subject to *their* security and logging.

Bottom line: AES-GCM at rest + non-extractable key is a meaningful defense
against offline data theft, not a guarantee of absolute security. The
strongest protection for a secret is still **not to store it at all** —
use a local/self-hosted provider (WebLLM, Ollama, a local endpoint) for
sensitive conversations when possible.

## WebLLM (local) mode

When you select **WebLLM** as your provider, inference runs **entirely
inside your browser via WebGPU**. No text you type, no response you
receive, and no API key of any kind is ever transmitted over the network
during a conversation — the only network activity is the one-time model
weight download (cached afterward).

## What Morphius does *not* do

- No analytics, telemetry, or tracking scripts.
- No server-side logging — there is no server.
- No third-party ads or trackers.
- Nothing is synced across devices; clearing your browser storage for
  this site deletes everything, permanently.

## Cloud provider caveats

If you choose a cloud provider (Gemini, OpenAI, Ollama-remote, or an
OpenAI-compatible endpoint), your messages are sent directly from your
browser to that provider's API according to **their** privacy policy and
data retention practices — Morphius has no visibility into or control
over what they do with it. Review the relevant provider's policy before
sending sensitive information.

## Reporting a privacy or security concern

See [SECURITY.md](SECURITY.md) for how to report a vulnerability or
privacy issue responsibly.

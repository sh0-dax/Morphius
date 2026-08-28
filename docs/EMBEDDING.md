# Embedding Morphius in Another Site

Morphius exposes a small, typed runtime API at `window.AIFace` that lets
you drive the face, read state, and react to internal changes from your
own code. No build step, no SDK install — if you load Morphius's own
`index.html`, everything you need is already on the page.

## Option A — Same-origin `window.AIFace` (recommended)

When Morphius runs on the same origin as your app, the full control
surface is available at `window.AIFace`:

```js
// Drive state / animation
window.AIFace.setState('thinking');

// Toggle lip-sync
window.AIFace.setSpeaking(true);

// Make it speak arbitrary text through its own TTS
window.AIFace.speak('Hello from the host page!', 'en-US');

// Push a session message exactly like a chat turn
window.AIFace.registerSessionMessage('assistant', 'Here is what I found...');
```

### Reacting to changes (event bus)

Instead of polling `getStatus()`, subscribe to internal events:

```js
window.AIFace.on('stateChange', ({ state, previous }) => { ... });
window.AIFace.on('speakingChange', ({ speaking }) => { ... });
window.AIFace.on('providerSwitch', ({ provider, previous }) => { ... });
window.AIFace.on('sessionUpdate', ({ reason, sessionId, messageCount }) => { ... });

// `on(...)` returns an unsubscribe function
const off = window.AIFace.on('stateChange', handler);
off(); // stop listening
```

Supported events: `stateChange`, `speakingChange`, `providerSwitch`,
`sessionUpdate`. A handler that throws is isolated and reported to the
console — it never breaks the other listeners.

## Option B — `<iframe>` + `postMessage` (same-origin only)

If you embed Morphius in an `<iframe>`, drive it with `postMessage`:

```html
<iframe id="morphius" src="/Morphius/" width="480" height="480"></iframe>

<script>
  const frame = document.getElementById('morphius').contentWindow;
  frame.postMessage({ source: 'ai-face-control', type: 'state', value: 'thinking' }, '*');
</script>
```

Supported `postMessage` control types: `state`, `speaking`, `token`,
`speakText`. **Cross-origin frames are ignored** — Morphius only accepts
messages from its own origin (a security measure), so this route is for
same-host embeds only. Prefer Option A when possible.

## Which should I use?

| | same-origin window.AIFace | iframe + postMessage |
|---|---|---|
| Setup time | minimal | ~2 minutes |
| Framework agnostic | Yes | Yes |
| Cross-origin allowed | N/A (same page) | No (rejected for security) |
| Full API surface | Full (sessions, messages, events) | State/speaking/token/speakText only |

See [`types/window-aiface.d.ts`](../types/window-aiface.d.ts) for the
full typed API surface (drop it into your TS project for autocomplete).

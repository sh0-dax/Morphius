// ---- Master audio bus (pure, DOM-free) ----
// Single source of truth for the app-wide master volume and output device.
// Engines call the helpers here at playback setup so volume/device changes
// apply consistently to browser TTS, on-device audio and live streams.

let volume = 1.0;
let outputDevice = 'default';

export function getMasterVolume() {
  return volume;
}

export function setMasterVolume(v) {
  const n = Number(v);
  volume = (Number.isFinite(n) ? n : 1.0);
  volume = Math.min(1, Math.max(0, volume));
  return volume;
}

export function getOutputDevice() {
  return outputDevice;
}

export function setOutputDevice(id) {
  outputDevice = (typeof id === 'string' && id) ? id : 'default';
  return outputDevice;
}

// Inserts a gain node (pre-wired to the destination) whose gain follows the
// current master volume. Returns the node so callers can keep a live handle.
export function attachMasterGain(ctx, destination) {
  const dest = destination || ctx.destination;
  const g = ctx.createGain();
  g.gain.value = volume;
  g.connect(dest);
  return g;
}

// Routes an AudioContext (or media element) to the selected output when the
// browser supports setSinkId. Resolves true when applied, false otherwise.
export async function routeOutput(ctx) {
  if (!ctx || typeof ctx.setSinkId !== 'function') return false;
  if (outputDevice === 'default') return true;
  try {
    await ctx.setSinkId(outputDevice);
    return true;
  } catch (e) {
    return false;
  }
}
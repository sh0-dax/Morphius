// ============================================================
// M7 — Agent memory interface (Model != Memory).
// This thin layer is the ONLY place the agent talks to history.
// In M7 it reads session metadata via an injected `sessionList()`
// (the browser passes chatStore.listSessions; tests pass a fake).
// M8+ swaps this module for a real long-term memory store — the
// agent.js callers only ever see recall()/store().
// ============================================================

export function createMemory({ sessionList, eventStore }) {
  async function recall(query) {
    const list = typeof sessionList === 'function' ? await sessionList() : [];
    const safe = (Array.isArray(list) ? list : []).filter(Boolean);
    const latest = safe.length ? safe[0] : null;
    return {
      count: safe.length,
      title: (latest && latest.title) || '',
      latest,
    };
  }

  async function store(event) {
    if (typeof eventStore === 'function') await eventStore(event);
    return true;
  }

  return { recall, store };
}
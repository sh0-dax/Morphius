/**
 * Type definitions for Morphius's runtime control surface: window.AIFace
 *
 * Drop this file in your project as `morphius.d.ts` (or add its path to
 * your tsconfig's `include`) to get autocomplete and type-checking when
 * embedding or scripting against a running Morphius instance from an
 * external TypeScript / React / Vue project.
 *
 * These types mirror the real API exposed at the bottom of js/app.js —
 * update them if that surface changes.
 */

export type MorphiusState =
  | 'idle'
  | 'greeting'
  | 'listening'
  | 'thinking'
  | 'speaking'
  | 'responding'
  | 'alert'
  | 'error'
  | 'paying'
  | 'processing';

export type MorphiusRole = 'user' | 'assistant' | 'system' | 'error';

export interface MorphiusMessage {
  role: MorphiusRole;
  content: string;
}

export interface MorphiusSessionSummary {
  id: string;
  title: string;
  updatedAt: number;
}

export interface MorphiusSessionsAPI {
  /** List saved chat sessions (most recent first). */
  list(): Promise<MorphiusSessionSummary[]> | MorphiusSessionSummary[];
  /** Load a session by id and make it the active conversation. */
  load(sessionId: string): Promise<void> | void;
  /** Permanently delete a saved session. */
  remove(sessionId: string): Promise<void> | void;
}

export interface MorphiusStatus {
  /** Current face/expression state. */
  state?: MorphiusState;
  /** Whether audio playback (TTS) is currently active. */
  speaking?: boolean;
  /** Timestamp (performance.now()) of the last recorded activity. */
  lastActivityAt?: number;
  /** True while a streaming LLM response is being written to the face. */
  streamDriven?: boolean;
  [key: string]: unknown;
}

// --- Event bus ---
export interface MorphiusEventMap {
  /** Fired whenever the expression state changes. */
  stateChange: { state: MorphiusState; previous: MorphiusState };
  /** Fired when the lip-sync / speaking flag toggles. */
  speakingChange: { speaking: boolean };
  /** Fired when the active LLM provider is switched. */
  providerSwitch: { provider: string; previous: string };
  /** Fired after a session is restored, saved, or a new chat begins. */
  sessionUpdate: { reason: 'restore' | 'save' | 'new'; sessionId: string | null; messageCount: number };
}

export type MorphiusEventName = keyof MorphiusEventMap;
export type MorphiusEventHandler<K extends MorphiusEventName> = (detail: MorphiusEventMap[K]) => void;

export interface MorphiusAPI {
  // --- Face & state control ---
  /** Force the avatar into a specific expression state. No-op for unknown states. */
  setState(state: MorphiusState): void;
  /** Toggle the lip-sync / speaking animation independent of actual audio. */
  setSpeaking(isSpeaking: boolean): void;
  /** Notify Morphius that a new LLM token has arrived (drives "responding" state + activity timer). */
  onToken(): void;
  /** Reset the face to idle and stop any active speech playback. */
  reset(): void;

  // --- Voice ---
  /** Speak the given text through the configured TTS backend. */
  speak(text: string, lang?: string): void | Promise<void>;
  /** Stop any in-progress text-to-speech playback. */
  stopTTS(): void;
  /** Toggle mute on the master audio bus. */
  toggleMute(): void;

  // --- Model / loading ---
  /** Current model download/load progress, if a model is loading. */
  modelProgress(): number | { loaded: number; total: number } | null;

  // --- Status ---
  /** Read the current internal status object (state, speaking, timers, etc). */
  getStatus(): MorphiusStatus;

  // --- Event bus ---
  /** Subscribe to an internal event. Returns an unsubscribe function. */
  on<K extends MorphiusEventName>(event: K, handler: MorphiusEventHandler<K>): () => void;
  /** Unsubscribe a handler from an event. */
  off<K extends MorphiusEventName>(event: K, handler: MorphiusEventHandler<K>): void;

  // --- Chat & sessions ---
  /** Start a new chat session, clearing the active conversation. */
  newChat(): void;
  /** Persist the currently active session to IndexedDB. */
  saveCurrentSession(): void | Promise<void>;
  /** Restore a previously saved session by id. */
  restoreSession(sessionId: string): void | Promise<void>;
  /** Open the built-in sessions browser modal. */
  openSessions(): void;
  /** Get the sanitized message history for the active session. */
  getMessages(): MorphiusMessage[];
  /** CRUD helpers for saved sessions. */
  sessions: MorphiusSessionsAPI;
  /**
   * Inject a message into the active session as if it came from the given
   * role, updating the transcript, persistence, and face reaction.
   * Returns false if the role is invalid or content is empty.
   */
  registerSessionMessage(role: MorphiusRole, content: string): boolean;
}

declare global {
  interface Window {
    AIFace: MorphiusAPI;
  }
}

export {};

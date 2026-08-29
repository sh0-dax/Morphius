// ============================================================
// AI Face — Projection subsystem
// Self-contained add-on that projects images and 3D models
// (GLB/GLTF/VRM) into the Three.js scene as holograms, with a
// raycast gesture layer (pan / pinch-scale / rotate / double-tap
// reset / wheel) that plays nice with the existing OrbitControls.
//
// The host app (app.js) owns the "show mode" behaviours (face-hide,
// stage-expand, auto-frame, free-orbit, restore) and is notified via
// onProjectionChange({ count, hasImage, hasModel }). This module keeps
// the model/image rendering, gestures and anime.js v4 polish (entry /
// float / exit / speaking-pulse) internal, degrading cleanly when
// anime is unavailable or prefers-reduced-motion is set.
//
// Pure math (aspect fit, scale clamp, gesture classification)
// lives in pure.js so it can be unit-tested under Node.
// ============================================================

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clampProjectionScale, projectionFitAspect, classifyProjectionGesture } from './pure.js';

let anime = null;
try {
  const m = await import('animejs');
  anime = m.animate || m.default || m;
} catch (e) {
  anime = null;
}

const FRAME_COLOR = 0x00ffc8;
const SCALE_MIN = 0.3;
const SCALE_MAX = 4;

function prefersReducedMotion() {
  return typeof window !== 'undefined' && window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function createProjectionManager(opts = {}) {
  const {
    scene, camera, renderer,
    headRef = () => null,
    onStatus = () => {},
    onProjectionChange = () => {},
  } = opts;

  const noopApi = { projectImage: () => null, projectModel: () => null, clear: () => {}, destroy: () => {}, dispose: () => {} };
  if (!scene || !camera || !renderer) return noopApi;

  const dom = renderer.domElement;
  const root = new THREE.Group();
  root.name = 'projection-root';
  scene.add(root);

  const items = new Set();
  let modelItem = null;
  const reducedMotion = prefersReducedMotion();

  const tmpVec = new THREE.Vector3();
  const raycaster = new THREE.Raycaster();

  let gesture = null;
  let pinch = null;
  let lastTap = 0;

  function hasImageItem() {
    for (const it of items) if (it.userData.type === 'image') return true;
    return false;
  }
  function emitItems() {
    onProjectionChange({ count: items.size, hasImage: hasImageItem(), hasModel: !!modelItem });
  }

  function isProjectionObj(obj) {
    let o = obj;
    while (o) {
      if (o.userData && o.userData.isProjection) return true;
      o = o.parent;
    }
    return false;
  }

  function pickProjection(clientX, clientY) {
    const rect = dom.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    tmpVec.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    tmpVec.y = -(((clientY - rect.top) / rect.height) * 2 - 1);
    raycaster.setFromCamera(tmpVec, camera);
    const hits = raycaster.intersectObjects(root.children, true);
    const h = hits.find((x) => isProjectionObj(x.object));
    return h ? findProjectionItem(h.object) : null;
  }

  function findProjectionItem(obj) {
    let o = obj;
    while (o) {
      if (o.userData && o.userData.isProjection) return o;
      o = o.parent;
    }
    return null;
  }

  function dist2(a, b) {
    const dx = a.clientX - b.clientX, dy = a.clientY - b.clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }
  function angle2(a, b) {
    return Math.atan2(b.clientY - a.clientY, b.clientX - a.clientX);
  }

  // ---- animation helpers (graceful anime fallback) ----
  function animateObj(target, props) {
    if (!anime) return;
    try { anime({ targets: target, ...props }); } catch (e) {}
  }

  function entryAnimate(item) {
    item.scale.setScalar(0.001);
    if (anime && !reducedMotion) {
      try {
        anime({
          targets: item.scale,
          x: [0, 1.15, 1], y: [0, 1.15, 1], z: [0, 1.15, 1],
          duration: 700, ease: 'outBack',
        });
      } catch (e) { item.scale.setScalar(1); }
    } else {
      item.scale.setScalar(1);
    }
  }

  function floatAnimate(item, originY) {
    if (reducedMotion) return;
    let paused = false;
    let rafId = 0;
    const t0 = performance.now();
    const loop = () => {
      if (paused || !item.parent) return;
      const t = (performance.now() - t0) / 1000;
      item.position.y = originY + Math.sin(t * 1.2) * 0.04;
      rafId = requestAnimationFrame(loop);
    };
    item.userData.pauseFloat = () => { paused = true; cancelAnimationFrame(rafId); };
    item.userData.resumeFloat = () => { paused = false; loop(); };
    rafId = requestAnimationFrame(loop);
  }

  // ---- image projection ----
  async function loadTexture(source) {
    if (source instanceof Blob || (source && source.name)) {
      const url = URL.createObjectURL(source);
      try { return await new THREE.TextureLoader().loadAsync(url); }
      finally { setTimeout(() => URL.revokeObjectURL(url), 1000); }
    }
    return new THREE.TextureLoader().loadAsync(String(source));
  }

  function buildImageItem(texture) {
    const iw = texture.image ? texture.image.width : 1;
    const ih = texture.image ? texture.image.height : 1;
    const { w, h } = projectionFitAspect(iw, ih, 1.6);
    const group = new THREE.Group();
    group.userData.isProjection = true;
    group.userData.type = 'image';
    group.userData.baseScale = 1;

    const mat = new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false });
    const geo = new THREE.PlaneGeometry(w, h);
    const plane = new THREE.Mesh(geo, mat);
    group.add(plane);

    const edgeGeo = new THREE.EdgesGeometry(geo);
    const frameMat = new THREE.LineBasicMaterial({ color: FRAME_COLOR, transparent: true, opacity: 0.6 });
    const frame = new THREE.LineSegments(edgeGeo, frameMat);
    group.add(frame);

    const glowGeo = new THREE.PlaneGeometry(w + 0.06, h + 0.06);
    const glowMat = new THREE.MeshBasicMaterial({ color: FRAME_COLOR, transparent: true, opacity: 0.08, side: THREE.BackSide });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.position.z = -0.02;
    group.add(glow);

    const origin = { x: 1.6, y: 0.3, z: 0.5 };
    group.position.set(origin.x, origin.y, origin.z);
    group.lookAt(camera.position);

    items.add(group);
    root.add(group);
    entryAnimate(group);
    floatAnimate(group, origin.y);
    emitItems();
    const head = headRef();
    if (head && head.rotation) {
      const targetRot = Math.atan2(origin.x - (head.position ? head.position.x : 0), origin.z - (head.position ? head.position.z : 0));
      if (anime && !reducedMotion) animateObj(head.rotation, { y: targetRot, duration: 600, ease: 'outQuad' });
      else head.rotation.y = targetRot;
    }
    return group;
  }

  async function projectImage(source) {
    try {
      const tex = await loadTexture(source);
      return buildImageItem(tex);
    } catch (e) {
      onStatus('Image projection failed', 'warn');
      return null;
    }
  }

  // ---- GLB/VRM projection (self-contained loader) ----
  async function projectModel(source) {
    let url;
    let owned = false;
    if (source instanceof Blob || (source && source.name)) {
      url = URL.createObjectURL(source);
      owned = true;
    } else {
      url = String(source);
    }
    try {
      const loader = new GLTFLoader();
      const gltf = await loader.loadAsync(url);
      const model = gltf.scene;

      model.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      model.position.sub(center);
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      const scale = clampProjectionScale(1.4 / maxDim, 0.05, 6);
      model.scale.setScalar(scale);

      const wrapper = new THREE.Group();
      wrapper.name = 'model-projection';
      wrapper.userData.isProjection = true;
      wrapper.userData.type = 'model';
      wrapper.userData.baseScale = 1;
      wrapper.position.set(0, 0.1, 0);
      wrapper.add(model);

      if (modelItem) removeItem(modelItem);
      modelItem = wrapper;
      items.add(wrapper);
      root.add(wrapper);
      modelItem = wrapper;

      entryAnimate(wrapper);
      floatAnimate(wrapper, 0.1);
      emitItems();
      return wrapper;
    } catch (e) {
      onStatus('Projection load failed', 'err');
      return null;
    } finally {
      if (owned && url) setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
  }

  function disposeItem(item) {
    item.traverse((c) => {
      if (c.geometry) c.geometry.dispose();
      const mats = c.material ? (Array.isArray(c.material) ? c.material : [c.material]) : [];
      mats.forEach((m) => { if (!m) return; if (m.map) m.map.dispose(); m.dispose(); });
    });
    if (item === modelItem) modelItem = null;
    emitItems();
  }

  function removeItem(item) {
    if (!item) return;
    if (item.userData.pauseFloat) { try { item.userData.pauseFloat(); } catch (e) {} }
    if (speaking) stopPulse();
    const finish = () => {
      root.remove(item);
      items.delete(item);
      disposeItem(item);
    };
    if (anime && !reducedMotion) {
      try {
        anime({
          targets: item.scale,
          x: 0, y: 0, z: 0,
          duration: 260, ease: 'inBack',
          complete: finish,
        });
        return;
      } catch (e) { /* fall through to immediate */ }
    }
    finish();
  }

  function clear() {
    [...items].forEach(removeItem);
  }

  function resetItem(item) {
    if (!item) return;
    if (item.userData.type === 'model') {
      item.position.set(0, 0.1, 0); item.rotation.set(0, 0, 0);
    } else {
      item.position.set(1.6, 0.3, 0.5); item.rotation.set(0, 0, 0);
    }
    if (anime && !reducedMotion) animateObj(item.scale, { x: 1, y: 1, z: 1, duration: 600, ease: 'outBack' });
    else item.scale.setScalar(1);
  }

  // ---- speaking-pulse (audio-reactive polish, guarded) ----
  let speaking = false;
  let pulseAnim = null;

  function targetItem() {
    if (modelItem) return modelItem;
    for (const it of items) return it;
    return null;
  }

  function startPulse(item) {
    if (!item || !anime) return;
    const base = (item.userData && item.userData.baseScale > 0) ? item.userData.baseScale : Math.max(0.001, item.scale.x || 1);
    pulseAnim = anime({
      targets: item.scale,
      x: [base, base * 1.06], y: [base, base * 1.06], z: [base, base * 1.06],
      duration: 520,
      direction: 'alternate',
      loop: true,
      ease: 'easeInOutSine',
    });
  }

  function stopPulse() {
    if (pulseAnim && pulseAnim.pause) { try { pulseAnim.pause(); } catch (e) {} }
    pulseAnim = null;
    const it = targetItem();
    const base = (it && it.userData && it.userData.baseScale > 0) ? it.userData.baseScale : 1;
    if (it) {
      if (anime && !reducedMotion) {
        try { anime({ targets: it.scale, x: base, y: base, z: base, duration: 350, ease: 'outQuad' }); }
        catch (e) { it.scale.set(base, base, base); }
      } else {
        it.scale.set(base, base, base);
      }
    }
  }

  function setSpeaking(active) {
    speaking = !!active;
    if (speaking) {
      if (anime && !reducedMotion) startPulse(targetItem());
    } else {
      stopPulse();
    }
  }
  // ---- gesture event handling ----
  function gesturePoint(ev, i) {
    return ev.touches ? ev.touches[i] : ev;
  }

  function onPointerDown(ev) {
    const tc = ev.touches ? ev.touches.length : 1;
    const p0 = gesturePoint(ev, 0);
    if (tc >= 2 && gesture) {
      ev.preventDefault();
      ev.stopPropagation();
      pinch = {
        dist: dist2(ev.touches[0], ev.touches[1]),
        angle: angle2(ev.touches[0], ev.touches[1]),
        startScale: gesture.item.scale.x || 1,
        startRot: gesture.item.rotation.y || 0,
      };
      return;
    }
    const item = pickProjection(p0.clientX, p0.clientY);
    if (!item) { gesture = null; pinch = null; return; }
    ev.preventDefault();
    ev.stopPropagation();
    gesture = { item, touchCount: tc, moved: false, lastX: p0.clientX, lastY: p0.clientY };
  }

  function onPointerMove(ev) {
    if (!gesture) return;
    const tc = ev.touches ? ev.touches.length : 1;
    if (tc >= 2 && pinch && gesture.item) {
      ev.preventDefault();
      ev.stopPropagation();
      const d = dist2(ev.touches[0], ev.touches[1]);
      const scale = clampProjectionScale(pinch.startScale * (d / Math.max(1, pinch.dist)), SCALE_MIN, SCALE_MAX);
      gesture.item.scale.setScalar(scale);
      const a = angle2(ev.touches[0], ev.touches[1]);
      gesture.item.rotation.y = pinch.startRot + (a - pinch.angle);
      return;
    }
    const p0 = gesturePoint(ev, 0);
    const dx = p0.clientX - gesture.lastX;
    const dy = p0.clientY - gesture.lastY;
    if (Math.abs(dx) + Math.abs(dy) > 3) gesture.moved = true;
    ev.preventDefault();
    ev.stopPropagation();
    const speed = Math.max(0.5, Math.abs(camera.position.z)) * 0.003 * 4;
    gesture.item.position.x += dx * speed;
    gesture.item.position.y -= dy * speed;
    gesture.lastX = p0.clientX;
    gesture.lastY = p0.clientY;
  }

  function onPointerUp(ev) {
    if (!gesture) return;
    const tc = ev.touches ? ev.touches.length : 0;
    if (tc === 0) {
      const intent = classifyProjectionGesture('pointerup', 0, gesture.moved, null);
      if (intent === 'tap') {
        const now = performance.now();
        if (now - lastTap < 350) { resetItem(gesture.item); lastTap = 0; }
        else lastTap = now;
      }
      if (gesture.item && gesture.item.userData.pauseFloat) gesture.item.userData.pauseFloat();
      gesture = null;
      pinch = null;
    } else if (tc === 1 && gesture.item) {
      const p0 = gesturePoint(ev, 0);
      gesture.touchCount = 1;
      gesture.lastX = p0.clientX;
      gesture.lastY = p0.clientY;
    }
  }

  function onWheel(ev) {
    const item = pickProjection(ev.clientX, ev.clientY);
    if (!item) return;
    ev.preventDefault();
    ev.stopPropagation();
    const dir = ev.deltaY > 0 ? 0.88 : 1.12;
    const s = clampProjectionScale((item.scale.x || 1) * dir, SCALE_MIN, SCALE_MAX);
    if (anime && !reducedMotion) animateObj(item.scale, { x: s, y: s, z: s, duration: 300, ease: 'outQuad' });
    else item.scale.setScalar(s);
  }

  function attachGestures() {
    dom.addEventListener('pointerdown', onPointerDown, true);
    dom.addEventListener('pointermove', onPointerMove, true);
    dom.addEventListener('pointerup', onPointerUp, true);
    dom.addEventListener('pointercancel', onPointerUp, true);
    dom.addEventListener('wheel', onWheel, { capture: true, passive: false });
    dom.addEventListener('touchstart', onPointerDown, { capture: true, passive: false });
    dom.addEventListener('touchmove', onPointerMove, { capture: true, passive: false });
    dom.addEventListener('touchend', onPointerUp, { capture: true, passive: false });
  }

  attachGestures();

  return {
    projectImage,
    projectModel,
    clear,
    setSpeaking,
    items: () => [...items],
    destroy() {
      clear();
      scene.remove(root);
      dom.removeEventListener('pointerdown', onPointerDown, true);
      dom.removeEventListener('pointermove', onPointerMove, true);
      dom.removeEventListener('pointerup', onPointerUp, true);
      dom.removeEventListener('pointercancel', onPointerUp, true);
      dom.removeEventListener('wheel', onWheel, true);
      dom.removeEventListener('touchstart', onPointerDown, true);
      dom.removeEventListener('touchmove', onPointerMove, true);
      dom.removeEventListener('touchend', onPointerUp, true);
    },
  };
}

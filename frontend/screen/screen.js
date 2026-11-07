/* J.A.R.V.I.S screen: an original holographic HUD drawn with the browser's own Canvas (no libraries).
 *
 * window.HUD is the whole interface the app uses:
 *   HUD.setState(state, detail)        listening | hearing | working | speaking | paused | locked | asleep | problem
 *   HUD.setLevel(0..1)                 live sound level (your voice while hearing, J.A.R.V.I.S's while speaking)
 *   HUD.addMessage(who, text, opts)    who: "you" | "jarvis" | "system"; opts.private marks on-screen-only replies
 *   HUD.setTimers([{label, remaining, total}])
 *   HUD.addActivity({time, text, kind})   kind: ok | warn | bad
 *   HUD.setSystems([{name, ok}])
 *   HUD.setVoice({taught, created, samples, bar, on, strict, problem, message})   the Settings page's Voice section
 *   HUD.teach({phase, mode, done, of, tip, message, outcome, can_redo, ...})       progress while teaching your voice
 *   HUD.setup({keys, key_notes, models, mics, mic, voice, autostart, listening, busy, notes, open})   the setup guide
 * With no app attached (opened as a plain file), a demo plays so the look can be judged.
 */
(() => {
  "use strict";

  // ---- state ------------------------------------------------------------------------------------
  const LOOK = {
    starting:  { rgb: [63, 240, 255],  speed: 1.0, word: "Initialising" },
    listening: { rgb: [63, 240, 255],  speed: 1.0, word: "Listening" },
    hearing:   { rgb: [130, 255, 250], speed: 1.8, word: "Hearing you" },
    working:   { rgb: [255, 179, 71],  speed: 3.2, word: "Working" },
    speaking:  { rgb: [63, 240, 255],  speed: 1.4, word: "Speaking" },
    paused:    { rgb: [143, 163, 175], speed: 0.12, word: "Paused" },
    locked:    { rgb: [143, 163, 175], speed: 0.12, word: "Locked" },
    asleep:    { rgb: [143, 163, 175], speed: 0.08, word: "Asleep" },
    problem:   { rgb: [255, 75, 75],   speed: 0.6, word: "Attention" },
  };

  let state = "starting";
  const colour = [63, 240, 255];          // eased towards the current state's colour
  let speed = 1, level = 0, levelTarget = 0, boot = 0, booting = true;
  let lastLevelAt = 0;                    // when the app last sent a real sound level
  let attached = false, stopDemo = null;  // attached = running inside the real J.A.R.V.I.S window
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = (id) => document.getElementById(id);
  const rgba = (a) => `rgba(${colour[0] | 0}, ${colour[1] | 0}, ${colour[2] | 0}, ${a})`;
  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
  const ease = (x) => 1 - Math.pow(1 - clamp(x), 3);

  // ---- canvases ------------------------------------------------------------------------------------
  const backdrop = $("backdrop"), bctx = backdrop.getContext("2d");
  const reactor = $("reactor"), rctx = reactor.getContext("2d");
  let hexLayer = null;
  const particles = Array.from({ length: 70 }, () => ({
    x: Math.random(), y: Math.random(), r: Math.random() * 1.6 + 0.3,
    vx: (Math.random() - 0.5) * 0.004, vy: -Math.random() * 0.006 - 0.001, a: Math.random() * 0.5 + 0.15,
  }));

  function fit(canvas, ctx) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: width, h: height };
  }

  let bsize = { w: 0, h: 0 }, rsize = { w: 0, h: 0 };
  function resize() {
    bsize = fit(backdrop, bctx);
    rsize = fit(reactor, rctx);
    hexLayer = drawHexGrid(bsize.w, bsize.h);
  }

  function drawHexGrid(w, h) {
    const c = document.createElement("canvas");
    c.width = Math.max(1, w);
    c.height = Math.max(1, h);
    const g = c.getContext("2d");
    const s = 26, hh = Math.sqrt(3) * s;
    g.strokeStyle = "rgba(63, 240, 255, 0.035)";
    g.lineWidth = 1;
    for (let row = -1; row * hh * 0.5 < h + hh; row++) {
      for (let col = -1; col * s * 3 < w + s * 3; col++) {
        const x = col * s * 3 + (row % 2 ? s * 1.5 : 0), y = row * hh * 0.5;
        g.beginPath();
        for (let i = 0; i < 6; i++) {
          const a = (Math.PI / 3) * i;
          g.lineTo(x + s * Math.cos(a), y + s * Math.sin(a));
        }
        g.closePath();
        g.stroke();
      }
    }
    return c;
  }

  // ---- backdrop: hex grid, vignette, drifting particles ---------------------------------------------
  function drawBackdrop(dt) {
    const { w, h } = bsize;
    bctx.clearRect(0, 0, w, h);
    if (hexLayer) bctx.drawImage(hexLayer, 0, 0, w, h);
    const v = bctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.1, w / 2, h / 2, Math.max(w, h) * 0.7);
    v.addColorStop(0, rgba(0.05));
    v.addColorStop(0.6, "rgba(0, 0, 0, 0.55)");
    v.addColorStop(1, "rgba(0, 0, 0, 0.92)");
    bctx.fillStyle = v;
    bctx.fillRect(0, 0, w, h);
    for (const p of particles) {
      p.x += p.vx * dt * speed * 0.5;
      p.y += p.vy * dt * (0.4 + speed * 0.3);
      if (p.y < -0.02) { p.y = 1.02; p.x = Math.random(); }
      if (p.x < -0.02) p.x = 1.02; else if (p.x > 1.02) p.x = -0.02;
      bctx.beginPath();
      bctx.fillStyle = rgba(p.a);
      bctx.arc(p.x * w, p.y * h, p.r, 0, Math.PI * 2);
      bctx.fill();
    }
  }

  // ---- the reactor --------------------------------------------------------------------------------
  const rot = { ticks: 0, segs: 0, arcs: 0, dash: 0, inner: 0, sweep: 0, orbit: 0 };

  function ring(ctx, cx, cy, r, from, to, width, alpha, glow = 10) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, from, to);
    ctx.lineWidth = width;
    ctx.strokeStyle = rgba(alpha);
    ctx.shadowColor = rgba(0.9);
    ctx.shadowBlur = glow;
    ctx.stroke();
  }

  // how much of a ring has been drawn during the boot sequence
  const grown = (delay) => ease((boot - delay) / 0.35);

  function drawReactor(t, dt) {
    const { w, h } = rsize;
    const ctx = rctx;
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2, R = Math.min(w, h) * 0.4;
    if (R <= 10) return;
    const s = reduceMotion ? 0.2 : speed;
    rot.ticks += 0.05 * s * dt;
    rot.segs -= 0.12 * s * dt;
    rot.arcs += 0.45 * s * dt;
    rot.dash -= 0.3 * s * dt;
    rot.inner += 0.8 * s * dt;
    rot.sweep += 1.6 * s * dt;
    rot.orbit += 0.08 * s * dt;
    ctx.save();
    ctx.lineCap = "butt";

    // soft glow behind everything
    const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 1.25);
    glow.addColorStop(0, rgba(0.16 + level * 0.12));
    glow.addColorStop(0.45, rgba(0.05));
    glow.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, w, h);

    // 1. outer tick ring
    let g = grown(0.0);
    if (g > 0) {
      const n = 120;
      ctx.shadowColor = rgba(0.8);
      ctx.shadowBlur = 6;
      for (let i = 0; i < n * g; i++) {
        const a = rot.ticks + (i / n) * Math.PI * 2;
        const long = i % 10 === 0;
        const r1 = R * 1.0, r2 = R * (long ? 0.95 : 0.975);
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
        ctx.lineTo(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2);
        ctx.lineWidth = long ? 2 : 1;
        ctx.strokeStyle = rgba(long ? 0.85 : 0.4);
        ctx.stroke();
      }
      ring(ctx, cx, cy, R * 1.04, 0, Math.PI * 2 * g, 1, 0.25, 4);
    }

    // 2. segmented ring
    g = grown(0.12);
    if (g > 0) {
      const n = 24, gap = 0.06;
      for (let i = 0; i < n * g; i++) {
        const a = rot.segs + (i / n) * Math.PI * 2;
        ring(ctx, cx, cy, R * 0.9, a, a + (Math.PI * 2) / n - gap, i % 6 === 0 ? 7 : 4, i % 6 === 0 ? 0.95 : 0.55);
      }
    }

    // 3. sweeping arcs
    g = grown(0.24);
    if (g > 0) {
      [[0, 1.3, 3, 0.9], [2.2, 0.7, 3, 0.7], [3.6, 0.35, 5, 1]].forEach(([off, len, wid, al]) => {
        const a = rot.arcs + off;
        ring(ctx, cx, cy, R * 0.8, a, a + len * g, wid, al, 14);
      });
      ring(ctx, cx, cy, R * 0.775, -rot.arcs * 0.6, -rot.arcs * 0.6 + 4.8 * g, 1, 0.3, 0);
    }

    // 4. radar sweep while hearing or working
    if ((state === "hearing" || state === "working") && boot >= 1) {
      const a = rot.sweep;
      const wedge = ctx.createConicGradient ? ctx.createConicGradient(a - 0.9, cx, cy) : null;
      if (wedge) {
        wedge.addColorStop(0, "rgba(0,0,0,0)");
        wedge.addColorStop(0.14, rgba(0.18));
        wedge.addColorStop(0.145, "rgba(0,0,0,0)");
        ctx.shadowBlur = 0;
        ctx.fillStyle = wedge;
        ctx.beginPath();
        ctx.arc(cx, cy, R * 0.86, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 5. waveform ring: ripples with the live sound level
    g = grown(0.36);
    if (g > 0) {
      const base = R * 0.66, pts = 220;
      const amp = R * (0.012 + level * 0.11);
      ctx.beginPath();
      for (let i = 0; i <= pts * g; i++) {
        const th = (i / pts) * Math.PI * 2;
        const n = Math.sin(th * 6 + t * 4.1) * 0.5 + Math.sin(th * 13 - t * 7.3) * 0.3 + Math.sin(th * 3 + t * 2.2) * 0.2;
        const r = base + n * amp;
        const x = cx + Math.cos(th - Math.PI / 2) * r, y = cy + Math.sin(th - Math.PI / 2) * r;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.lineWidth = 2;
      ctx.strokeStyle = rgba(0.95);
      ctx.shadowColor = rgba(1);
      ctx.shadowBlur = 16;
      ctx.stroke();
    }

    // 6. dashed ring
    g = grown(0.48);
    if (g > 0) {
      ctx.setLineDash([3, 9]);
      ring(ctx, cx, cy, R * 0.55, rot.dash, rot.dash + Math.PI * 2 * g, 2, 0.6, 6);
      ctx.setLineDash([]);
    }

    // 7. inner ring with notches
    g = grown(0.6);
    if (g > 0) {
      for (let i = 0; i < 6 * g; i++) {
        const a = rot.inner + (i / 6) * Math.PI * 2;
        ring(ctx, cx, cy, R * 0.42, a, a + 0.78, 9, 0.85, 18);
      }
    }

    // 8. the core: a glowing triangle-in-circle, like a reactor
    g = grown(0.72);
    if (g > 0) {
      const pulse = 0.85 + Math.sin(t * (state === "paused" ? 0.8 : 2.6)) * 0.08 + level * 0.25;
      const cr = R * 0.24 * pulse * g;
      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, cr * 1.5);
      core.addColorStop(0, "rgba(255,255,255,0.95)");
      core.addColorStop(0.25, rgba(0.9));
      core.addColorStop(1, "rgba(0,0,0,0)");
      ctx.shadowBlur = 0;
      ctx.fillStyle = core;
      ctx.beginPath();
      ctx.arc(cx, cy, cr * 1.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      for (let i = 0; i < 3; i++) {
        const a = -Math.PI / 2 + (i / 3) * Math.PI * 2 - rot.inner * 0.25;
        ctx.lineTo(cx + Math.cos(a) * cr * 0.9, cy + Math.sin(a) * cr * 0.9);
      }
      ctx.closePath();
      ctx.lineWidth = 2;
      ctx.strokeStyle = rgba(0.9);
      ctx.shadowColor = rgba(1);
      ctx.shadowBlur = 20;
      ctx.stroke();
      ring(ctx, cx, cy, cr, 0, Math.PI * 2, 1.5, 0.9, 20);
    }

    // 9. small read-outs orbiting the outside
    if (boot >= 1) {
      ctx.shadowBlur = 0;
      ctx.font = `${Math.max(9, R * 0.045)}px Consolas, monospace`;
      ctx.fillStyle = rgba(0.75);
      ctx.textAlign = "center";
      const now = new Date();
      const labels = [`SYS ${(97 + Math.sin(t * 0.3) * 1.5).toFixed(1)}%`, `LVL ${(level * 100).toFixed(0).padStart(3, "0")}`,
        `T+${now.getSeconds().toString().padStart(2, "0")}`, state.toUpperCase()];
      labels.forEach((text, i) => {
        const a = rot.orbit + (i / labels.length) * Math.PI * 2;
        ctx.fillText(text, cx + Math.cos(a) * R * 1.13, cy + Math.sin(a) * R * 1.13 + 4);
      });
    }
    ctx.restore();
  }

  // ---- animation loop ----------------------------------------------------------------------------
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const look = LOOK[state] || LOOK.listening;
    for (let i = 0; i < 3; i++) colour[i] += (look.rgb[i] - colour[i]) * Math.min(1, dt * 4);
    speed += (look.speed - speed) * Math.min(1, dt * 3);
    if (state === "speaking" && now - lastLevelAt > 300) {
      // J.A.R.V.I.S's own voice has no live level, so give the wave a natural speech rhythm
      levelTarget = 0.25 + 0.45 * Math.abs(Math.sin(now / 115)) * (0.6 + 0.4 * Math.random());
    }
    level += (levelTarget - level) * Math.min(1, dt * 12);
    if (booting) boot = Math.min(1.2, boot + dt / 2.2);
    drawBackdrop(dt);
    drawReactor(now / 1000, dt);
    requestAnimationFrame(frame);
  }

  // ---- panels --------------------------------------------------------------------------------------
  function setState(next, detail = "") {
    if (!LOOK[next]) return;
    if (state === "speaking" && next !== "speaking") levelTarget = 0.04;
    state = next;
    document.body.className = document.body.className.replace(/\bstate-\S+/g, "").trim() + ` state-${next}`;
    $("status-word").textContent = LOOK[next].word;
    $("status-detail").textContent = detail || " ";
    $("pause-btn").textContent = ["paused"].includes(next) ? "Resume listening" : "Pause listening";
  }

  function typeInto(el, text, done) {
    if (reduceMotion) { el.textContent = text; done && done(); return; }
    el.classList.add("typing");
    let i = 0;
    const step = () => {
      i = Math.min(text.length, i + 2);
      el.textContent = text.slice(0, i);
      if (i < text.length) setTimeout(step, 16);
      else { el.classList.remove("typing"); done && done(); }
    };
    step();
  }

  const MAX_MESSAGES = 60;
  function addMessage(who, text, opts = {}) {
    const list = $("conversation");
    const li = document.createElement("li");
    li.className = `msg ${who}` + (opts.private ? " private" : "");
    const tag = document.createElement("span");
    tag.className = "who";
    tag.textContent = who === "you" ? "You" : who === "jarvis" ? "J.A.R.V.I.S" : "System";
    const body = document.createElement("span");
    body.className = "text";
    li.append(tag, body);
    list.append(li);
    while (list.children.length > MAX_MESSAGES) list.firstChild.remove();
    typeInto(body, String(text), () => { list.scrollTop = list.scrollHeight; });
    list.scrollTop = list.scrollHeight;
  }

  let timers = [];
  function fmt(sec) {
    sec = Math.max(0, Math.round(sec));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
  }
  function renderTimers() {
    const list = $("timers");
    list.replaceChildren();
    if (!timers.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "No timers";
      list.append(li);
      return;
    }
    for (const tm of timers) {
      const li = document.createElement("li");
      li.className = "timer";
      const dial = document.createElement("div");
      dial.className = "dial";
      dial.style.setProperty("--p", tm.total ? clamp(tm.remaining / tm.total) : 0);
      dial.textContent = fmt(tm.remaining);
      const label = document.createElement("div");
      label.className = "label";
      label.textContent = tm.label || "Timer";
      li.append(dial, label);
      list.append(li);
    }
  }
  function setTimers(list) {
    timers = (Array.isArray(list) ? list : []).slice(0, 10).map((t) => ({
      label: String(t.label || ""), remaining: Number(t.remaining) || 0, total: Number(t.total) || 0,
    }));
    renderTimers();
  }

  const MAX_ACTIVITY = 40;
  function addActivity(item) {
    const list = $("activity");
    const li = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = item.time || new Date().toTimeString().slice(0, 8);
    const text = document.createElement("span");
    text.className = ["ok", "warn", "bad"].includes(item.kind) ? item.kind : "ok";
    text.textContent = String(item.text || "");
    li.append(time, text);
    list.prepend(li);
    while (list.children.length > MAX_ACTIVITY) list.lastChild.remove();
  }

  function setSystems(list) {
    const ul = $("systems");
    ul.replaceChildren();
    for (const s of list || []) {
      const li = document.createElement("li");
      li.textContent = String(s.name || "");
      if (!s.ok) li.classList.add("down");
      ul.append(li);
    }
  }

  function setLevel(x) {
    levelTarget = clamp(Number(x) || 0);
    lastLevelAt = performance.now();
    const meter = $("guide-level");
    if (meter) meter.style.width = `${Math.round(levelTarget * 100)}%`;
  }

  function tickClock() {
    const now = new Date();
    $("clock-time").textContent = now.toTimeString().slice(0, 8);
    $("clock-date").textContent = now.toLocaleDateString("en-GB", { weekday: "short", day: "2-digit", month: "short", year: "numeric" });
    for (const tm of timers) tm.remaining = Math.max(0, tm.remaining - 1);
    if (timers.length) renderTimers();
  }

  function runBoot(lines, onDone) {
    const el = $("boot-text");
    let i = 0;
    const next = () => {
      if (i >= lines.length) { setTimeout(onDone, 300); return; }
      typeInto(el, lines[i++], () => setTimeout(next, 380));
    };
    next();
  }

  // ---- settings: your voice ----------------------------------------------------------------------
  // The page only presses buttons (bare names, no data). The app answers with setVoice and teach.
  const BUTTONS = ["teach_start", "teach_record", "teach_redo", "teach_cancel", "voice_test", "voice_forget", "voice_status",
    "setup_status", "setup_key_gemini", "setup_key_groq", "setup_key_fish", "setup_get_gemini", "setup_get_groq",
    "setup_get_fish", "setup_test_keys", "setup_models", "setup_mic_test", "setup_autostart_on", "setup_autostart_off",
    "setup_finish"];
  const MAX_MICS = 17;                     // the one choice: a microphone's place in the list the app sent
  let demoPress = null;                    // stands in for the app in demo mode
  function press(name) {
    if (!BUTTONS.includes(name)) return;
    const api = attached && window.pywebview && window.pywebview.api;
    if (api && typeof api[name] === "function") api[name]();
    else if (!attached && demoPress) demoPress(name);
  }
  function choose(name, index) {
    if (name !== "setup_mic" || !Number.isInteger(index) || index < 0 || index >= MAX_MICS) return;
    const api = attached && window.pywebview && window.pywebview.api;
    if (api && typeof api.setup_mic === "function") api.setup_mic(index);
    else if (!attached && demoPress) demoPress(name, index);
  }

  let teaching = false, voiceTaught = false;
  const PHASE_WORDS = {
    preparing: "Getting ready...", waiting: "Press Record when you're ready", getting_ready: "Getting ready...",
    say_it: "Say \"hey jarvis\" now", comparing: "Comparing with other voices...",
    result: "Finished", cancelled: "Cancelled", failed: "Something went wrong",
  };

  function openSettings() {
    $("settings").hidden = false;
    press("voice_status");
    $("settings-close").focus();
  }
  function closeSettings() {
    if (teaching) press("teach_cancel");     // closing the window never leaves a session running
    $("settings").hidden = true;
  }

  function setVoice(s) {
    s = s || {};
    voiceTaught = !!s.taught;
    let text;
    if (s.problem) text = String(s.problem);
    else if (s.taught) {
      text = `Taught on ${String(s.created || "?")} from ${Number(s.samples) || 0} samples. ` +
        `Bar ${Number(s.bar || 0).toFixed(2)}. Voice check ${s.on ? "ON" : "OFF"}${s.strict ? " (strict)" : ""}.`;
    } else text = "Not taught yet: J.A.R.V.I.S answers any voice.";
    $("voice-status").textContent = text;
    $("voice-status").className = "setting-status " + (s.problem ? "bad" : s.on ? "ok" : "warn");
    $("voice-message").textContent = s.message ? String(s.message) : "";
    $("voice-teach").textContent = s.taught ? "Teach again" : "Teach my voice";
    $("voice-test").disabled = !s.taught || teaching;
    $("voice-forget").disabled = !s.taught || teaching;
    $("voice-teach").disabled = teaching;
  }

  function placeOnScale(el, from, to) {
    const a = clamp(Number(from) || 0), b = clamp(Number(to) || 0);
    el.style.left = `${a * 100}%`;
    el.style.width = `${Math.max(0.6, (b - a) * 100)}%`;
  }

  function teach(u) {
    u = u || {};
    const phase = String(u.phase || "");
    const finished = ["result", "cancelled", "failed"].includes(phase);
    teaching = !finished;
    $("teach").hidden = false;
    $("settings").hidden = false;
    const test = u.mode === "test";
    const done = Number(u.done) || 0, of = Number(u.of) || 20;
    const taking = ["waiting", "getting_ready", "say_it"].includes(phase);      // the sample being taken now
    const complete = phase === "comparing" || (phase === "result" && u.saved !== undefined);
    $("teach-count").textContent = test ? "TEST" : complete ? `${of} / ${of}` : `${Math.min(done + (taking ? 1 : 0), of)} / ${of}`;
    $("teach-dial").style.setProperty("--p", test || complete ? 1 : clamp(done / of));
    $("teach-phase").textContent = PHASE_WORDS[phase] || "";
    $("teach-tip").textContent = phase === "waiting" && u.tip && !test ? `Next: say it ${String(u.tip)}` : "";
    const msg = $("teach-message");
    msg.textContent = u.message ? String(u.message) : "";
    msg.className = "teach-message " + (u.outcome === "got_it" || u.recognised || u.saved ? "ok"
      : ["missed", "odd"].includes(u.outcome) || phase === "failed" || u.recognised === false || u.saved === false ? "warn" : "");
    $("teach").classList.toggle("listening", phase === "say_it");

    const showScale = phase === "result" && !test && Array.isArray(u.yours);
    $("teach-result").hidden = !showScale && !(test && u.score !== undefined);
    if (showScale) {
      placeOnScale($("scale-yours"), u.yours[0], u.yours[1]);
      placeOnScale($("scale-others"), 0, u.others);
      $("scale-bar").hidden = u.bar === null || u.bar === undefined;
      if (u.bar !== null && u.bar !== undefined) placeOnScale($("scale-bar"), u.bar, u.bar);
    } else if (test && u.score !== undefined) {
      placeOnScale($("scale-yours"), u.score, u.score);
      placeOnScale($("scale-others"), 0, 0);
      placeOnScale($("scale-bar"), u.bar, u.bar);
      $("scale-bar").hidden = false;
    }

    $("teach-record").hidden = finished;
    $("teach-cancel").hidden = finished;
    $("teach-redo").hidden = finished || test;
    $("teach-done").hidden = !finished;
    $("teach-record").disabled = phase !== "waiting";
    $("teach-redo").disabled = !(phase === "waiting" && u.can_redo);
    $("teach-record").textContent = test ? "Say it" : "Record";
    $("voice-section").classList.toggle("muted", teaching);
    ["voice-teach", "voice-test", "voice-forget"].forEach((id) => { $(id).disabled = teaching || (id !== "voice-teach" && !voiceTaught); });
    if (finished) $("teach-done").focus();
    else if (phase === "waiting") $("teach-record").focus();
  }

  function finishTeaching() {
    $("teach").hidden = true;
    $("voice-section").classList.remove("muted");
    teaching = false;
    press("voice_status");
    if (backToGuide) {                       // taught from the guide: carry on with the next step
      backToGuide = false;
      $("settings").hidden = true;
      openGuide(STEPS.indexOf("voice"));
    }
  }

  // ---- the setup guide -----------------------------------------------------------------------
  // Shows what the app sends with HUD.setup and presses its buttons. It never handles a key.
  const STEPS = ["welcome", "keys", "models", "mic", "voice", "autostart", "done"];
  const STEP_NAMES = ["Welcome", "Keys", "Models", "Microphone", "Your voice", "Start with Windows", "Ready"];
  const KEY_NOTES = { works: "works", refused: "was refused", unreachable: "couldn't be checked (no connection?)" };
  const BUSY_WORDS = { key: "Waiting for the key window...", test: "Testing your keys...", models: "Downloading...",
    mic: "Listening to the microphone...", autostart: "Changing autostart...", finish: "Starting..." };
  let step = 0, guide = {}, backToGuide = false;

  function stepDone(name) {
    const g = guide;
    if (name === "keys") return !!(g.keys && g.keys.gemini && g.keys.groq && g.keys.fish);
    if (name === "models") return !!g.models;
    if (name === "mic") return Number.isInteger(g.mic) && g.mic >= 0;
    if (name === "voice") return !!g.voice;
    if (name === "autostart") return g.autostart === "on";
    if (name === "done") return !!g.listening;
    return false;
  }

  function openGuide(at) {
    $("guide").hidden = false;
    if (Number.isInteger(at)) step = clamp(at, 0, STEPS.length - 1);
    renderGuide();
    press("setup_status");
    $("guide-next").focus();
  }
  function closeGuide() { $("guide").hidden = true; }

  function note(stepName) {
    const n = (guide.notes || {})[stepName];
    const el = document.querySelector(`[data-note="${stepName}"]`);
    if (!el) return;
    el.textContent = n && n.text ? String(n.text) : "";
    el.className = "guide-note " + (n && ["ok", "warn", "bad"].includes(n.kind) ? n.kind : "");
  }

  function status(id, text, kind) {
    const el = $(id);
    el.textContent = text;
    el.className = "setting-status " + kind;
  }

  function renderGuide() {
    const g = guide, busy = String(g.busy || "");
    // the rail of steps
    const rail = $("guide-rail");
    rail.replaceChildren();
    STEPS.forEach((name, i) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = STEP_NAMES[i];
      b.className = (i === step ? "current " : "") + (stepDone(name) ? "done" : "");
      b.addEventListener("click", () => { step = i; renderGuide(); });
      li.append(b);
      rail.append(li);
    });
    document.querySelectorAll(".guide-step").forEach((el) => { el.hidden = el.dataset.step !== STEPS[step]; });

    // keys: set or not, never the key itself
    document.querySelectorAll(".guide-key").forEach((row) => {
      const id = row.dataset.key, set = !!(g.keys && g.keys[id]);
      const tested = (g.key_notes || {})[id];
      const el = row.querySelector("[data-state]");
      el.textContent = set ? (tested ? `Saved, and ${KEY_NOTES[tested] || "didn't work"}` : "Saved") : "Not added yet";
      el.className = "guide-key-state " + (set ? (tested && tested !== "works" ? "warn" : "ok") : "warn");
      row.querySelector("[data-add]").textContent = set ? "Replace" : "Add key";
    });
    // models
    status("guide-models-state", g.models ? "Downloaded and checked." : "Not downloaded yet.", g.models ? "ok" : "warn");
    $("guide-models").textContent = g.models ? "Check again" : "Download";
    // microphones: one button each, chosen by its place in the list
    const mics = $("guide-mics");
    mics.replaceChildren();
    (Array.isArray(g.mics) ? g.mics : []).slice(0, MAX_MICS).forEach((name, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "guide-mic" + (g.mic === i ? " chosen" : "");
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", g.mic === i ? "true" : "false");
      b.textContent = String(name);
      b.disabled = !!busy;
      b.addEventListener("click", () => choose("setup_mic", i));
      mics.append(b);
    });
    ["keys", "models", "mic", "autostart", "done"].forEach(note);
    if (g.mic === -1 && !(g.notes || {}).mic) {
      const el = document.querySelector('[data-note="mic"]');
      el.textContent = "The microphone named in config.py wasn't found. Pick one above.";
      el.className = "guide-note warn";
    }
    // voice
    status("guide-voice-state", g.voice ? "Taught. Only your voice wakes J.A.R.V.I.S." :
      "Not taught yet: any voice can wake J.A.R.V.I.S.", g.voice ? "ok" : "warn");
    $("guide-teach").textContent = g.voice ? "Teach again" : "Teach my voice";
    $("guide-teach").disabled = !g.models || !!busy;
    // autostart
    status("guide-autostart-state", g.autostart === "on" ? "On: J.A.R.V.I.S starts when you sign in." :
      g.autostart === "off" ? "Off." : "Unknown.", g.autostart === "on" ? "ok" : "warn");
    // the summary
    const sum = $("guide-summary");
    sum.replaceChildren();
    STEPS.slice(1, -1).forEach((name) => {
      const li = document.createElement("li");
      li.textContent = STEP_NAMES[STEPS.indexOf(name)];
      li.className = stepDone(name) ? "ok" : "warn";
      sum.append(li);
    });
    $("guide-finish").textContent = g.listening ? "Finish" : "Start listening";
    // buttons wait while something is running (the sign-up links never need to)
    document.querySelectorAll("#guide [data-press]").forEach((b) => {
      b.disabled = !!busy && !b.dataset.press.startsWith("setup_get_");
    });
    $("guide-busy").textContent = BUSY_WORDS[busy] || "";
    $("guide-back").disabled = step === 0;
    $("guide-next").textContent = step === STEPS.length - 1 ? "Close"
      : stepDone(STEPS[step]) || step === 0 ? "Next" : "Skip for now";
  }

  function setup(state) {
    guide = state && typeof state === "object" ? state : {};
    if (guide.open) {
      $("settings").hidden = true;
      $("guide").hidden = false;
      step = 0;
    }
    renderGuide();
  }

  $("guide-open").addEventListener("click", () => { $("settings").hidden = true; openGuide(0); });
  $("guide-close").addEventListener("click", closeGuide);
  $("guide").addEventListener("keydown", (e) => { if (e.key === "Escape") closeGuide(); });
  $("guide-back").addEventListener("click", () => { step = Math.max(0, step - 1); renderGuide(); });
  $("guide-next").addEventListener("click", () => {
    if (step === STEPS.length - 1) { closeGuide(); return; }
    step++;
    renderGuide();
  });
  document.querySelectorAll("#guide [data-press]").forEach((b) => b.addEventListener("click", () => press(b.dataset.press)));
  $("guide-teach").addEventListener("click", () => {
    backToGuide = true;
    closeGuide();
    $("settings").hidden = false;
    press("teach_start");
  });

  $("settings-btn").addEventListener("click", openSettings);
  $("settings-close").addEventListener("click", closeSettings);
  $("settings").addEventListener("keydown", (e) => { if (e.key === "Escape") closeSettings(); });
  $("voice-teach").addEventListener("click", () => press("teach_start"));
  $("voice-test").addEventListener("click", () => press("voice_test"));
  $("voice-forget").addEventListener("click", () => press("voice_forget"));
  $("teach-record").addEventListener("click", () => press("teach_record"));
  $("teach-redo").addEventListener("click", () => press("teach_redo"));
  $("teach-cancel").addEventListener("click", () => press("teach_cancel"));
  $("teach-done").addEventListener("click", finishTeaching);

  // ---- the app's handle --------------------------------------------------------------------------
  const pauseHandlers = [];
  $("pause-btn").addEventListener("click", () => pauseHandlers.forEach((fn) => fn()));
  function attach() {
    // Called once by the real J.A.R.V.I.S window: no demo, and Pause asks the app (its only command).
    if (attached) return;
    attached = true;
    if (stopDemo) stopDemo();
    pauseHandlers.length = 0;
    pauseHandlers.push(() => {
      const api = window.pywebview && window.pywebview.api;
      if (api && api.toggle_pause) api.toggle_pause();
    });
  }

  window.HUD = { setState, setLevel, addMessage, setTimers, addActivity, setSystems, teach, setVoice, setup, attach,
    onPause: (fn) => pauseHandlers.push(fn) };

  window.addEventListener("resize", resize);
  resize();
  setState("starting");
  renderTimers();
  requestAnimationFrame(frame);
  setInterval(tickClock, 1000);
  tickClock();

  function finishBoot() {
    booting = false;
    boot = 1.2;
    document.body.classList.remove("booting");
  }

  // ---- demo (only when no app is attached) -----------------------------------------------------------
  function demo() {
    setSystems([{ name: "Microphone", ok: true }, { name: "Wake word", ok: true }, { name: "Groq", ok: true },
      { name: "Gemini", ok: true }, { name: "Fish Audio", ok: true }, { name: "Audit chain intact", ok: true }]);
    const scenes = [
      { you: "Open notepad", reply: "Opening notepad, sir.", act: ["open_app · ran", "ok"] },
      { you: "Set a timer for two minutes for the tea", reply: "Reminder set for 2 minutes.", act: ["set_timer · ran", "ok"],
        then: () => setTimers([...timers, { label: "the tea", remaining: 120, total: 120 }]) },
      { you: "Read my note called ideas", reply: "Rooftop garden. Ask about the solar panels.", private: true,
        act: ["read_note · ran", "ok"] },
      { you: "Delete report from my desktop", wait: "Waiting for your keypress on the confirmation window…",
        reply: "Cancelled.", act: ["delete_file · not confirmed", "warn"] },
    ];
    let n = 0, paused = false, timersRunning = [];
    const later = (ms, fn) => timersRunning.push(setTimeout(() => !paused && fn(), ms));
    const speech = (ms, loud) => {
      const start = performance.now();
      const id = setInterval(() => {
        const e = performance.now() - start;
        if (e > ms || paused) { clearInterval(id); setLevel(0.04); return; }
        setLevel(clamp(loud * (0.45 + 0.55 * Math.abs(Math.sin(e / 90)) * Math.random())));
      }, 60);
    };
    const play = () => {
      const sc = scenes[n++ % scenes.length];
      setState("listening");
      setLevel(0.04);
      later(1800, () => { addActivity({ text: "wake · hey jarvis 0.91", kind: "ok" }); setState("hearing", "go ahead"); speech(2200, 0.9); });
      later(2300, () => addMessage("you", sc.you));
      later(4300, () => { setState("working", "thinking"); setLevel(0); });
      if (sc.wait) later(5000, () => addMessage("system", sc.wait));
      later(sc.wait ? 7600 : 5600, () => {
        addActivity({ text: sc.act[0], kind: sc.act[1] });
        setState("speaking");
        speech(1800, 0.7);
        addMessage("jarvis", sc.private ? sc.reply : sc.reply, { private: !!sc.private });
        sc.then && sc.then();
      });
      later(sc.wait ? 9800 : 7800, () => setState("listening"));
      later(sc.wait ? 13500 : 11500, play);
    };
    HUD.onPause(() => {
      paused = !paused;
      if (paused) {
        timersRunning.forEach(clearTimeout);
        timersRunning = [];
        setLevel(0);
        setState("paused", "microphone off");
        addActivity({ text: "listening paused", kind: "warn" });
      } else {
        addActivity({ text: "listening resumed", kind: "ok" });
        play();
      }
    });
    // Settings in demo mode: pretend teaching, so the look can be judged (nothing is listened to).
    let demoDone = 0, demoTaught = false;
    const tips = ["normally", "a little quieter", "a little louder", "a bit faster", "from a bit further away"];
    const waitAt = (message, outcome) => teach({ phase: "waiting", mode: "teach", done: demoDone, of: 20,
      tip: tips[Math.floor(demoDone / 4) % tips.length], message, outcome, can_redo: demoDone > 0 });
    demoPress = (name, arg) => {
      if (name === "voice_status") {
        setVoice(demoTaught ? { taught: true, created: "2026-11-05 16:40", samples: 20, bar: 0.42, on: true } : { taught: false });
      } else if (name === "teach_start") {
        demoDone = 0;
        teach({ phase: "preparing", mode: "teach", done: 0, of: 20, message: "Getting ready..." });
        setTimeout(() => waitAt("Press Record, then say \"hey jarvis\" after the chime."), 700);
      } else if (name === "teach_record") {
        teach({ phase: "getting_ready", mode: "teach", done: demoDone, of: 20, message: "Getting ready..." });
        setTimeout(() => teach({ phase: "say_it", mode: "teach", done: demoDone, of: 20, message: "Say \"hey jarvis\" now." }), 900);
        setTimeout(() => {
          demoDone++;
          if (demoDone < 20) { waitAt("Got it.", "got_it"); return; }
          teach({ phase: "comparing", mode: "teach", done: 20, of: 20, message: "Comparing your voice with other voices..." });
          setTimeout(() => {
            demoTaught = true;
            teach({ phase: "result", mode: "teach", done: 0, of: 20, saved: true, yours: [0.58, 0.81], others: 0.21, bar: 0.38,
              message: "Your voice scored 0.58-0.81, other voices at most 0.21. The bar is 0.38." });
          }, 1200);
        }, 2200);
      } else if (name === "teach_redo") {
        demoDone = Math.max(0, demoDone - 1);
        waitAt("Took that one off. Record it again.");
      } else if (name === "teach_cancel") {
        teach({ phase: "cancelled", mode: "teach", message: "Cancelled. Nothing was saved." });
      } else if (name === "voice_test") {
        teach({ phase: "result", mode: "test", score: 0.66, bar: 0.38, recognised: true, message: "That's you." });
      } else if (name === "voice_forget") {
        demoTaught = false;
        setVoice({ taught: false, message: "Your voiceprint was deleted." });
      } else if (name.startsWith("setup_")) {
        demoSetup(name, arg);
      }
    };
    // The setup guide in demo mode: pretend answers (no key box, nothing downloaded or listened to).
    const fake = { keys: { gemini: true, groq: false, fish: false }, key_notes: {}, models: false,
      mics: ["Windows default", "Microphone Array (Realtek(R) Au", "Headset Microphone (USB Audio)"], mic: 1,
      voice: false, autostart: "off", listening: false, busy: "", notes: {} };
    const demoSetup = (name, arg) => {
      const show = () => { fake.voice = demoTaught; setup({ ...fake }); };
      const busyFor = (what, ms, then) => {
        fake.busy = what;
        show();
        setTimeout(() => { fake.busy = ""; then(); show(); }, ms);
      };
      if (name === "setup_status") show();
      else if (name.startsWith("setup_key_")) busyFor("key", 900, () => {
        fake.keys[name.slice(10)] = true;
        fake.notes.keys = { text: "Saved in the Windows Credential Manager.", kind: "ok" };
      });
      else if (name === "setup_test_keys") busyFor("test", 1200, () => {
        Object.keys(fake.keys).forEach((k) => { if (fake.keys[k]) fake.key_notes[k] = "works"; });
        fake.notes.keys = { text: "Every key you added works.", kind: "ok" };
      });
      else if (name === "setup_models") busyFor("models", 1500, () => {
        fake.models = true;
        fake.notes.models = { text: "Downloaded, and every file matched its fingerprint.", kind: "ok" };
      });
      else if (name === "setup_mic") {
        fake.mic = arg;
        fake.notes.mic = { text: "Saved. Now try it: press Test and say something.", kind: "ok" };
        show();
      } else if (name === "setup_mic_test") {
        speech(2200, 0.8);
        busyFor("mic", 2400, () => { setLevel(0); fake.notes.mic = { text: "I heard you clearly.", kind: "ok" }; });
      } else if (name === "setup_autostart_on" || name === "setup_autostart_off") busyFor("autostart", 600, () => {
        fake.autostart = name.endsWith("_on") ? "on" : "off";
        fake.notes.autostart = { text: `Autostart is now ${fake.autostart.toUpperCase()}.`, kind: "ok" };
      });
      else if (name === "setup_finish") busyFor("finish", 800, () => {
        fake.listening = true;
        fake.notes.done = { text: 'All set. Say "hey jarvis".', kind: "ok" };
      });
    };
    addMessage("system", "Demo mode: this is how the screen will look with J.A.R.V.I.S attached.");
    play();
    return () => { paused = true; timersRunning.forEach(clearTimeout); timersRunning = []; };
  }

  runBoot(["Initialising core systems…", "Loading wake word models… OK", "Securing voice channels… OK",
    "All systems online."], () => {
    finishBoot();
    if (!attached) stopDemo = demo();
  });
})();

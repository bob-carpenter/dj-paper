/*!
 * loop-widgets.js
 *
 * Interactive figures for the for-loop examples in "JAX à la Stan: The Sharp
 * Bits". Each figure steps through a Stan loop one scalar write at a time,
 * and through the JAX version one array operation at a time.
 *
 * Usage (any HTML page, including Quarto HTML output):
 *
 *   <div class="loop-widget" data-example="where"></div>
 *   <script src="loop-widgets.js"></script>
 *
 * In Quarto, put the code inside the div as ordinary code blocks. The widget
 * then shows those blocks, so its code looks like the paper's other code
 * blocks, and the PDF, which has no widget, prints them:
 *
 *   ::: {.loop-widget data-example="where"}
 *   ```stan
 *   for (n in 1:N) { ... }
 *   ```
 *   ```python
 *   mu = ...
 *   ```
 *   :::
 *
 * Their lines must be EX[example].stan.code and .jax.code, the lines the
 * steps highlight. If they differ, the console says so and the steps show the
 * widget's own copy. Without code blocks, the widget draws the code itself.
 *
 * Options, as data attributes on the div:
 *   data-example  where, cumsum, gather, vmap-ode, slicing, recurrence
 *                 or lagged-sum                             (required)
 *   data-mode     "code" (Stan and JAX code side by side), "stan" or "jax"
 *                                                           (default "code")
 *   data-step     starting step, a number or "end"          (default 0)
 *   data-static   hide the controls (for print and screenshots)
 *
 * Same design as stan-scatter-widget.js: no dependencies, styles injected
 * once and scoped to .lw, dark mode follows Quarto's light/dark toggle.
 * The static figures (loop-<example>-static.png) are rendered from
 * LoopWidgets.staticHTML by build_static.py, so they match this file.
 */
(function () {
  'use strict';

  var HOST = typeof window !== 'undefined' ? window : globalThis;
  var HAS_DOM = typeof document !== 'undefined';

  if (HOST.LoopWidgets) {
    if (HAS_DOM) HOST.LoopWidgets.initAll();
    return;
  }

  var CSS = [
    '.lw{--lw-text-2:#5f5e5a;--lw-text-3:#888780;--lw-border:rgba(0,0,0,.14);--lw-border-2:rgba(0,0,0,.3);',
    '--lw-surface:#f7f6f2;--lw-neutral:#ecebe4;--lw-acc-bg:#e6f1fb;--lw-acc-fg:#0c447c;--lw-acc-bd:#378add;',
    '--lw-z-bg:#eeedfe;--lw-z-fg:#3c3489;--lw-d-bg:#e1f5ee;--lw-d-fg:#085041;',
    '--lw-mono:var(--bs-font-monospace,ui-monospace,SFMono-Regular,Menlo,Consolas,"DejaVu Sans Mono",monospace);',
    'font-size:14px;line-height:1.5;text-align:left;margin:1rem 0}',
    '.lw.lw-dark{--lw-text-2:#b4b2a9;--lw-text-3:#8d8b84;--lw-border:rgba(255,255,255,.14);--lw-border-2:rgba(255,255,255,.3);',
    '--lw-surface:rgba(255,255,255,.04);--lw-neutral:rgba(255,255,255,.09);--lw-acc-bg:#0c447c;--lw-acc-fg:#b5d4f4;--lw-acc-bd:#85b7eb;',
    '--lw-z-bg:#3c3489;--lw-z-fg:#cecbf6;--lw-d-bg:#085041;--lw-d-fg:#9fe1cb}',
    '.lw *,.lw *::before,.lw *::after{box-sizing:border-box}',
    '.lw [hidden]{display:none!important}',
    // Side by side when both blocks fit, one above the other when they do not. Side by side, the columns are equally
    // wide when each block fits in half the width; otherwise the wider block keeps its width and the other takes the
    // rest. (A flex item's automatic minimum width is its longest code line, capped at the max-width of 100%.)
    '.lw-compare{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}',
    '.lw-compare>div{flex:1 1 0;min-width:auto;max-width:100%}',
    '.lw-ctitle{font-size:12px;font-weight:600;color:var(--lw-text-2);margin:0 0 6px}',
    // The page's own code blocks: Quarto styles them; keep the page's size, drop their outer margins.
    '.lw-qcode{font-size:var(--bs-body-font-size,1rem)}',
    '.lw-qcode .code-copy-outer-scaffold,.lw-qcode div.sourceCode{margin:0}',
    '.lw-qcode code.sourceCode>span{min-width:100%}',
    '.lw-qcode code.sourceCode>span.lw-on{background:var(--lw-acc-bg);box-shadow:inset 3px 0 0 var(--lw-acc-bd)}',
    '.lw-sr{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}',
    '.lw .lw-btn{font:inherit;font-size:13px;color:inherit;background:transparent;border:1px solid var(--lw-border-2);',
    'border-radius:8px;padding:6px 12px;margin:0;cursor:pointer;line-height:1.3;display:inline-flex;align-items:center;gap:6px}',
    '.lw .lw-btn:hover{background:var(--lw-neutral)}',
    '.lw .lw-btn:focus-visible{outline:2px solid var(--lw-acc-bd);outline-offset:2px}',
    '.lw .lw-btn[aria-pressed="true"]{background:var(--lw-neutral);border-color:currentColor}',
    '.lw-top{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin:0 0 1rem}',
    '.lw-seg{display:flex;flex-wrap:wrap;gap:6px}',
    '.lw-main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;align-items:start}',
    '.lw-block{margin:0 0 12px}',
    '.lw-btitle{font-size:12px;color:var(--lw-text-2);margin:0 0 6px}',
    '.lw-grid{display:grid;gap:4px;align-items:center;max-width:100%}',
    '.lw-rl{font-family:var(--lw-mono);font-size:12px;color:var(--lw-text-2);padding-right:8px;text-align:right;white-space:nowrap}',
    '.lw-ch{font-family:var(--lw-mono);font-size:11px;color:var(--lw-text-3);text-align:center}',
    '.lw-cell{aspect-ratio:4/3;min-width:0;border-radius:6px;display:flex;align-items:center;justify-content:center;text-align:center;',
    'padding:0 2px;overflow:hidden;white-space:pre;font-family:var(--lw-mono);font-size:11px;line-height:1.15;',
    'border:1px dashed var(--lw-border-2);color:var(--lw-text-3);transition:background-color .2s}',
    '.lw-cell.is-z{background:var(--lw-z-bg);color:var(--lw-z-fg);border:1px solid transparent}',
    '.lw-cell.is-d{background:var(--lw-d-bg);color:var(--lw-d-fg);border:1px solid transparent}',
    '.lw-cell.is-b{background:var(--lw-acc-bg);color:var(--lw-acc-fg);border:1px solid transparent}',
    '.lw-cell.is-data{background:var(--lw-neutral);color:var(--lw-text-2);border:1px solid transparent}',
    '.lw-cell.is-x{background:repeating-linear-gradient(135deg,var(--lw-neutral) 0 4px,transparent 4px 8px);',
    'color:var(--lw-text-3);border:1px solid var(--lw-border);text-decoration:line-through}',
    '.lw-cell.is-blank{border:none;background:none}',
    '.lw-cell.is-cur{outline:2px solid var(--lw-acc-bd);outline-offset:-2px}',
    '.lw-cell.is-hl{outline:2px dashed var(--lw-acc-bd);outline-offset:-2px}',
    '.lw-legend{display:flex;flex-wrap:wrap;gap:8px 14px;margin-top:6px;font-size:12px;color:var(--lw-text-2)}',
    '.lw-legend span{display:flex;align-items:center;gap:6px}',
    '.lw-sw{width:12px;height:12px;border-radius:3px;display:inline-block;flex:none}',
    '.lw-code{font-family:var(--lw-mono);font-size:12px;line-height:1.75;background:var(--lw-surface);',
    'border:1px solid var(--lw-border);border-radius:8px;padding:8px 4px;overflow-x:auto}',
    '.lw-line{padding:0 8px;border-radius:4px;white-space:pre}',
    '.lw-line.is-on{background:var(--lw-acc-bg);color:var(--lw-acc-fg)}',
    '.lw-state{font-size:13px;line-height:1.6;color:var(--lw-text-2);margin:10px 0 0;min-height:3.2em}',
    '.lw-steps{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:1rem 0 0}',
    '.lw-prog{font-size:13px;color:var(--lw-text-3);margin-left:4px}',
    '.lw[data-static] .lw-top,.lw[data-static] .lw-steps,.lw[data-static] .lw-state{display:none}',
    '@media (prefers-reduced-motion:reduce){.lw-cell{transition:none}}'
  ].join('\n');

  // Larger type for the static figures, which are scaled down in print.
  var STATIC_CSS = [
    'body{margin:0;background:#fff;width:1000px}',
    '.lw.lw-static{font-family:"Source Sans 3","Source Sans Pro",-apple-system,"Helvetica Neue",Arial,sans-serif;',
    'font-size:16px;margin:0;padding:24px 24px 16px;color:#1d1d1b}',
    '.lw-static h3{font-size:22px;font-weight:700;margin:0 0 14px}',
    '.lw-spanels{display:grid;gap:28px 40px}',
    '.lw-spanels.is-row{grid-template-columns:1fr 1fr}',
    '.lw-static .lw-grid{gap:6px}',
    '.lw-static .lw-cell{font-size:15px;border-radius:8px}',
    '.lw-static .lw-rl{font-size:14px}',
    '.lw-static .lw-ch{font-size:13px}',
    '.lw-static .lw-btitle{font-size:14px}',
    '.lw-static .lw-legend{font-size:15px;margin-top:22px;gap:10px 22px}',
    '.lw-static .lw-sw{width:16px;height:16px;border-radius:4px}'
  ].join('\n');

  var PLAY = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.5v9l7.5-4.5z" fill="currentColor"/></svg> Play';
  var PAUSE = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="2.5" y="1.5" width="2.5" height="9" fill="currentColor"/><rect x="7" y="1.5" width="2.5" height="9" fill="currentColor"/></svg> Pause';

  /* ---------- pure helpers ---------- */

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function key(b, r, c) { return b + ':' + r + ',' + c; }
  function range(n) { var a = []; for (var i = 0; i < n; i++) a.push(i); return a; }
  function num(v) {
    var s = String(Math.round(v * 10) / 10);
    return s.charAt(0) === '-' ? '−' + s.slice(1) : s;
  }
  function row(bid, r, cols) { return cols.map(function (c) { return key(bid, r, c); }); }

  // Records the state of every cell after each step.
  function Rec() { this.cells = {}; this.frames = []; }
  Rec.prototype.set = function (k, kind, label) { this.cells[k] = { k: kind, t: label }; };
  Rec.prototype.snap = function (o) {
    var c = {}, k;
    for (k in this.cells) c[k] = this.cells[k];
    this.frames.push({ cells: c, line: o.line || [], text: o.text || '', cur: o.cur || [], hl: o.hl || [] });
  };

  var SWATCH = {
    data: 'background:var(--lw-neutral)',
    z: 'background:var(--lw-z-bg)',
    d: 'background:var(--lw-d-bg)',
    b: 'background:var(--lw-acc-bg)',
    x: 'background:repeating-linear-gradient(135deg,var(--lw-neutral) 0 3px,transparent 3px 6px);border:1px solid var(--lw-border-2)',
    cur: 'outline:2px solid var(--lw-acc-bd);outline-offset:-2px',
    hl: 'outline:2px dashed var(--lw-acc-bd);outline-offset:-2px'
  };
  var OUTLINES = [['cur', 'written in this step'], ['hl', 'read in this step']];

  function legendHTML(items) {
    return items.map(function (it) {
      return '<span><i class="lw-sw" style="' + SWATCH[it[0]] + '"></i>' + esc(it[1]) + '</span>';
    }).join('');
  }

  function blocksHTML(blocks, f, cw) {
    return blocks.map(function (b) {
      var h = '<div class="lw-block">';
      if (b.title) h += '<div class="lw-btitle">' + esc(b.title) + '</div>';
      // labelw: a shared minimum width for the row labels, so that blocks with as many columns line up.
      var lw = b.labelw ? 'minmax(' + b.labelw + ',max-content)' : 'max-content';
      h += '<div class="lw-grid" style="grid-template-columns:' + lw + ' repeat(' + b.cols + ',minmax(0,' + cw + 'px))">';
      if (b.head) {
        h += '<span></span>';
        b.head.forEach(function (x) { h += '<span class="lw-ch">' + esc(x) + '</span>'; });
      }
      b.rows.forEach(function (rw, r) {
        h += '<span class="lw-rl">' + esc(rw.label) + '</span>';
        var at = rw.at || range(b.cols);
        for (var c = 0; c < b.cols; c++) {
          if (at.indexOf(c) < 0) { h += '<span class="lw-cell is-blank"></span>'; continue; }
          var k = key(b.id, r, c), x = f.cells[k], cls = 'lw-cell';
          if (x) cls += ' is-' + x.k;
          if (f.cur.indexOf(k) >= 0) cls += ' is-cur';
          else if (f.hl.indexOf(k) >= 0) cls += ' is-hl';
          h += '<span class="' + cls + '">' + (x ? esc(x.t) : '') + '</span>';
        }
      });
      return h + '</div></div>';
    }).join('');
  }

  function codeHTML(src, on) {
    return src.map(function (l, i) {
      return '<div class="lw-line' + (on.indexOf(i) >= 0 ? ' is-on' : '') + '">' + (l ? esc(l) : '&nbsp;') + '</div>';
    }).join('');
  }

  /* ---------- code blocks written in the page ---------- */

  // Quarto code blocks inside the widget div, by language: stan, and python for JAX.
  function takeBlocks(root) {
    var out = {};
    root.querySelectorAll('div.sourceCode').forEach(function (div) {
      var pre = div.querySelector('pre.sourceCode'), m = null;
      if (pre && pre.classList.contains('stan')) m = 'stan';
      else if (pre && pre.classList.contains('python')) m = 'jax';
      if (!m || out[m]) return;
      var p = div.parentNode;
      out[m] = p && p.classList.contains('code-copy-outer-scaffold') ? p : div;
    });
    return out;
  }

  // Pandoc writes one span per line as the children of <code>.
  function linesOf(block) {
    var code = block.querySelector('code');
    return code ? Array.prototype.slice.call(code.children) : [];
  }

  function sameCode(block, src) {
    var got = linesOf(block).map(function (s) { return s.textContent.replace(/\s+$/, ''); });
    return got.length === src.length && got.every(function (l, i) { return l === src[i].replace(/\s+$/, ''); });
  }

  // A copy for the step view, without the ids of the original.
  function stepCopy(block) {
    var c = block.cloneNode(true);
    c.removeAttribute('id');
    c.querySelectorAll('[id]').forEach(function (e) { e.removeAttribute('id'); });
    return { node: c, lines: linesOf(c) };
  }

  /* ---------- the examples ---------- */

  var EX = {};

  // 1. Branch inside the loop: jnp.where.
  EX.where = (function () {
    var x = [-1, 0.4, 1.5, -0.3, 2.2, 0.9], a = 1, b = 2, c = 0.5, d = 1, N = x.length;
    var base = x.map(function (v) { return a + b * v; });
    var bump = x.map(function (v) { return d * (v - c); });
    var take = x.map(function (v) { return v > c; });
    var mu = base.map(function (v, n) { return v + (take[n] ? bump[n] : 0); });
    var skipped = take.filter(function (t) { return !t; }).length;
    var pars = 'With a = ' + a + ', b = ' + b + ', c = ' + c + ', d = ' + d + '.';
    function data(R, bid) { x.forEach(function (v, n) { R.set(key(bid, 0, n), 'data', num(v)); }); }
    return {
      aria: 'Branch inside a loop: Stan adds the branch value element by element where x is above c; JAX computes both branches for every element and selects with jnp.where.',
      jaxLabel: 'JAX jnp.where',
      legend: [['data', 'x, data'], ['z', 'computed'], ['d', 'branch taken'], ['x', 'computed, then discarded']],
      still: { step: 6, layout: 'col', a: '(a) Stan loop, after 3 of 6 iterations', b: '(b) JAX: both branches for every element, then one select' },
      stan: {
        code: ['for (n in 1:N) {', '  mu[n] = a + b * x[n];', '  if (x[n] > c)', '    mu[n] += d * (x[n] - c);', '}'],
        blocks: [{ id: 's', cols: N, rows: [{ label: 'x' }, { label: 'mu' }] }],
        frames: function () {
          var R = new Rec();
          data(R, 's');
          R.snap({ text: pars + ' Stan visits one n at a time, and runs the branch only where x[n] > c.' });
          for (var n = 0; n < N; n++) {
            R.set(key('s', 1, n), 'z', num(base[n]));
            R.snap({ line: [1], cur: [key('s', 1, n)], hl: [key('s', 0, n)],
              text: 'n = ' + (n + 1) + ': mu[n] = a + b·x[n] = ' + num(base[n]) + '.' });
            if (take[n]) {
              R.set(key('s', 1, n), 'd', num(mu[n]));
              R.snap({ line: [2, 3], cur: [key('s', 1, n)], hl: [key('s', 0, n)],
                text: 'x[n] = ' + num(x[n]) + ' > c, so the branch adds d·(x[n] − c) = ' + num(bump[n]) + ': mu[n] = ' + num(mu[n]) + '.' });
            } else {
              R.snap({ line: [2], hl: [key('s', 0, n)], text: 'x[n] = ' + num(x[n]) + ' ≤ c, so the branch is skipped.' });
            }
          }
          return R.frames;
        }
      },
      jax: {
        code: ['base = a + b * x', 'bump = d * (x - c)          # every element', 'take = x > c', 'mu = base + jnp.where(take, bump, 0.0)'],
        blocks: [{ id: 'j', cols: N, rows: [{ label: 'x' }, { label: 'a + b·x' }, { label: 'd·(x − c)' }, { label: 'x > c' }, { label: 'mu' }] }],
        frames: function () {
          var R = new Rec(), all = range(N);
          data(R, 'j');
          R.snap({ text: pars + ' Each line of the JAX version is one operation on all elements at once.' });
          base.forEach(function (v, n) { R.set(key('j', 1, n), 'z', num(v)); });
          R.snap({ line: [0], cur: row('j', 1, all), text: 'One operation computes a + b·x for every element.' });
          bump.forEach(function (v, n) { R.set(key('j', 2, n), 'z', num(v)); });
          R.snap({ line: [1], cur: row('j', 2, all), text: 'The branch value d·(x − c) is computed for every element too, also where x ≤ c.' });
          take.forEach(function (t, n) { R.set(key('j', 3, n), t ? 'd' : 'data', t ? 'true' : 'false'); });
          R.snap({ line: [2], cur: row('j', 3, all), text: 'The condition becomes a boolean mask.' });
          mu.forEach(function (v, n) {
            R.set(key('j', 4, n), take[n] ? 'd' : 'z', num(v));
            if (!take[n]) R.set(key('j', 2, n), 'x', num(bump[n]));
          });
          R.snap({ line: [3], cur: row('j', 4, all),
            text: 'jnp.where keeps bump where the mask is true and 0 elsewhere. The ' + skipped +
                  ' discarded bump values were still computed, so a NaN among them would still reach the gradient.' });
          return R.frames;
        }
      }
    };
  })();

  // 2. Running sum: a candidate's weekly support as a random walk, with jnp.cumsum.
  EX.cumsum = (function () {
    // Weeks t = 1..T in columns 1..T; column 0 of the support row holds support0, the support before week 1.
    // Labels i‥j stand for Δi + … + Δj, where Δt = delta[t] and entry 0 is support0.
    var T = 8, B = 4, weeks = range(T).map(function (t) { return t + 1; });
    function lab(i, j) { return i + '‥' + j; }
    function given(R, bid) {
      weeks.forEach(function (t) { R.set(key(bid, 0, t), 'data', 'Δ' + t); });
      R.set(key(bid, 1, 0), 'data', '0');
    }
    var blocks = function (bid) {
      return [{ id: bid, cols: T + 1, labelw: '6em', rows: [{ label: 'delta', at: weeks }, { label: 'support' }] }];
    };
    return {
      aria: 'Running sum: Stan adds one week\'s change at a time, each addition waiting for the previous one; jnp.cumsum sums within blocks in parallel, then adds the totals of earlier blocks, and support0 is added to every week at once.',
      jaxLabel: 'JAX cumsum',
      legend: [['data', 'given: the changes Δt = delta[t], and support0 (0)'], ['z', 'i‥j: Δi + … + Δj'],
               ['d', 'total of the earlier block added'], ['b', '0‥j: plus support0, the support in week j']],
      still: { step: 4, layout: 'col', a: '(a) Stan loop, after 4 of 8 weeks', b: '(b) JAX: sums within blocks, then the earlier block totals' },
      stan: {
        code: ['support[1] = support0 + delta[1];', 'for (t in 2:T)', '  support[t] = support[t - 1] + delta[t];'],
        blocks: blocks('s'),
        frames: function () {
          var R = new Rec();
          given(R, 's');
          R.snap({ text: 'support0 is the support before the first week, and delta[t] the change in week t.' });
          weeks.forEach(function (t) {
            R.set(key('s', 1, t), 'b', lab(0, t));
            R.snap({ line: t === 1 ? [0] : [1, 2], cur: [key('s', 1, t)], hl: [key('s', 1, t - 1), key('s', 0, t)],
              text: t === 1 ? 'support[1] = support0 + delta[1].'
                            : 't = ' + t + ': adds delta[' + t + '] to support[' + (t - 1) + '], so it has to wait for week ' + (t - 1) + '.' });
          });
          return R.frames;
        }
      },
      jax: {
        code: ['support = support0 + jnp.cumsum(delta)'],
        blocks: blocks('j'),
        frames: function () {
          var R = new Rec(), cur = [];
          given(R, 'j');
          R.snap({ text: 'One call to cumsum. XLA splits delta into blocks, here of 4 weeks (16 on CPU).' });
          weeks.forEach(function (t) { R.set(key('j', 1, t), 'z', lab(t - (t - 1) % B, t)); cur.push(key('j', 1, t)); });
          R.snap({ line: [0], cur: cur, text: 'Every partial sum within a block is computed at once, in all blocks in parallel.' });
          cur = [];
          weeks.slice(B).forEach(function (t) { R.set(key('j', 1, t), 'd', lab(1, t)); cur.push(key('j', 1, t)); });
          R.snap({ line: [0], cur: cur, hl: [key('j', 1, B)],
            text: 'Adding the total of the earlier block (' + lab(1, B) + ') to the next block finishes the running sum: 2 levels instead of ' +
                  T + ' dependent steps. Each 16-fold increase in T adds one level.' });
          cur = [];
          weeks.forEach(function (t) { R.set(key('j', 1, t), 'b', lab(0, t)); cur.push(key('j', 1, t)); });
          R.snap({ line: [0], cur: cur, hl: [key('j', 1, 0)], text: 'Adding support0 gives the support in every week at once.' });
          return R.frames;
        }
      }
    };
  })();

  // 3. Group lookup: indexing.
  EX.gather = (function () {
    var G = 3, P = 2, group = [0, 2, 1, 0, 2, 2], N = group.length, KIND = ['z', 'd', 'b'];
    function blocks(bid) {
      return [
        { id: bid + 'b', side: true, title: 'beta: one row of coefficients per group', cols: P, head: ['p 0', 'p 1'],
          rows: range(G).map(function (g) { return { label: 'g ' + g }; }) },
        { id: bid + 'r', title: 'coefficients used by each observation, and mu', cols: P + 1, head: ['p 0', 'p 1', 'mu'],
          rows: group.map(function (g, n) { return { label: 'n ' + n + ' (g ' + g + ')' }; }) }
      ];
    }
    function beta(R, bid) {
      for (var g = 0; g < G; g++) for (var p = 0; p < P; p++) R.set(key(bid + 'b', g, p), KIND[g], 'β' + g + ',' + p);
    }
    function useRow(R, bid, n) {
      for (var p = 0; p < P; p++) R.set(key(bid + 'r', n, p), KIND[group[n]], 'β' + group[n] + ',' + p);
    }
    return {
      aria: 'Group lookup: Stan reads the coefficients of each observation\'s group inside the loop; beta[group] gathers them for all observations in one operation.',
      jaxLabel: 'JAX beta[group]',
      legend: [['z', 'group 0'], ['d', 'group 1'], ['b', 'group 2']],
      still: { step: 3, layout: 'row', a: '(a) Stan loop, after 3 of 6 observations', b: '(b) JAX: one gather for all observations' },
      stan: {
        code: ['for (n in 1:N)', '  y[n] ~ normal(X[n] * beta[group[n]],', '                sigma);'],
        blocks: blocks('s'),
        frames: function () {
          var R = new Rec();
          beta(R, 's');
          R.snap({ text: 'Groups have different sizes (2, 1 and 3 observations). Stan looks up one group row per loop iteration.' });
          for (var n = 0; n < N; n++) {
            useRow(R, 's', n);
            R.set(key('sr', n, P), KIND[group[n]], 'X·β');
            R.snap({ line: [0, 1], cur: row('sr', n, range(P + 1)), hl: row('sb', group[n], range(P)),
              text: 'Observation ' + (n + 1) + ' of ' + N + ': reads the row of its group and computes mu[n].' });
          }
          return R.frames;
        }
      },
      jax: {
        code: ['# group: (N,) 0-based; X: (N, P); beta: (G, P)', 'mu = jnp.sum(X * beta[group], axis=-1)'],
        blocks: blocks('j'),
        frames: function () {
          var R = new Rec(), cur = [], n;
          beta(R, 'j');
          R.snap({ text: 'group is an index array, here [' + group.join(', ') + '].' });
          for (n = 0; n < N; n++) { useRow(R, 'j', n); cur = cur.concat(row('jr', n, range(P))); }
          R.snap({ line: [1], cur: cur, text: 'beta[group] gathers the row of every observation\'s group in one operation, for groups of any size.' });
          cur = [];
          for (n = 0; n < N; n++) { R.set(key('jr', n, P), KIND[group[n]], 'X·β'); cur.push(key('jr', n, P)); }
          R.snap({ line: [1], cur: cur,
            text: 'Multiplying by X and summing each row gives all of mu at once. In the gradient, the gather becomes a scatter-add into beta.' });
          return R.frames;
        }
      }
    };
  })();

  // 4. One ODE solve per subject: jax.vmap.
  EX['vmap-ode'] = (function () {
    var steps = [3, 5, 2, 4], J = steps.length, K = Math.max.apply(null, steps);
    var total = steps.reduce(function (s, v) { return s + v; }, 0);
    var labels = range(J).map(function (j) { return { label: 'subject ' + j }; });
    return {
      aria: 'One ODE solve per subject: Stan solves the subjects one after another; under jax.vmap one solver loop advances every subject at each iteration, and finished subjects are masked.',
      jaxLabel: 'JAX vmap',
      legend: [['z', 'adaptive solver step'], ['x', 'finished subject, masked']],
      still: { step: steps[0] + steps[1], layout: 'row', a: '(a) Stan loop, after 2 of 4 subjects', b: '(b) JAX vmap: one loop for all subjects' },
      stan: {
        code: ['for (j in 1:J) {', '  array[T] vector[2] x =', '    ode_rk45(pk, x0, 0, ts, theta[j]);', '  y[j] ~ normal(to_vector(x[:, 2]), sigma);', '}'],
        blocks: [{ id: 's', title: 'solver steps of each subject', cols: K, head: range(K).map(function (k) { return String(k + 1); }), rows: labels }],
        frames: function () {
          var R = new Rec();
          R.snap({ text: 'Each subject needs a different number of adaptive steps (here ' + steps.join(', ') + '). Stan solves one subject after another.' });
          for (var j = 0; j < J; j++) {
            for (var k = 0; k < steps[j]; k++) {
              R.set(key('s', j, k), 'z', 'step ' + (k + 1));
              R.snap({ line: [1, 2], cur: [key('s', j, k)], text: 'Subject ' + (j + 1) + ', step ' + (k + 1) + ' of ' + steps[j] + '.' });
            }
          }
          return R.frames;
        }
      },
      jax: {
        code: ['def subject_lp(y_j, theta_j):', '    x = odeint(pk, x0, ts, *theta_j)   # ts from 0', '    return st.norm.logpdf(y_j, x[1:, 1], sigma).sum()', '', 'lp = jax.vmap(subject_lp)(y, theta).sum()'],
        blocks: [{ id: 'j', title: 'iterations of the batched solver loop', cols: K, head: range(K).map(function (k) { return String(k + 1); }), rows: labels }],
        frames: function () {
          var R = new Rec();
          R.snap({ text: 'Under vmap, one solver loop runs for all subjects, and each subject keeps its own step size.' });
          for (var k = 0; k < K; k++) {
            var cur = [], idle = 0;
            for (var j = 0; j < J; j++) {
              if (k < steps[j]) { R.set(key('j', j, k), 'z', 'step ' + (k + 1)); cur.push(key('j', j, k)); }
              else { R.set(key('j', j, k), 'x', 'idle'); idle++; }
            }
            var text = 'Iteration ' + (k + 1) + ': ' + cur.length + ' subject' + (cur.length > 1 ? 's take' : ' takes') + ' a step at once' +
                       (idle ? ', ' + idle + ' finished subject' + (idle > 1 ? 's are' : ' is') + ' masked.' : '.');
            if (k === K - 1) text += ' The loop ends when the last subject reaches the final time: ' + K + ' iterations instead of ' + total + ' steps one after another.';
            R.snap({ line: [1, 4], cur: cur, text: text });
          }
          return R.frames;
        }
      }
    };
  })();

  // 5. Recurrence on observed data: slicing.
  EX.slicing = (function () {
    var T = 7, pairs = range(T - 1);
    function data(R, bid) { range(T).forEach(function (t) { R.set(key(bid, 0, t), 'data', 'y' + t); }); }
    function term(t) { return 'y' + (t - 1) + '→y' + t; }
    return {
      aria: 'Recurrence on observed data: Stan evaluates one term at a time; JAX lines up the shifted views y[1:] and y[:-1] and evaluates every term in one call.',
      jaxLabel: 'JAX slicing',
      legend: [['data', 'y, data'], ['b', 'view y[1:]'], ['d', 'view y[:-1]'], ['z', 'log density term']],
      still: { step: 3, layout: 'col', a: '(a) Stan loop, after 3 of 6 terms', b: '(b) JAX: two shifted views, one call' },
      stan: {
        code: ['for (t in 2:T)', '  y[t] ~ normal(alpha + beta * y[t - 1],', '                sigma);'],
        blocks: [{ id: 's', cols: T, rows: [{ label: 'y' }, { label: 'term', at: range(T).slice(1) }] }],
        frames: function () {
          var R = new Rec();
          data(R, 's');
          R.snap({ text: 'Every term pairs y[t] with y[t − 1]. Both are observed, so nothing is carried from one step to the next.' });
          for (var t = 1; t < T; t++) {
            R.set(key('s', 1, t), 'z', term(t));
            R.snap({ line: [0, 1], cur: [key('s', 1, t)], hl: [key('s', 0, t - 1), key('s', 0, t)],
              text: 'Term ' + t + ' of ' + (T - 1) + ': y[t] given y[t − 1].' });
          }
          return R.frames;
        }
      },
      jax: {
        code: ['lp = st.norm.logpdf(', '    y[1:],                    # y1 … y6', '    alpha + beta * y[:-1],    # y0 … y5', '    sigma).sum()'],
        blocks: [{ id: 'j', cols: T, rows: [{ label: 'y' }, { label: 'y[1:]', at: pairs }, { label: 'y[:-1]', at: pairs }, { label: 'term', at: pairs }] }],
        frames: function () {
          var R = new Rec();
          data(R, 'j');
          R.snap({ text: 'Two shifted views of y line up every pair.' });
          pairs.forEach(function (i) { R.set(key('j', 1, i), 'b', 'y' + (i + 1)); });
          R.snap({ line: [1], cur: row('j', 1, pairs), hl: row('j', 0, range(T).slice(1)), text: 'y[1:] is y without its first value.' });
          pairs.forEach(function (i) { R.set(key('j', 2, i), 'd', 'y' + i); });
          R.snap({ line: [2], cur: row('j', 2, pairs), hl: row('j', 0, pairs), text: 'y[:-1] is y without its last value. Each column now holds one pair.' });
          pairs.forEach(function (i) { R.set(key('j', 3, i), 'z', term(i + 1)); });
          R.snap({ line: [0, 1, 2, 3], cur: row('j', 3, pairs),
            text: 'One element-wise call evaluates all ' + (T - 1) + ' terms. The slices are fused into it, so no shifted copy of y is written.' });
          return R.frames;
        }
      }
    };
  })();

  // 6. Recurrence on computed values: recurrence (the renewal equation, p = S = 3, one input per step).
  // One frame per step in both views, played at the same speed: lax.scan runs the steps in order, as Stan does.
  EX.recurrence = (function () {
    var S = 3, T = 8, n = T - S, days = range(T).slice(S);
    function lab(t) { return 'i' + t; }
    function lags(t) { return range(S).map(function (s) { return t - 1 - s; }); }   // nearest first
    function given(R, bid) {
      range(S).forEach(function (t) { R.set(key(bid, 0, t), 'data', lab(t)); });
      days.forEach(function (t) { R.set(key(bid, 1, t), 'data', 'R' + t); });
    }
    function series(bid, title) {
      return { id: bid, title: title, cols: T, labelw: '6em', rows: [{ label: 'infections' }, { label: 'R', at: days }] };
    }
    function weighted(t) {
      return lags(t).map(function (u, s) { return 'g' + (s + 1) + '·' + lab(u); }).join(' + ');
    }
    return {
      aria: 'Renewal equation: Stan computes each day\'s infections from the S days before it; recurrence passes f the last S days, ' +
            'nearest first, with that day\'s R_t, and pushes each new day onto the front of that window. Both run the steps in order.',
      jaxLabel: 'JAX recurrence',
      legend: [['data', 'seed and R, given'], ['b', 'window f sees (lags)'], ['z', 'new day'], ['x', 'dropped from the window']],
      still: { step: 2, layout: 'col', a: '(a) Stan loop, after 2 of ' + n + ' steps', b: '(b) recurrence: f sees the last S days, nearest first' },
      stan: {
        ms: 900,
        code: ['infections[1:S] = seed;', 'for (t in (S + 1):T)', '  infections[t] = R[t] * dot_product(', '      g, reverse(infections[(t - S):(t - 1)]));'],
        blocks: [series('s', 'infections: S = ' + S + ' seed days, then the loop')],
        frames: function () {
          var R = new Rec();
          given(R, 's');
          R.snap({ line: [0], cur: row('s', 0, range(S)),
            text: 'The first S = ' + S + ' days are the seed. Each new day needs the S days before it, which the loop has just computed.' });
          days.forEach(function (t, k) {
            R.set(key('s', 0, t), 'z', lab(t));
            R.snap({ line: [1, 2, 3], cur: [key('s', 0, t)],
              hl: lags(t).map(function (u) { return key('s', 0, u); }).concat([key('s', 1, t)]),
              text: 'Step ' + (k + 1) + ' of ' + n + ': ' + lab(t) + ' = R' + t + '·(' + weighted(t) + ').' +
                    (k === 0 ? ' reverse puts the nearest day first, to line up with g[1].' : '') });
          });
          return R.frames;
        }
      },
      jax: {
        ms: 900,
        code: ['infections = recurrence(',
               '    lambda lags, R_t: R_t * (g @ lags),  # lags[s-1] = infections[t-s]',
               '    seed,                                # init: S days, in time order',
               '    R[S:])                               # inputs: one R_t per new day'],
        blocks: [
          // As many columns as the result, so the carry's cells are as large as the days'.
          { id: 'w', title: 'carry: the window f sees, nearest first', cols: T, labelw: '6em',
            head: range(T).map(function (c) { return c < S ? '[' + c + ']' : c === S ? 'dropped' : ''; }),
            rows: [{ label: 'lags', at: range(S + 1) }] },
          series('j', 'result: seed, then the new days')
        ],
        frames: function () {
          var R = new Rec();
          function win(t) { lags(t).forEach(function (u, s) { R.set(key('w', 0, s), 'b', lab(u)); }); }
          // scan writes only the new days; recurrence puts seed in front of them after the loop.
          days.forEach(function (t) { R.set(key('j', 1, t), 'data', 'R' + t); });
          win(S);
          R.snap({ line: [2], cur: row('w', 0, range(S)),
            text: 'seed holds the S = ' + S + ' known days in time order. The carry starts with them nearest first: ' +
                  'lags = [' + lags(S).map(lab).join(', ') + ']. The loop writes only the new days.' });
          days.forEach(function (t, k) {
            win(t);
            if (k > 0) R.set(key('w', 0, S), 'x', lab(t - 1 - S));
            R.set(key('j', 0, t), 'z', lab(t));
            var text = 'Step ' + (k + 1) + ' of ' + n + ': ';
            if (k === 0) text += 'f(lags, R' + t + ') = R' + t + '·(g @ lags) gives ' + lab(t) + '. Each step is one iteration of jax.lax.scan, ' +
                                 'compiled once and run in order, as in Stan.';
            else text += lab(t - 1) + ' was pushed onto the front of the window and ' + lab(t - 1 - S) + ' dropped off; f gives ' + lab(t) + '.';
            R.snap({ line: [1, 3], cur: [key('j', 0, t)],
              hl: row('w', 0, range(S)).concat([key('j', 1, t)]),   // f reads the lags from the carry, and R_t from the inputs
              text: text });
          });
          range(S).forEach(function (t) { R.set(key('j', 0, t), 'data', lab(t)); });
          R.snap({ line: [0, 2], cur: row('j', 0, range(S)),
            text: 'After the loop, recurrence puts seed in front of the new days: ' + S + ' + ' + n + ' = ' + T + ' days, like Stan\'s infections.' });
          return R.frames;
        }
      }
    };
  })();

  // 7. Delay from infections to deaths, and ragged weekly sums: lagged_sum, segment_sum.
  EX['lagged-sum'] = (function () {
    var T = 8, D = 3, week = [0, 0, 0, 1, 1, 1, 1, 1], W = 2, WK = ['d', 'b'];
    var first = range(W).map(function (w) { return week.indexOf(w); });
    function dl(upto) { return upto === 0 ? 'π0' : upto === 1 ? 'π0,1' : 'π0‥' + upto; }
    function blocks(bid) {
      return [{ id: bid, cols: T, rows: [{ label: 'infections' }, { label: 'deaths' }, { label: 'weekly', at: range(W) }] }];
    }
    function data(R, bid) { range(T).forEach(function (t) { R.set(key(bid, 0, t), 'data', 'i' + t); }); }
    var terms = range(T).reduce(function (s, t) { return s + Math.min(t + 1, D); }, 0);
    return {
      aria: 'Delay from infections to deaths and weekly sums: Stan adds one delayed term at a time and then each day into its week; JAX uses one lagged_sum and one segment_sum.',
      jaxLabel: 'JAX lagged_sum, segment_sum',
      legend: [['data', 'infections, input'], ['z', 'π0‥d: delays 0 to d added'], ['d', 'week 0 (3 days)'], ['b', 'week 1 (5 days)']],
      still: { step: 12, layout: 'col', a: '(a) Stan loops, after 12 of ' + (terms + T) + ' updates', b: '(b) JAX: one lagged_sum, one segment_sum' },
      stan: {
        code: ['for (t in 1:T)', '  for (d in 1:min(t, D))', '    deaths[t] += pi[d] * infections[t - d + 1];', 'for (t in 1:T)', '  weekly[week[t]] += deaths[t];'],
        blocks: blocks('s'),
        frames: function () {
          var R = new Rec(), t, d;
          data(R, 's');
          range(T).forEach(function (t2) { R.set(key('s', 1, t2), 'data', '0'); });
          range(W).forEach(function (w) { R.set(key('s', 2, w), 'data', '0'); });
          R.snap({ text: 'Stan adds one delayed term at a time, then each day into its week: ' + terms + ' + ' + T + ' = ' + (terms + T) + ' scalar updates.' });
          for (t = 0; t < T; t++) {
            for (d = 0; d < Math.min(t + 1, D); d++) {
              R.set(key('s', 1, t), 'z', dl(d));
              R.snap({ line: [1, 2], cur: [key('s', 1, t)], hl: [key('s', 0, t - d)],
                text: 'Day ' + (t + 1) + ', delay of ' + d + ' day' + (d === 1 ? '' : 's') + ': one multiply-add.' });
            }
          }
          for (t = 0; t < T; t++) {
            var w = week[t];
            R.set(key('s', 1, t), WK[w], dl(Math.min(t, D - 1)));
            R.set(key('s', 2, w), WK[w], 'days\n' + first[w] + '‥' + t);
            R.snap({ line: [3, 4], cur: [key('s', 2, w)], hl: [key('s', 1, t)], text: 'Day ' + (t + 1) + ' is added into its week.' });
          }
          return R.frames;
        }
      },
      jax: {
        code: ['deaths = lagged_sum(pi, infections)', 'weekly = jax.ops.segment_sum(', '    deaths, week, num_segments=W)'],
        blocks: blocks('j'),
        frames: function () {
          var R = new Rec(), t;
          data(R, 'j');
          R.snap({ text: 'Two operations: one lagged_sum and one segment_sum.' });
          for (t = 0; t < T; t++) R.set(key('j', 1, t), 'z', dl(Math.min(t, D - 1)));
          R.snap({ line: [0], cur: row('j', 1, range(T)),
            text: 'lagged_sum computes every deaths[t] = pi[0]·infections[t] + pi[1]·infections[t − 1] + … at once: each reads only infections, ' +
                  'so no output waits for another. Infections before the start count as 0, so the first D − 1 = ' + (D - 1) + ' values are partial sums.' });
          for (t = 0; t < T; t++) R.set(key('j', 1, t), WK[week[t]], dl(Math.min(t, D - 1)));
          range(W).forEach(function (w) {
            R.set(key('j', 2, w), WK[w], 'days\n' + first[w] + '‥' + week.lastIndexOf(w));
          });
          R.snap({ line: [1, 2], cur: row('j', 2, range(W)),
            text: 'segment_sum adds each day into its week with one scatter-add. The weeks have 3 and 5 days, and nothing is padded.' });
          return R.frames;
        }
      }
    };
  })();

  var NAMES = Object.keys(EX);

  function frames(ex, mode) {
    var c = '_' + mode;
    if (!ex[c]) ex[c] = ex[mode].frames();
    return ex[c];
  }

  /* ---------- static figure (used by build_static.py) ---------- */

  function staticHTML(name) {
    var ex = EX[name], st = ex.still;
    var fs = frames(ex, 'stan'), fj = frames(ex, 'jax');
    var a = fs[st.step], b = fj[fj.length - 1];
    var clean = { cells: b.cells, line: [], text: '', cur: [], hl: [] };
    var outl = OUTLINES.filter(function (o) { return a[o[0]].length > 0; });
    var cw = st.layout === 'row' ? 72 : 100;
    return '<!doctype html><html><head><meta charset="utf-8"><style>' + CSS + '\n' + STATIC_CSS + '</style></head><body>' +
      '<div class="lw lw-static"><div class="lw-spanels' + (st.layout === 'row' ? ' is-row' : '') + '">' +
      '<div><h3>' + esc(st.a) + '</h3>' + blocksHTML(ex.stan.blocks, a, cw) + '</div>' +
      '<div><h3>' + esc(st.b) + '</h3>' + blocksHTML(ex.jax.blocks, clean, cw) + '</div>' +
      '</div><div class="lw-legend">' + legendHTML(ex.legend.concat(outl)) + '</div></div></body></html>';
  }

  /* ---------- theme ---------- */

  function pageIsDark() {
    var b = document.body;
    if (!b) return false;
    if (b.classList.contains('quarto-dark')) return true;
    if (b.classList.contains('quarto-light')) return false;
    var m = (getComputedStyle(b).backgroundColor || '').match(/[\d.]+/g);
    if (!m || m.length < 3 || (m.length > 3 && +m[3] === 0)) return false;
    return (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) / 255 < 0.5;
  }

  var instances = [];
  function syncTheme() {
    var dark = pageIsDark();
    instances.forEach(function (root) { root.classList.toggle('lw-dark', dark); });
  }

  function injectCSS() {
    if (document.getElementById('lw-css')) return;
    var s = document.createElement('style');
    s.id = 'lw-css';
    s.textContent = CSS;
    (document.head || document.documentElement).appendChild(s);
  }

  /* ---------- one widget ---------- */

  function template(ex) {
    return '<p class="lw-sr">' + esc(ex.aria) + '</p>' +
      '<div class="lw-top"><div class="lw-seg" role="group" aria-label="View">' +
        '<button type="button" class="lw-btn" data-mode="code">Compare code</button>' +
        '<button type="button" class="lw-btn" data-mode="stan">Stan loop</button>' +
        '<button type="button" class="lw-btn" data-mode="jax">' + esc(ex.jaxLabel) + '</button>' +
      '</div></div>' +
      '<div class="lw-compare" hidden></div>' +
      '<div class="lw-main">' +
        '<div><div class="lw-blocks" role="img" aria-label="' + esc(ex.aria) + '"></div><div class="lw-legend"></div></div>' +
        '<div><div class="lw-side"></div><div class="lw-codebox"></div><p class="lw-state" aria-live="polite"></p></div>' +
      '</div>' +
      '<div class="lw-steps">' +
        '<button type="button" class="lw-btn" data-act="back" aria-label="Back one step">‹ Back</button>' +
        '<button type="button" class="lw-btn" data-act="play">' + PLAY + '</button>' +
        '<button type="button" class="lw-btn" data-act="next" aria-label="Forward one step">Next ›</button>' +
        '<button type="button" class="lw-btn" data-act="reset">Reset</button>' +
        '<span class="lw-prog"></span>' +
      '</div>';
  }

  function create(root) {
    if (root.__lw) return root.__lw;
    var d = root.dataset, ex = EX[d.example];
    if (!ex) { root.textContent = 'loop-widgets.js: unknown data-example "' + d.example + '"'; return null; }
    injectCSS();

    var mode = d.mode === 'stan' || d.mode === 'jax' ? d.mode : 'code';
    var fr = frames(ex, mode === 'code' ? 'stan' : mode);
    var step = d.step === 'end' ? fr.length - 1 : Math.min(fr.length - 1, Math.max(0, parseInt(d.step, 10) || 0));
    var timer = null;

    var page = takeBlocks(root), copies = {};
    root.classList.add('lw');
    root.innerHTML = template(ex);
    var q = function (sel) { return root.querySelector(sel); };
    var el = { main: q('.lw-main'), steps: q('.lw-steps'), compare: q('.lw-compare'),
               blocks: q('.lw-blocks'), side: q('.lw-side'), codebox: q('.lw-codebox'), state: q('.lw-state'), prog: q('.lw-prog'), play: q('[data-act="play"]') };
    q('.lw-legend').innerHTML = legendHTML(ex.legend.concat(OUTLINES));

    // The comparison shows the page's code blocks themselves; the steps use copies of them.
    ['stan', 'jax'].forEach(function (m) {
      var col = document.createElement('div'), b = page[m];
      col.innerHTML = '<p class="lw-ctitle">' + (m === 'stan' ? 'Stan' : 'JAX') + '</p>';
      if (b) {
        var box = document.createElement('div');
        box.className = 'lw-qcode';
        box.appendChild(b);
        col.appendChild(box);
        if (sameCode(b, ex[m].code)) copies[m] = stepCopy(b);
        else if (HOST.console) console.warn('loop-widgets.js (' + d.example + '): the ' + m + ' code block in the page differs from ' +
                                            'the lines the steps highlight, so the steps show their own copy of the code.');
      } else {
        col.insertAdjacentHTML('beforeend', '<div class="lw-code">' + codeHTML(ex[m].code, []) + '</div>');
      }
      el.compare.appendChild(col);
    });

    function showCode(m, on) {
      var c = copies[m];
      el.codebox.classList.toggle('lw-qcode', !!c);
      if (!c) { el.codebox.innerHTML = '<div class="lw-code">' + codeHTML(ex[m].code, on) + '</div>'; return; }
      if (el.codebox.firstChild !== c.node) { el.codebox.textContent = ''; el.codebox.appendChild(c.node); }
      c.lines.forEach(function (ln, i) { ln.classList.toggle('lw-on', on.indexOf(i) >= 0); });
    }

    function render() {
      var isCode = mode === 'code';
      el.main.hidden = isCode;
      el.steps.hidden = isCode;
      el.compare.hidden = !isCode;
      root.querySelectorAll('button[data-mode]').forEach(function (b) {
        b.setAttribute('aria-pressed', String(b.getAttribute('data-mode') === mode));
      });
      if (isCode) return;
      fr = frames(ex, mode);
      step = Math.min(step, fr.length - 1);
      var f = fr[step], m = ex[mode];
      el.blocks.innerHTML = blocksHTML(m.blocks.filter(function (b) { return !b.side; }), f, 64);
      el.side.innerHTML = blocksHTML(m.blocks.filter(function (b) { return b.side; }), f, 64);
      showCode(mode, f.line);
      el.state.textContent = f.text;
      el.prog.textContent = 'step ' + step + ' / ' + (fr.length - 1);
    }
    function stop() {
      if (timer) { clearInterval(timer); timer = null; }
      el.play.innerHTML = PLAY;
      el.state.setAttribute('aria-live', 'polite');
    }
    function go(dir) {
      step = Math.max(0, Math.min(fr.length - 1, step + dir));
      render();
      if (step === fr.length - 1 && dir > 0) stop();
    }
    function togglePlay() {
      if (timer) { stop(); return; }
      if (step >= fr.length - 1) step = 0;
      el.play.innerHTML = PAUSE;
      el.state.setAttribute('aria-live', 'off');
      render();
      timer = setInterval(function () {
        if (step >= fr.length - 1) { stop(); return; }
        go(1);
      }, ex[mode].ms || (mode === 'stan' ? 520 : 1300));   // per step; an example sets ms when both views take as many steps
    }

    root.addEventListener('click', function (ev) {
      var b = ev.target.closest('button');
      if (!b || !root.contains(b)) return;
      var m = b.getAttribute('data-mode'), act = b.getAttribute('data-act');
      if (m) { stop(); mode = m; step = 0; render(); }
      else if (act === 'back') go(-1);
      else if (act === 'next') go(1);
      else if (act === 'play') togglePlay();
      else if (act === 'reset') { stop(); step = 0; render(); }
    });

    render();
    instances.push(root);
    root.classList.toggle('lw-dark', pageIsDark());
    root.__lw = { render: render, stop: stop };
    return root.__lw;
  }

  function initAll() {
    document.querySelectorAll('.loop-widget').forEach(create);
  }

  HOST.LoopWidgets = { initAll: initAll, create: create, names: NAMES, staticHTML: staticHTML, frames: function (n, m) { return frames(EX[n], m); } };

  function start() {
    initAll();
    syncTheme();
    // Quarto's light/dark toggle swaps classes on <body>.
    if (window.MutationObserver && document.body) {
      new MutationObserver(syncTheme).observe(document.body, { attributes: true, attributeFilter: ['class'] });
    }
  }

  if (HAS_DOM) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
    else start();
  }
})();

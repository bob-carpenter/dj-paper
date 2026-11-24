/*!
 * stan-scatter-widget.js
 *
 * Interactive figure: Stan's nested loop that fills the lower-triangular
 * matrix A one scalar at a time, versus the JAX version that builds a
 * diagonal base and scatters z in a single call.
 *
 * Usage (any HTML page, including Quarto HTML output):
 *
 *   <div class="stan-scatter" data-k="4"></div>
 *   <script src="stan-scatter-widget.js"></script>
 *
 * In Quarto, put the code inside the div as ordinary code blocks (```stan and
 * ```python), as for loop-widgets.js. The widget then shows those blocks, in
 * the style of the paper's other code blocks, and the PDF prints them. Their
 * lines must be STAN and JAX below, the lines the steps highlight.
 *
 * Options, as data attributes on the div:
 *   data-k       matrix size K, 2 to 6, fixed      (default 4; no slider)
 *   data-mode    "code", "stan" or "jax"           (default "code")
 *   data-step    starting step, a number or "end"  (default 0)
 *   data-static  hide the controls (for print and screenshots)
 *
 * No dependencies. Styles are injected once and scoped to .ssw, so the
 * widget does not touch the rest of the page. Several widgets can sit on
 * one page. Dark mode follows Quarto's light/dark toggle (body.quarto-dark)
 * and otherwise the page's background colour.
 */
(function () {
  'use strict';

  if (window.StanScatter) {
    window.StanScatter.initAll();
    return;
  }

  var CSS = [
    '.ssw{--ssw-text-2:#5f5e5a;--ssw-text-3:#888780;--ssw-border:rgba(0,0,0,.14);--ssw-border-2:rgba(0,0,0,.3);',
    '--ssw-surface:#f7f6f2;--ssw-neutral:#ecebe4;--ssw-acc-bg:#e6f1fb;--ssw-acc-fg:#0c447c;--ssw-acc-bd:#378add;',
    '--ssw-z-bg:#eeedfe;--ssw-z-fg:#3c3489;--ssw-d-bg:#e1f5ee;--ssw-d-fg:#085041;',
    '--ssw-mono:var(--bs-font-monospace,ui-monospace,SFMono-Regular,Menlo,Consolas,"DejaVu Sans Mono",monospace);',
    'font-size:14px;line-height:1.5;text-align:left;margin:1rem 0}',
    '.ssw.ssw-dark{--ssw-text-2:#b4b2a9;--ssw-text-3:#8d8b84;--ssw-border:rgba(255,255,255,.14);--ssw-border-2:rgba(255,255,255,.3);',
    '--ssw-surface:rgba(255,255,255,.04);--ssw-neutral:rgba(255,255,255,.09);--ssw-acc-bg:#0c447c;--ssw-acc-fg:#b5d4f4;--ssw-acc-bd:#85b7eb;',
    '--ssw-z-bg:#3c3489;--ssw-z-fg:#cecbf6;--ssw-d-bg:#085041;--ssw-d-fg:#9fe1cb}',
    '.ssw *,.ssw *::before,.ssw *::after{box-sizing:border-box}',
    '.ssw [hidden]{display:none!important}',
    // Side by side when both blocks fit, one above the other when they do not. Side by side, the columns are equally
    // wide when each block fits in half the width; otherwise the wider block keeps its width and the other takes the
    // rest. (A flex item's automatic minimum width is its longest code line, capped at the max-width of 100%.)
    '.ssw-compare{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}',
    '.ssw-compare>div{flex:1 1 0;min-width:auto;max-width:100%}',
    '.ssw-ctitle{font-size:12px;font-weight:600;color:var(--ssw-text-2);margin:0 0 6px}',
    // The page's own code blocks: Quarto styles them; keep the page's size, drop their outer margins.
    '.ssw-qcode{font-size:var(--bs-body-font-size,1rem)}',
    '.ssw-qcode .code-copy-outer-scaffold,.ssw-qcode div.sourceCode{margin:0}',
    '.ssw-qcode code.sourceCode>span{min-width:100%}',
    '.ssw-qcode code.sourceCode>span.ssw-on{background:var(--ssw-acc-bg);box-shadow:inset 3px 0 0 var(--ssw-acc-bd)}',
    '.ssw-sr{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}',
    '.ssw .ssw-btn{font:inherit;font-size:13px;color:inherit;background:transparent;border:1px solid var(--ssw-border-2);',
    'border-radius:8px;padding:6px 12px;margin:0;cursor:pointer;line-height:1.3;display:inline-flex;align-items:center;gap:6px}',
    '.ssw .ssw-btn:hover{background:var(--ssw-neutral)}',
    '.ssw .ssw-btn:focus-visible{outline:2px solid var(--ssw-acc-bd);outline-offset:2px}',
    '.ssw .ssw-btn[aria-pressed="true"]{background:var(--ssw-neutral);border-color:currentColor}',
    '.ssw-top{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin:0 0 1rem}',
    '.ssw-seg{display:flex;flex-wrap:wrap;gap:6px}',
    '.ssw-main{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px;align-items:start}',
    '.ssw-mat{display:grid;gap:4px;max-width:340px}',
    '.ssw-cell{aspect-ratio:1/1;border-radius:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;',
    'gap:1px;font-family:var(--ssw-mono);font-size:12px;line-height:1.2;border:1px dashed var(--ssw-border-2);color:var(--ssw-text-3);transition:background-color .2s}',
    '.ssw-cell.is-z{background:var(--ssw-z-bg);color:var(--ssw-z-fg);border:1px solid transparent}',
    '.ssw-cell.is-d{background:var(--ssw-d-bg);color:var(--ssw-d-fg);border:1px solid transparent}',
    '.ssw-cell.is-0{background:var(--ssw-neutral);color:var(--ssw-text-2);border:1px solid transparent}',
    '.ssw-cell.is-cur{outline:2px solid var(--ssw-acc-bd);outline-offset:-2px}',
    '.ssw-cell.is-hl{outline:2px solid currentColor;outline-offset:-2px}',
    '.ssw-legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px;font-size:12px;color:var(--ssw-text-2)}',
    '.ssw-legend span{display:flex;align-items:center;gap:6px}',
    '.ssw-sw{width:12px;height:12px;border-radius:3px;display:inline-block}',
    '.ssw-code{font-family:var(--ssw-mono);font-size:12px;line-height:1.75;background:var(--ssw-surface);',
    'border:1px solid var(--ssw-border);border-radius:8px;padding:8px 4px;overflow-x:auto}',
    '.ssw-line{padding:0 8px;border-radius:4px;white-space:pre}',
    '.ssw-line.is-on{background:var(--ssw-acc-bg);color:var(--ssw-acc-fg)}',
    '.ssw-state{font-size:13px;line-height:1.6;color:var(--ssw-text-2);margin:10px 0 0;min-height:3.2em}',
    '.ssw-steps{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:1rem 0 0}',
    '.ssw-prog{font-size:13px;color:var(--ssw-text-3);margin-left:4px}',
    '.ssw-strips{margin-top:1.25rem}',
    '.ssw-strip-title{font-size:12px;color:var(--ssw-text-2);margin:0 0 6px}',
    '.ssw-strip{display:flex;flex-wrap:wrap;gap:4px;margin:0 0 12px}',
    '.ssw-pill{font-family:var(--ssw-mono);font-size:11px;padding:2px 6px;border-radius:6px;cursor:default}',
    '.ssw-pill.is-z{background:var(--ssw-z-bg);color:var(--ssw-z-fg)}',
    '.ssw-pill.is-pend{border:1px dashed var(--ssw-border-2);color:var(--ssw-text-3)}',
    '.ssw[data-static] .ssw-top,.ssw[data-static] .ssw-steps,.ssw[data-static] .ssw-state{display:none}',
    '@media (prefers-reduced-motion:reduce){.ssw-cell{transition:none}}'
  ].join('\n');

  var STAN = [
    'for (j in 1:(K - 1)) {',
    '  for (i in (j + 1):K) {',
    '    A[i, j] = z[count];',
    '    count += 1;',
    '  }',
    '  for (i in 1:(j - 1)) {',
    '    A[i, j] = 0.0;',
    '  }',
    '  A[j, j] = sqrt(c[j]);',
    '}',
    '// after the loop',
    'for (i in 1:(K - 1)) A[i, K] = 0;',
    'A[K, K] = sqrt(c[K]);'
  ];
  var JAX = [
    'rows = [i for j in range(K-1)',
    '          for i in range(j+1, K)]',
    'cols = [j for j in range(K-1)',
    '          for i in range(j+1, K)]',
    'A = (jnp.diag(jnp.sqrt(c))',
    '     .at[rows, cols].set(z))'
  ];
  var JAX_TEXT = [
    'The comprehensions replay the loop without touching A: they only record where each z[k] goes. count is just k, the position in that list.',
    'jnp.diag(jnp.sqrt(c)) lays down every diagonal entry and every zero at once, which covers the zero loop, the diagonal line, and the column-K tail.',
    null
  ];
  var MODES = ['code', 'stan', 'jax'];
  var PLAY = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.5v9l7.5-4.5z" fill="currentColor"/></svg> Play';
  var PAUSE = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="2.5" y="1.5" width="2.5" height="9" fill="currentColor"/><rect x="7" y="1.5" width="2.5" height="9" fill="currentColor"/></svg> Pause';

  var TEMPLATE =
    '<p class="ssw-sr">Interactive figure comparing Stan\'s nested loop, which fills the matrix A one cell at a time, ' +
    'with the JAX version, which builds a diagonal base and scatters z in one call.</p>' +
    '<div class="ssw-top">' +
      '<div class="ssw-seg" role="group" aria-label="View">' +
        '<button type="button" class="ssw-btn" data-mode="code">Compare code</button>' +
        '<button type="button" class="ssw-btn" data-mode="stan">Stan loop</button>' +
        '<button type="button" class="ssw-btn" data-mode="jax">JAX scatter</button>' +
      '</div>' +
    '</div>' +
    '<div class="ssw-compare" hidden></div>' +
    '<div class="ssw-main">' +
      '<div>' +
        '<div class="ssw-mat" role="img" aria-label="Matrix A"></div>' +
        '<div class="ssw-legend">' +
          '<span><i class="ssw-sw" style="background:var(--ssw-z-bg)"></i>z, strict lower</span>' +
          '<span><i class="ssw-sw" style="background:var(--ssw-d-bg)"></i>√c, diagonal</span>' +
          '<span><i class="ssw-sw" style="background:var(--ssw-neutral)"></i>0, upper</span>' +
        '</div>' +
      '</div>' +
      '<div>' +
        '<div class="ssw-codebox"></div>' +
        '<p class="ssw-state" aria-live="polite"></p>' +
      '</div>' +
    '</div>' +
    '<div class="ssw-steps">' +
      '<button type="button" class="ssw-btn" data-act="back" aria-label="Back one step">‹ Back</button>' +
      '<button type="button" class="ssw-btn" data-act="play">' + PLAY + '</button>' +
      '<button type="button" class="ssw-btn" data-act="next" aria-label="Forward one step">Next ›</button>' +
      '<button type="button" class="ssw-btn" data-act="reset">Reset</button>' +
      '<span class="ssw-prog"></span>' +
    '</div>' +
    '<div class="ssw-strips"></div>';

  /* ---------- pure helpers ---------- */

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // Stan's write order for z, 0-based: column j, then rows i > j.
  function trace(K) {
    var t = [];
    for (var j = 0; j < K - 1; j++) for (var i = j + 1; i < K; i++) t.push([i, j]);
    return t;
  }

  // Every scalar write Stan makes, in order (K*K of them).
  function stanEvents(K) {
    var ev = [], count = 1, i, j;
    for (j = 1; j <= K - 1; j++) {
      for (i = j + 1; i <= K; i++) {
        ev.push({ line: [2, 3], r: i - 1, c: j - 1, kind: 'z', k: count - 1,
          st: 'j = ' + j + ', i = ' + i + ', count = ' + count + ': writes z[' + (count - 1) + '] at row ' + (i - 1) +
              ', col ' + (j - 1) + ' (0-based), then count becomes ' + (count + 1) + '.' });
        count++;
      }
      for (i = 1; i <= j - 1; i++) {
        ev.push({ line: [6], r: i - 1, c: j - 1, kind: 'zero', st: 'j = ' + j + ', i = ' + i + ': zero above the diagonal.' });
      }
      ev.push({ line: [8], r: j - 1, c: j - 1, kind: 'diag', k: j - 1, st: 'j = ' + j + ': diagonal gets √c[' + (j - 1) + '].' });
    }
    for (i = 1; i <= K - 1; i++) {
      ev.push({ line: [11], r: i - 1, c: K - 1, kind: 'zero', st: 'i = ' + i + ': the loop never reached column K, so it is zeroed afterwards.' });
    }
    ev.push({ line: [12], r: K - 1, c: K - 1, kind: 'diag', k: K - 1, st: 'A[K, K] = √c[' + (K - 1) + '], the last diagonal entry the loop skipped.' });
    return ev;
  }

  function baseCells(K, cells) {
    for (var r = 0; r < K; r++) for (var c = r; c < K; c++) cells[r + ',' + c] = r === c ? { kind: 'diag', k: r } : { kind: 'zero' };
  }

  function totalSteps(mode, K) {
    return mode === 'stan' ? K * K : 2;
  }

  // done: null = all shown solid; a number = pills from that index on are pending.
  function strip(title, list, done) {
    var pills = list.map(function (rc, k) {
      var cls = 'ssw-pill' + (done !== null && k >= done ? ' is-pend' : ' is-z');
      return '<span class="' + cls + '" data-k="' + rc[0] + ',' + rc[1] + '">' + k + ':(' + rc[0] + ',' + rc[1] + ')</span>';
    }).join('');
    return '<div class="ssw-strip-title">' + esc(title) + '</div><div class="ssw-strip">' + pills + '</div>';
  }

  function codeHTML(src, on) {
    return '<div class="ssw-code">' + src.map(function (l, i) {
      return '<div class="ssw-line' + (on.indexOf(i) >= 0 ? ' is-on' : '') + '">' + (l ? esc(l) : '&nbsp;') + '</div>';
    }).join('') + '</div>';
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
    instances.forEach(function (root) { root.classList.toggle('ssw-dark', dark); });
  }

  function injectCSS() {
    if (document.getElementById('ssw-css')) return;
    var s = document.createElement('style');
    s.id = 'ssw-css';
    s.textContent = CSS;
    (document.head || document.documentElement).appendChild(s);
  }

  /* ---------- one widget ---------- */

  function create(root) {
    if (root.__ssw) return root.__ssw;
    injectCSS();

    var d = root.dataset;
    var K = Math.min(6, Math.max(2, parseInt(d.k, 10) || 4));
    var mode = MODES.indexOf(d.mode) >= 0 ? d.mode : 'code';
    var total = totalSteps(mode, K);
    var step = d.step === 'end' ? total : Math.min(total, Math.max(0, parseInt(d.step, 10) || 0));
    var timer = null;

    var page = takeBlocks(root), copies = {}, SRC = { stan: STAN, jax: JAX };
    root.classList.add('ssw');
    root.innerHTML = TEMPLATE;
    var q = function (sel) { return root.querySelector(sel); };
    var el = {
      mat: q('.ssw-mat'), codebox: q('.ssw-codebox'), state: q('.ssw-state'), strips: q('.ssw-strips'),
      prog: q('.ssw-prog'), play: q('[data-act="play"]'),
      main: q('.ssw-main'), steps: q('.ssw-steps'), compare: q('.ssw-compare')
    };

    // The comparison shows the page's code blocks themselves; the steps use copies of them.
    ['stan', 'jax'].forEach(function (m) {
      var col = document.createElement('div'), b = page[m];
      col.innerHTML = '<p class="ssw-ctitle">' + (m === 'stan' ? 'Stan' : 'JAX') + '</p>';
      if (b) {
        var box = document.createElement('div');
        box.className = 'ssw-qcode';
        box.appendChild(b);
        col.appendChild(box);
        if (sameCode(b, SRC[m])) copies[m] = stepCopy(b);
        else if (window.console) console.warn('stan-scatter-widget.js: the ' + m + ' code block in the page differs from ' +
                                              'the lines the steps highlight, so the steps show their own copy of the code.');
      } else {
        col.insertAdjacentHTML('beforeend', codeHTML(SRC[m], []));
      }
      el.compare.appendChild(col);
    });

    function setCode(m, on) {
      var c = copies[m];
      el.codebox.classList.toggle('ssw-qcode', !!c);
      if (!c) { el.codebox.innerHTML = codeHTML(SRC[m], on); return; }
      if (el.codebox.firstChild !== c.node) { el.codebox.textContent = ''; el.codebox.appendChild(c.node); }
      c.lines.forEach(function (ln, i) { ln.classList.toggle('ssw-on', on.indexOf(i) >= 0); });
    }

    function render() {
      var isCode = mode === 'code';
      el.main.hidden = isCode;
      el.steps.hidden = isCode;
      el.strips.hidden = isCode;
      el.compare.hidden = !isCode;
      root.querySelectorAll('button[data-mode]').forEach(function (b) {
        b.setAttribute('aria-pressed', String(b.getAttribute('data-mode') === mode));
      });
      if (isCode) return;
      var cells = {}, T = trace(K), on = [], st = '', cur = null, empty = '', strips = '', prog = '';
      total = totalSteps(mode, K);
      step = Math.min(step, total);

      if (mode === 'stan') {
        var ev = stanEvents(K);
        empty = 'NaN';
        for (var n = 0; n < step; n++) cells[ev[n].r + ',' + ev[n].c] = ev[n];
        if (step > 0) {
          var e = ev[step - 1];
          cur = e.r + ',' + e.c; on = e.line; st = e.st;
        } else {
          st = 'Stan initialises A to NaN, so the loop has to write all K² cells itself, one scalar at a time.';
        }
        if (step === total) st += ' Done: ' + total + ' separate writes.';
        var recorded = ev.slice(0, step).filter(function (x) { return x.kind === 'z'; }).length;
        strips = strip('Write addresses of z, recorded as the loop runs. This list is exactly rows, cols (0-based).', T, recorded);
        setCode('stan', on);
        prog = 'step ' + step + ' / ' + total;
      } else {
        if (step >= 1) baseCells(K, cells);
        if (step >= 2) T.forEach(function (rc, k) { cells[rc[0] + ',' + rc[1]] = { kind: 'z', k: k }; });
        on = [[0, 1, 2, 3], [4], [5]][step];
        st = JAX_TEXT[step] || ('One scatter drops all ' + T.length + ' z values at once: z[k] goes to (rows[k], cols[k]). Same matrix as the loop.');
        strips = strip('rows[k], cols[k] as k:(row, col)', T, null);
        setCode('jax', on);
        prog = 'phase ' + step + ' / 2';
      }

      el.prog.textContent = prog;

      el.mat.style.gridTemplateColumns = 'repeat(' + K + ',minmax(0,1fr))';
      var h = '';
      for (var r = 0; r < K; r++) {
        for (var c2 = 0; c2 < K; c2++) {
          var key = r + ',' + c2, x = cells[key], cls = 'ssw-cell', lab = empty;
          if (x) {
            cls += x.kind === 'z' ? ' is-z' : x.kind === 'diag' ? ' is-d' : ' is-0';
            lab = x.kind === 'z' ? 'z[' + x.k + ']' : x.kind === 'diag' ? '√c[' + x.k + ']' : '0';
          }
          if (key === cur) cls += ' is-cur';
          h += '<div class="' + cls + '" data-k="' + key + '"><span>' + lab + '</span></div>';
        }
      }
      el.mat.innerHTML = h;
      el.state.textContent = st;
      el.strips.innerHTML = strips;
    }

    function stop() {
      if (timer) { clearInterval(timer); timer = null; }
      el.play.innerHTML = PLAY;
      el.state.setAttribute('aria-live', 'polite');
    }
    function go(dir) {
      step = Math.max(0, Math.min(total, step + dir));
      render();
      if (step === total && dir > 0) stop();
    }
    function togglePlay() {
      if (timer) { stop(); return; }
      if (step >= total) step = 0;
      el.play.innerHTML = PAUSE;
      el.state.setAttribute('aria-live', 'off');
      render();
      timer = setInterval(function () {
        if (step >= total) { stop(); return; }
        go(1);
      }, mode === 'stan' ? 420 : 1100);
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
    el.strips.addEventListener('mouseover', function (ev) {
      var k = ev.target.getAttribute && ev.target.getAttribute('data-k');
      if (!k) return;
      var cell = el.mat.querySelector('[data-k="' + k + '"]');
      if (cell) cell.classList.add('is-hl');
    });
    el.strips.addEventListener('mouseout', function () {
      el.mat.querySelectorAll('.is-hl').forEach(function (c) { c.classList.remove('is-hl'); });
    });

    render();
    instances.push(root);
    root.classList.toggle('ssw-dark', pageIsDark());
    root.__ssw = { render: render, stop: stop };
    return root.__ssw;
  }

  function initAll() {
    document.querySelectorAll('.stan-scatter').forEach(create);
  }

  window.StanScatter = { initAll: initAll, create: create };

  function start() {
    initAll();
    syncTheme();
    // Quarto's light/dark toggle swaps classes on <body>.
    if (window.MutationObserver && document.body) {
      new MutationObserver(syncTheme).observe(document.body, { attributes: true, attributeFilter: ['class'] });
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();

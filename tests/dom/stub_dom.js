// A tiny stub DOM for callqa/static/index.html. Plain node, no npm, no network.
//
// Usage: node stub_dom.js <page.html> <scenario.json>
// The scenario holds canned answers for fetch():
//   { "routes": { "GET /api/config": {"status": 200, "body": {...}}, ... },
//     "sequences": { "GET /api/run/r1": [ {...}, {...} ] },   // one answer per call, the last repeats
//     "clicks": ["#run-button", "button.fixcall:Wants a manager"],
//     "steps": [ {"select": "#old-select", "value": "a.json"}, {"click": "#compare-button", "text": ""} ],  // after the clicks
//     "queries": { "slow": "#calls .turn.slow" } }   // each prints a list of {id, tag, className, text, hidden, open, href}
// A click is "selector" or "selector:text the element must contain".
// It loads the page's inline script, performs the clicks, lets promises and timers settle (a fake
// clock, so nothing really waits) and prints one JSON object of DOM facts for the Python tests.
'use strict';
const fs = require('fs');
const vm = require('vm');

class TextNode {
  constructor(data) { this.nodeType = 3; this.data = String(data); this.parentNode = null; }
  get textContent() { return this.data; }
}

class Element {
  constructor(tag) {
    this.nodeType = 1;
    this.tagName = tag.toUpperCase();
    this.id = '';
    this.className = '';
    this.childNodes = [];
    this.parentNode = null;
    this.attrs = {};
    this.listeners = {};
    this.hidden = false;
    this.open = false;
    this.disabled = false;
    this.value = '';
    this.title = '';
    this.scrolled = 0;
    const self = this;
    this.classList = {
      contains(name) { return self.className.split(/\s+/).indexOf(name) !== -1; },
      add(name) { if (!this.contains(name)) { self.className = (self.className + ' ' + name).trim(); } },
      toggle(name) {
        if (this.contains(name)) {
          self.className = self.className.split(/\s+/).filter(function (c) { return c && c !== name; }).join(' ');
        } else { this.add(name); }
      }
    };
  }
  get firstChild() { return this.childNodes[0] || null; }
  get children() { return this.childNodes.filter(function (n) { return n.nodeType === 1; }); }
  appendChild(node) {
    if (node.parentNode) { node.parentNode.removeChild(node); }
    node.parentNode = this;
    this.childNodes.push(node);
    return node;
  }
  removeChild(node) {
    const at = this.childNodes.indexOf(node);
    if (at !== -1) { this.childNodes.splice(at, 1); node.parentNode = null; }
    return node;
  }
  get textContent() { return this.childNodes.map(function (n) { return n.textContent; }).join(''); }
  set textContent(text) {
    this.childNodes.forEach(function (n) { n.parentNode = null; });
    this.childNodes = [];
    if (text !== '' && text !== null && text !== undefined) { this.appendChild(new TextNode(text)); }
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; }
  hasAttribute(name) { return name in this.attrs; }
  removeAttribute(name) { delete this.attrs[name]; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  dispatch(type) {
    const event = { type: type, target: this, preventDefault() {}, stopPropagation() {} };
    (this.listeners[type] || []).forEach(function (fn) { fn(event); });
  }
  scrollIntoView(options) { this.scrolled += 1; (this.scrollBlocks = this.scrollBlocks || []).push(options && options.block); }
  // Selectors: "tag", ".a.b", "tag.a", "#id", and descendants of those separated by spaces ("#calls .turn.slow").
  matches(selector) {
    const parts = selector.match(/^([a-zA-Z]*)((?:\.[\w-]+)*)(?:#([\w-]+))?$/);
    if (!parts) { throw new Error('stub selector not supported: ' + selector); }
    if (parts[1] && this.tagName !== parts[1].toUpperCase()) { return false; }
    if (parts[3] && this.id !== parts[3]) { return false; }
    return parts[2].split('.').filter(Boolean).every(function (c) { return this.classList.contains(c); }, this);
  }
  matchesChain(chain) {
    if (!this.matches(chain[chain.length - 1])) { return false; }
    let at = chain.length - 2;
    for (let up = this.parentNode; at >= 0 && up; up = up.parentNode) {
      if (up.nodeType === 1 && up.matches(chain[at])) { at -= 1; }
    }
    return at < 0;
  }
  querySelectorAll(selector) {
    const found = [];
    const chain = selector.trim().split(/\s+/);
    (function walk(node) {
      node.children.forEach(function (child) {
        if (child.matchesChain(chain)) { found.push(child); }
        walk(child);
      });
    })(this);
    return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

function buildDocument(markup) {
  const body = new Element('body');
  // Every element in the markup that has an id gets a stub, so the page's getElementById works.
  const tag = /<(\w+)([^>]*\sid="([\w-]+)"[^>]*)>/g;
  let m;
  while ((m = tag.exec(markup)) !== null) {
    const node = new Element(m[1]);
    node.id = m[3];
    node.hidden = /\shidden(\s|$)/.test(m[2]);
    const attr = /([\w-]+)="([^"]*)"/g;  // the markup's own attributes, for href and the like
    let a;
    while ((a = attr.exec(m[2])) !== null) { if (a[1] !== 'id' && a[1] !== 'class') { node.attrs[a[1]] = a[2]; } }
    body.appendChild(node);
  }
  const listeners = {};
  return {
    body: body,
    listeners: listeners,
    addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
    createElement(name) { return new Element(name); },
    createTextNode(text) { return new TextNode(text); },
    getElementById(id) { return body.querySelectorAll('#' + id)[0] || null; }
  };
}

async function main() {
  const html = fs.readFileSync(process.argv[2], 'utf8');
  const scenario = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  const script = /<script>([\s\S]*?)<\/script>/.exec(html)[1];
  const document = buildDocument(html.split('<script>')[0]);

  const fetchLog = [];
  const fetchBodies = [];  // {key, body} for every request that sent a body
  const seen = {};
  function fetch(url, options) {
    const method = ((options && options.method) || 'GET').toUpperCase();
    const key = method + ' ' + url;
    fetchLog.push(key);
    if (options && options.body) { fetchBodies.push({ key: key, body: JSON.parse(options.body) }); }
    let answer = scenario.routes && scenario.routes[key];
    const sequence = scenario.sequences && scenario.sequences[key];
    if (sequence) {
      seen[key] = seen[key] || 0;
      answer = sequence[Math.min(seen[key], sequence.length - 1)];
      seen[key] += 1;
    }
    if (!answer) { return Promise.reject(new Error('stub fetch: no answer for ' + key)); }
    const status = answer.status === undefined ? 200 : answer.status;
    const body = answer.body === undefined ? {} : answer.body;
    return Promise.resolve({
      ok: status >= 200 && status < 300,
      status: status,
      json() { return Promise.resolve(JSON.parse(JSON.stringify(body))); }
    });
  }

  // Fake clock: timers queue up and are run by hand, so polling is visible and nothing really waits.
  let now = 0;
  const timers = [];
  function setTimeout(fn, ms) { timers.push({ at: now + (ms || 0), fn: fn }); return timers.length; }

  const sandbox = { document: document, fetch: fetch, setTimeout: setTimeout, encodeURIComponent: encodeURIComponent };
  vm.createContext(sandbox);
  const settle = function () { return new Promise(function (resolve) { setImmediate(resolve); }); };

  vm.runInContext(script, sandbox, { filename: 'index.html:script' });

  async function drain() {
    for (let step = 0; step < 200; step += 1) {
      await settle();
      if (timers.length === 0) { return; }
      timers.sort(function (a, b) { return a.at - b.at; });
      const next = timers.shift();
      now = next.at;
      next.fn();
    }
    throw new Error('stub: timers never settled (page polls forever?)');
  }
  await drain();

  for (const target of (scenario.clicks || [])) {
    const at = target.indexOf(':');
    const selector = at === -1 ? target : target.slice(0, at);
    const text = at === -1 ? '' : target.slice(at + 1);
    const matches = document.body.querySelectorAll(selector).filter(function (n) {
      return n.textContent.indexOf(text) !== -1;
    });
    if (matches.length === 0) { throw new Error('stub: nothing to click for ' + target); }
    matches[0].dispatch('click');
    await drain();
  }

  for (const step of (scenario.steps || [])) {
    if (step.key) {  // a keydown on the document, with the element named by step.target as event.target
      const where = document.body.querySelectorAll(step.target)[0];
      if (!where) { throw new Error('stub: no key target ' + step.target); }
      (document.listeners.keydown || []).forEach(function (fn) {
        fn({ type: 'keydown', key: step.key, target: where, ctrlKey: false, metaKey: false, altKey: false });
      });
      await drain();
      continue;
    }
    const target = document.body.querySelectorAll(step.select || step.click).filter(function (n) {
      return n.textContent.indexOf(step.text || '') !== -1;
    })[0];
    if (!target) { throw new Error('stub: nothing for step ' + JSON.stringify(step)); }
    if (step.select) { target.value = step.value; target.dispatch('change'); } else { target.dispatch('click'); }
    await drain();
  }

  function summary(n) {
    return { id: n.id, tag: n.tagName.toLowerCase(), className: n.className, text: n.textContent, hidden: n.hidden,
             open: n.open, disabled: n.disabled, href: n.getAttribute('href'), value: n.value,
             attrs: n.attrs, scrollBlocks: n.scrollBlocks || [],
             childTags: n.children.map(function (k) { return k.tagName.toLowerCase(); }),
             childClasses: n.children.map(function (k) { return k.className; }),
             childTexts: n.children.map(function (k) { return k.textContent; }) };
  }

  const byId = function (id) { return document.getElementById(id); };
  const out = byId('pass-rate') ? {
    passRate: { text: byId('pass-rate').textContent, className: byId('pass-rate').className },
    passSummary: byId('pass-summary').textContent,
    runMeta: byId('run-meta').textContent,
    modeBadge: byId('mode-badge').textContent,
    beforeAfter: { hidden: byId('before-after').hidden, text: byId('before-after').textContent },
    errorBanner: { hidden: byId('error-banner').hidden, text: byId('error-banner').textContent },
    progress: { hidden: byId('progress').hidden, text: byId('progress').textContent },
    runButtonDisabled: byId('run-button').disabled,
    loadingHidden: byId('loading').hidden,
    nextFix: {
      hidden: byId('next-fix').hidden,
      className: byId('next-fix').className,
      buttons: byId('next-fix').querySelectorAll('button.fixcall').map(function (b) { return b.textContent; })
    },
    cards: document.body.querySelectorAll('details.call').map(function (c) {
      return {
        id: c.id,
        className: c.className,
        open: c.open,
        verdict: c.querySelector('.verdict').textContent,
        marks: c.querySelectorAll('mark').map(function (n) { return n.textContent; }),
        failingTurns: c.querySelectorAll('.turn.failing').length
      };
    }),
    marks: document.body.querySelectorAll('mark').map(function (n) { return n.textContent; }),
    sopValue: byId('sop-input').value,
    agentOptions: byId('agent-select').children.map(function (o) { return o.value; }),
    pendingTimers: timers.length,
    fetchLog: fetchLog
  } : { fetchLog: fetchLog, pendingTimers: timers.length };
  out.fetchBodies = fetchBodies;
  out.queries = {};
  Object.keys(scenario.queries || {}).forEach(function (key) {
    out.queries[key] = document.body.querySelectorAll(scenario.queries[key]).map(summary);
  });
  process.stdout.write(JSON.stringify(out));
}

main().catch(function (err) {
  process.stderr.write(String((err && err.stack) || err) + '\n');
  process.exit(1);
});

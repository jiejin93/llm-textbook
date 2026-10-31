/* 解锁系统（前端）：从 #llm-locked-payload 读取加密正文，
   PBKDF2(解锁码) 派生密钥 → SHA-256 计数器流解密 → 注入正文 → 局部重渲染公式。
   解锁码校验依赖内嵌的 SHA-256(明文) 摘要：错误码解出乱码必然摘要不符。 */
(function () {
  var payloadEl = document.getElementById("llm-locked-payload");
  var root = document.getElementById("llm-locked");
  if (!payloadEl || !root) return;

  var CODE_KEY = "llmbook.unlock";
  var payload = JSON.parse(payloadEl.textContent);
  var enc = payload.enc;

  function b64ToU8(b64) {
    var bin = atob(b64);
    var u8 = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return u8;
  }

  function u8Concat() {
    var len = 0, i;
    for (i = 0; i < arguments.length; i++) len += arguments[i].length;
    var out = new Uint8Array(len);
    var off = 0;
    for (i = 0; i < arguments.length; i++) { out.set(arguments[i], off); off += arguments[i].length; }
    return out;
  }

  function u32be(n) {
    return new Uint8Array([(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255]);
  }

  function hex(buf) {
    var u8 = new Uint8Array(buf), s = "";
    for (var i = 0; i < u8.length; i++) s += ("0" + u8[i].toString(16)).slice(-2);
    return s;
  }

  async function deriveKeyMaterial(code, saltU8, iters) {
    var km = await crypto.subtle.importKey(
      "raw", new TextEncoder().encode(code), "PBKDF2", false, ["deriveBits"]);
    return new Uint8Array(
      await crypto.subtle.deriveBits({ name: "PBKDF2", salt: saltU8, iterations: iters, hash: "SHA-256" }, km, 256));
  }

  async function decryptWith(code) {
    var km = await deriveKeyMaterial(code, b64ToU8(enc.salt), enc.iters);
    var nonce = b64ToU8(enc.nonce);
    var ct = b64ToU8(enc.ct);
    var ks = new Uint8Array(ct.length);
    var blocks = Math.ceil(ct.length / 32) + 1;
    var jobs = [];
    for (var i = 0; i < blocks; i++) {
      jobs.push(crypto.subtle.digest("SHA-256", u8Concat(km, nonce, u32be(i))));
    }
    var digests = await Promise.all(jobs);
    var views = digests.map(function (d) { return new Uint8Array(d); });
    for (var j = 0; j < ct.length; j++) {
      ks[j] = views[j >> 5][j & 31];
    }
    var plain = new Uint8Array(ct.length);
    for (var k = 0; k < ct.length; k++) plain[k] = ct[k] ^ ks[k];
    var macU8 = new Uint8Array(await crypto.subtle.digest("SHA-256", plain));
    var macBin = "";
    for (var m = 0; m < macU8.length; m++) macBin += String.fromCharCode(macU8[m]);
    if (btoa(macBin) !== enc.mac) throw new Error("BAD_CODE");
    return new TextDecoder().decode(plain);
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function panelHTML() {
    return ''
      + '<section class="lock-panel">'
      + '<div class="lock-badge">&#128274;</div>'
      + '<h2>本章为付费内容</h2>'
      + '<p class="lock-desc">你正在阅读免费试读版：上面是本章导读，下方是本章目录。'
      + '解锁后可获得<b>全部 22 章 + 3 附录</b>的在线阅读与 PDF 离线版，'
      + '以及后续所有更新。</p>'
      + '<ol>'
      + '<li>微信扫描下方二维码添加好友，备注「<b>llm 教材</b>」</li>'
      + '<li>转账 <b>&yen;49.9</b>，说明要「在线解锁码」或「PDF 离线版」</li>'
      + '<li>在本页输入解锁码，<b>一个码解锁全站</b>，本浏览器会记住</li>'
      + '</ol>'
      + '<img class="lock-qr" src="../assets/wechat-qr.jpg" alt="微信扫码添加好友">'
      + '<p class="lock-qr-caption">微信扫码 · 备注「llm 教材」</p>'
      + '<div class="unlock-form">'
      + '<input type="text" id="llm-code-input" placeholder="输入解锁码" autocomplete="off">'
      + '<button type="button" id="llm-code-btn">解锁</button>'
      + '</div>'
      + '<p class="unlock-msg" id="llm-code-msg"></p>'
      + '<p class="lock-fine">已付款但解锁码失效？微信联系我重发即可。'
      + '价格与权益详情见<a href="../buy/">获取完整版</a>。</p>'
      + '</section>';
  }

  function renderLocked() {
    var html = payload.teaser
      + '<nav class="lock-toc"><p class="lock-toc-title">本章目录（' + payload.outline.length + ' 节）</p><ol>'
      + payload.outline.map(function (o) {
          return '<li><a href="#' + esc(o[0]) + '">' + esc(o[1]) + '</a></li>';
        }).join("")
      + '</ol></nav>'
      + panelHTML();
    root.innerHTML = html;
    bindForm();
  }

  function renderUnlocked(html) {
    root.innerHTML = '<div class="unlocked-strip">&#10003; 已解锁 · 感谢支持，祝阅读愉快</div>' + html;
    if (typeof window.renderBookMath === "function") window.renderBookMath(root);
  }

  function bindForm() {
    var btn = document.getElementById("llm-code-btn");
    var input = document.getElementById("llm-code-input");
    var msg = document.getElementById("llm-code-msg");
    if (!btn || !input) return;
    async function tryUnlock() {
      var code = (input.value || "").trim();
      if (!code) { msg.textContent = "请输入解锁码"; msg.className = "unlock-msg err"; return; }
      btn.disabled = true;
      btn.textContent = "解锁中…";
      try {
        var html = await decryptWith(code);
        localStorage.setItem(CODE_KEY, code);
        msg.textContent = "";
        renderUnlocked(html);
      } catch (e) {
        msg.textContent = "解锁码不正确，请核对后重试（微信联系作者可重发）";
        msg.className = "unlock-msg err";
        btn.disabled = false;
        btn.textContent = "解锁";
      }
    }
    btn.addEventListener("click", tryUnlock);
    input.addEventListener("keydown", function (ev) { if (ev.key === "Enter") tryUnlock(); });
  }

  (async function boot() {
    var saved = null;
    try { saved = localStorage.getItem(CODE_KEY); } catch (e) { /* 隐私模式 */ }
    if (saved) {
      try {
        var html = await decryptWith(saved);
        renderUnlocked(html);
        return;
      } catch (e) {
        try { localStorage.removeItem(CODE_KEY); } catch (e2) {}
      }
    }
    renderLocked();
  })();
})();

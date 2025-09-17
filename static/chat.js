const chatbox = document.getElementById('chatbox');
const form = document.getElementById('chatform');
const uname = document.getElementById('username');
const msgInput = document.getElementById('message');

const submitBtn = form?.querySelector('button');

// ——— Helpers ———
function addMsg(text, who = 'you') {
  const div = document.createElement('div');
  div.className = `chatmsg ${who}`;
  div.textContent = text;
  chatbox.appendChild(div);
  chatbox.scrollTop = chatbox.scrollHeight;
  return div;
}

function setTyping(on = true) {
  if (on) {
    if (!document.getElementById('typing-ind')) {
      const div = document.createElement('div');
      div.id = 'typing-ind';
      div.className = 'chatmsg bot';
      div.textContent = '… mengetik';
      chatbox.appendChild(div);
      chatbox.scrollTop = chatbox.scrollHeight;
    }
  } else {
    const t = document.getElementById('typing-ind');
    if (t) t.remove();
  }
}

function setBusy(busy) {
  if (!submitBtn) return;
  submitBtn.disabled = busy;
  if (busy) {
    submitBtn.dataset._old = submitBtn.textContent;
    submitBtn.textContent = 'Mengirim…';
  } else if (submitBtn.dataset._old) {
    submitBtn.textContent = submitBtn.dataset._old;
    delete submitBtn.dataset._old;
  }
}

// ——— init small UX ———
try {
  const savedName = localStorage.getItem('sebayu_name');
  if (savedName) uname.value = savedName;
} catch {}
msgInput?.focus();

// simpan nama otomatis
uname?.addEventListener('change', () => {
  try { localStorage.setItem('sebayu_name', uname.value.trim()); } catch {}
});

// Enter = submit, Esc = clear
msgInput?.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    msgInput.value = '';
  }
});

// ——— main submit ———
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = msgInput.value.trim();
  if (!text) return;
  const username = (uname.value.trim() || 'web-user');

  addMsg(text, 'you');
  msgInput.value = '';
  setBusy(true);
  setTyping(true);

  // fungsi request dengan 1x retry kalau error jaringan
  async function sendOnce() {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ text, username })
    });
    if (!res.ok) {
      // tampilkan pesan yang enak dibaca
      const txt = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} ${res.statusText}${txt ? ' - ' + txt : ''}`);
    }
    return res.json();
  }

  try {
    let data;
    try {
      data = await sendOnce();
    } catch (err) {
      // retry sekali
      data = await sendOnce();
    }
    setTyping(false);
    addMsg((data && data.reply) ? data.reply : '(tidak ada jawaban)', 'bot');
  } catch (err) {
    setTyping(false);
    addMsg('Gagal menghubungi server. Coba lagi sebentar ya.', 'bot');
    console.error(err);
  } finally {
    setBusy(false);
    msgInput.focus();
  }
});

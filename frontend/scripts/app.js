const elements = {
  chatForm: document.querySelector('#chat-form'),
  userInput: document.querySelector('#user-input'),
  chatLog: document.querySelector('#chat-log'),
  statusBar: document.querySelector('#status-bar'),
  backendUrl: document.querySelector('#backend-url'),
  langSelect: document.querySelector('#lang-select'),
  characterSelect: document.querySelector('#character-select'),
  systemPrompt: document.querySelector('#system-prompt'),
  kInput: document.querySelector('#k-input'),
  tempInput: document.querySelector('#temp-input'),
  modelInput: document.querySelector('#model-input'),
  newSessionBtn: document.querySelector('#new-session-btn'),
  resetLocalBtn: document.querySelector('#reset-local-btn'),
  clearChatBtn: document.querySelector('#clear-chat-btn'),
  sessionInfo: document.querySelector('#session-info code'),
};

const storageKey = 'chatmycv-settings';
const defaultSettings = {
  backendUrl: 'http://localhost:8000',
  lang: 'en',
  character: '',
  temperature: 0.7,
  k: 5,
  model: '',
  systemPrompt: '',
};

const state = {
  sessionId: null,
  isSending: false,
};

function loadSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem(storageKey));
    if (!stored) return;
    Object.assign(defaultSettings, stored);
  } catch (error) {
    console.warn('Unable to load settings from localStorage', error);
  }
}

function applySettingsToUI() {
  const cfg = defaultSettings;
  elements.backendUrl.value = cfg.backendUrl;
  elements.langSelect.value = cfg.lang;
  elements.characterSelect.value = cfg.character;
  elements.tempInput.value = cfg.temperature;
  elements.kInput.value = cfg.k;
  elements.modelInput.value = cfg.model;
  elements.systemPrompt.value = cfg.systemPrompt;
}

function persistSettings() {
  const settings = {
    backendUrl: elements.backendUrl.value.trim() || defaultSettings.backendUrl,
    lang: elements.langSelect.value,
    character: elements.characterSelect.value,
    temperature: parseFloat(elements.tempInput.value) || defaultSettings.temperature,
    k: parseInt(elements.kInput.value, 10) || defaultSettings.k,
    model: elements.modelInput.value.trim(),
    systemPrompt: elements.systemPrompt.value.trim(),
  };

  localStorage.setItem(storageKey, JSON.stringify(settings));
}

function setStatus(message, type = 'info') {
  elements.statusBar.textContent = message;
  elements.statusBar.className = `status-bar ${type}`;
}

function updateSessionInfo() {
  elements.sessionInfo.textContent = state.sessionId || 'not started';
}

function ensureUrl(path) {
  const base = elements.backendUrl.value.trim() || defaultSettings.backendUrl;
  try {
    return new URL(path, base).toString();
  } catch (error) {
    throw new Error('Backend URL is invalid');
  }
}

function addMessage(role, content, meta = '') {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;

  const paragraphs = content.split(/\n{2,}/);
  paragraphs.forEach((block, idx) => {
    const p = document.createElement('p');
    p.className = 'message-content';
    p.textContent = block;
    wrapper.appendChild(p);
    if (idx < paragraphs.length - 1) {
      const spacer = document.createElement('div');
      spacer.style.height = '8px';
      wrapper.appendChild(spacer);
    }
  });

  if (meta) {
    const metaEl = document.createElement('div');
    metaEl.className = 'message-meta';
    metaEl.textContent = meta;
    wrapper.appendChild(metaEl);
  }

  elements.chatLog.appendChild(wrapper);
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
  return wrapper;
}

function toggleForm(disabled) {
  elements.chatForm.querySelector('button[type="submit"]').disabled = disabled;
  elements.userInput.disabled = disabled;
}

function buildPayload(query) {
  const payload = {
    lang: elements.langSelect.value,
    query,
    session_id: state.sessionId,
    k: parseInt(elements.kInput.value, 10) || defaultSettings.k,
    temperature: parseFloat(elements.tempInput.value) || defaultSettings.temperature,
  };

  const character = elements.characterSelect.value;
  const systemPrompt = elements.systemPrompt.value.trim();
  const model = elements.modelInput.value.trim();

  if (character) payload.character = character;
  if (systemPrompt) payload.system_prompt = systemPrompt;
  if (model) payload.model = model;

  return payload;
}

function formatMeta(responseData) {
  const parts = [];
  if (responseData.character) parts.push(`Character: ${responseData.character}`);
  if (responseData.retrieved_docs_count !== undefined) {
    parts.push(`Docs: ${responseData.retrieved_docs_count}`);
  }
  if (responseData.context_used) parts.push('Used RAG context');

  const usage = responseData.usage || {};
  if (usage.total_tokens) {
    const tokens = [
      usage.prompt_tokens ? `P:${usage.prompt_tokens}` : null,
      usage.completion_tokens ? `C:${usage.completion_tokens}` : null,
      `T:${usage.total_tokens}`,
    ].filter(Boolean).join(' / ');
    parts.push(`Tokens ${tokens}`);
  }

  return parts.join(' • ');
}

async function handleSend(event) {
  event.preventDefault();
  if (state.isSending) return;

  const query = elements.userInput.value.trim();
  if (!query) return;

  persistSettings();
  addMessage('user', query);
  elements.userInput.value = '';

  try {
    state.isSending = true;
    toggleForm(true);
    setStatus('Sending...', 'info');

    const response = await fetch(ensureUrl('/chat/'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(buildPayload(query)),
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.error || `Request failed (${response.status})`);
    }

    const data = await response.json();
    state.sessionId = data.session_id || state.sessionId;
    updateSessionInfo();

    // Keep analytics in dev tools only; do not show to end users
    console.debug('ChatMyCV usage/meta:', formatMeta(data));
    addMessage('assistant', data.response || '[empty response]');
    setStatus('Response received', 'success');
  } catch (error) {
    console.error(error);
    addMessage('assistant', `⚠️ ${error.message}`);
    setStatus(error.message, 'error');
  } finally {
    state.isSending = false;
    toggleForm(false);
    elements.userInput.focus();
  }
}

function resetLocalChat() {
  elements.chatLog.innerHTML = '';
  setStatus('Local history cleared', 'info');
}

function startNewSession() {
  state.sessionId = null;
  updateSessionInfo();
  resetLocalChat();
  setStatus('Started a new session', 'success');
}

async function clearRemoteHistory() {
  if (!state.sessionId) {
    setStatus('No active session to clear on server', 'warning');
    return;
  }

  try {
    setStatus('Clearing remote history...', 'info');
    const response = await fetch(ensureUrl('/chat/clear'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ session_id: state.sessionId }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || 'Failed to clear history');
    }

    const result = await response.json();
    if (result.status === 'success') {
      setStatus('Remote history cleared', 'success');
    } else {
      throw new Error(result.error || 'Failed to clear history');
    }
  } catch (error) {
    setStatus(error.message, 'error');
  }
}

function initEventListeners() {
  elements.chatForm.addEventListener('submit', handleSend);
  elements.userInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!state.isSending) {
        elements.chatForm.requestSubmit();
      }
    }
  });
  elements.resetLocalBtn.addEventListener('click', resetLocalChat);
  elements.newSessionBtn.addEventListener('click', startNewSession);
  elements.clearChatBtn.addEventListener('click', clearRemoteHistory);

  [
    elements.backendUrl,
    elements.langSelect,
    elements.characterSelect,
    elements.tempInput,
    elements.kInput,
    elements.modelInput,
    elements.systemPrompt,
  ].forEach((input) => input.addEventListener('change', persistSettings));
}

function init() {
  loadSettings();
  applySettingsToUI();
  updateSessionInfo();
  initEventListeners();
  setStatus('Ready', 'info');
}

init();

const chatHistory = document.getElementById('chatHistory');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const sessionList = document.getElementById('sessionList');

const STORAGE_SESSIONS = 'metro_sessions';
const STORAGE_ACTIVE = 'metro_active_session';

let sessions = [];
let activeSessionId = null;
let historyContext = [];

// Auto-resize textarea
userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if (this.value === '') this.style.height = 'auto';
});

function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function clearChat() {
    if (!activeSessionId) return;
    if (confirm('确定要清空当前会话吗？')) {
        historyContext = [];
        updateSessionHistory(activeSessionId, []);
        renderChatFromHistory([]);
    }
}

function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const icon = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

    msgDiv.innerHTML = `
        <div class="avatar">${icon}</div>
        <div class="bubble">
            <div class="content"></div>
        </div>
    `;

    chatHistory.appendChild(msgDiv);

    if (role === 'assistant') {
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return msgDiv.querySelector('.content');
    } else {
        msgDiv.querySelector('.content').textContent = text;
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return null;
    }
}

function createWelcomeMessage() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant';
    msgDiv.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="bubble">
            <div class="content">
                您好，我是城市轨道交通应急处置助手。我可以为您提供关于
                <b>火灾</b>、<b>信号故障</b>、<b>人流拥挤</b>
                等突发事件的标准处置流程查询。<br><br>
                请直接描述您遇到的情况，例如：“站台发现火情怎么办？”
            </div>
        </div>
    `;
    chatHistory.appendChild(msgDiv);
}

function renderChatFromHistory(history) {
    chatHistory.innerHTML = '';
    createWelcomeMessage();
    history.forEach(msg => {
        if (msg.role === 'user') {
            appendMessage('user', msg.content);
        } else if (msg.role === 'assistant') {
            const contentDiv = appendMessage('assistant', '');
            contentDiv.innerHTML = marked.parse(msg.content || '');
        }
    });
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function loadSessions() {
    try {
        sessions = JSON.parse(localStorage.getItem(STORAGE_SESSIONS) || '[]');
    } catch {
        sessions = [];
    }
    activeSessionId = localStorage.getItem(STORAGE_ACTIVE);
    if (!sessions.length) {
        createNewSession();
        return;
    }
    if (!activeSessionId || !sessions.find(s => s.id === activeSessionId)) {
        activeSessionId = sessions[0].id;
        localStorage.setItem(STORAGE_ACTIVE, activeSessionId);
    }
    setActiveSession(activeSessionId);
    renderSessionList();
}

function saveSessions() {
    localStorage.setItem(STORAGE_SESSIONS, JSON.stringify(sessions));
}

function createNewSession() {
    const id = `s_${Date.now()}`;
    const session = {
        id,
        title: '新会话',
        history: []
    };
    sessions.unshift(session);
    activeSessionId = id;
    saveSessions();
    localStorage.setItem(STORAGE_ACTIVE, activeSessionId);
    renderSessionList();
    setActiveSession(activeSessionId);
}

function deleteSession(id) {
    const idx = sessions.findIndex(s => s.id === id);
    if (idx === -1) return;
    if (!confirm('确定要删除该会话吗？')) return;
    sessions.splice(idx, 1);
    if (!sessions.length) {
        createNewSession();
        return;
    }
    if (activeSessionId === id) {
        activeSessionId = sessions[0].id;
        localStorage.setItem(STORAGE_ACTIVE, activeSessionId);
    }
    saveSessions();
    renderSessionList();
    setActiveSession(activeSessionId);
}

function setActiveSession(id) {
    const session = sessions.find(s => s.id === id);
    if (!session) return;
    activeSessionId = id;
    localStorage.setItem(STORAGE_ACTIVE, activeSessionId);
    historyContext = session.history || [];
    renderChatFromHistory(historyContext);
    renderSessionList();
}

function updateSessionHistory(id, history) {
    const session = sessions.find(s => s.id === id);
    if (!session) return;
    session.history = history;
    saveSessions();
}

function updateSessionTitle(id, text) {
    const session = sessions.find(s => s.id === id);
    if (!session) return;
    if (session.title === '新会话') {
        session.title = text.slice(0, 12);
        saveSessions();
        renderSessionList();
    }
}

function renderSessionList() {
    if (!sessionList) return;
    sessionList.innerHTML = '';
    sessions.forEach(s => {
        const item = document.createElement('div');
        item.className = `nav-item session-item ${s.id === activeSessionId ? 'active' : ''}`;
        item.innerHTML = `
            <span class="session-title">${s.title}</span>
            <button class="session-delete" title="删除会话">
                <i class="fa-solid fa-xmark"></i>
            </button>
        `;
        item.onclick = () => setActiveSession(s.id);
        item.querySelector('.session-delete').onclick = (e) => {
            e.stopPropagation();
            deleteSession(s.id);
        };
        sessionList.appendChild(item);
    });
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    appendMessage('user', text);

    const contentDiv = appendMessage('assistant', '');
    contentDiv.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> 思考中...';

    let fullResponse = '';

    try {
        const response = await fetch('/api/v1/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: text,
                history: historyContext,
                stream: true
            })
        });

        if (!response.ok) throw new Error('Network response was not ok');

        contentDiv.innerHTML = '';
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            fullResponse += chunk;
            contentDiv.innerHTML = marked.parse(fullResponse);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }

        historyContext.push({ role: 'user', content: text });
        historyContext.push({ role: 'assistant', content: fullResponse });
        updateSessionHistory(activeSessionId, historyContext);
        updateSessionTitle(activeSessionId, text);

    } catch (error) {
        contentDiv.innerHTML = `<span style="color: #ef4444;"><i class="fa-solid fa-circle-exclamation"></i> 请求失败: ${error.message}</span>`;
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

function sendScenario(type) {
    let prompt = '';
    switch (type) {
        case '火灾':
            prompt = '站台发生火灾，我该怎么处理？';
            break;
        case '信号':
            prompt = '信号系统发生“红光带”故障，处置流程是什么？';
            break;
        case '客流':
            prompt = '发生严重拥挤踩踏风险时，值班站长的处置措施？';
            break;
    }
    userInput.value = prompt;
    sendMessage();
}

function loadScenario(type) {
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    event.currentTarget.classList.add('active');
}

loadSessions();


/* LPX — Dark AI Chat
   Preserves Flask endpoints, Ollama, PDFs, chat history and source links. */

const $ = (selector) => document.querySelector(selector);

const searchForm = $("#search-form");
const questionInput = $("#question");
const searchButton = $("#search-button");
const conversation = $("#conversation");
const chatDock = $("#chat-dock");
const chatInput = $("#chat-input");
const chatSend = $("#chat-send");
const hero = $("#hero");
const statusDot = $("#status-dot");
const statusText = $("#status-text");
const newChatButton = $("#new-chat");
const pdfInput = $("#pdf-input");
const useWebToggle = $("#use-web");
const useModelToggle = $("#use-model");

let conversationId = null;
let documentIds = [];
let busy = false;
let historyItems = [];

/* ---------- Safe rendering ---------- */

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}

function cleanAnswer(value) {
  let text = String(value ?? "").trim();

  // Remove source/reference sections appended to an answer.
  text = text.split(
    /^\s*(?:#{1,3}\s*)?(?:sources|references|web sources)\s*:?\s*$/im
  )[0];

  // Remove standalone numbered search-result Markdown links.
  text = text.replace(
    /^\s*(?:[-*]\s*)?\[\d+\]\s+\[[^\]]+\]\(https?:\/\/[^)]*\).*$/gm,
    ""
  );

  // Remove standalone raw URLs.
  text = text.replace(/^\s*https?:\/\/\S+\s*$/gm, "");

  // Remove common headings.
  text = text.replace(
    /^\s*(?:here(?:'s| is) (?:a quick summary|the summary)|answer|response)\s*:?\s*/i,
    ""
  );

  // Remove duplicate paragraphs.
  const seen = new Set();
  const paragraphs = text.split(/\n\s*\n/);
  const unique = [];

  for (const paragraph of paragraphs) {
    const p = paragraph.trim();
    if (!p) continue;

    const key = p.toLowerCase().replace(/\W+/g, " ").trim();
    if (key && !seen.has(key)) {
      seen.add(key);
      unique.push(p);
    }
  }

  return unique.join("\n\n").trim();
}

function renderMarkdown(value) {
  const text = cleanAnswer(value);
  const codeBlocks = [];

  let output = escapeHtml(text).replace(
    /```([a-zA-Z0-9_+-]*)\n?([\s\S]*?)```/g,
    (_, language, code) => {
      const index = codeBlocks.length;
      codeBlocks.push(
        `<pre class="code-block"><div class="code-header"><span>${escapeHtml(language || "code")}</span><button type="button" class="copy-code" data-code-index="${index}">Copy</button></div><code>${code.replace(/\n$/, "")}</code></pre>`
      );
      return `@@CODE_BLOCK_${index}@@`;
    }
  );

  // Render inline code before other Markdown.
  output = output
    .replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>')
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");

  const lines = output.split("\n");
  const formatted = [];
  let listType = null;

  function closeList() {
    if (listType) {
      formatted.push(`</${listType}>`);
      listType = null;
    }
  }

  for (const line of lines) {
    const trimmed = line.trim();
    const unordered = line.match(/^\s*[-*]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+\.\s+(.+)$/);

    if (unordered || ordered) {
      const nextType = unordered ? "ul" : "ol";
      if (listType !== nextType) {
        closeList();
        listType = nextType;
        formatted.push(`<${listType}>`);
      }
      formatted.push(`<li>${(unordered || ordered)[1]}</li>`);
    } else {
      closeList();

      if (!trimmed) continue;

      if (/^@@CODE_BLOCK_\d+@@$/.test(trimmed)) {
        formatted.push(trimmed);
      } else if (/^<(h[1-3]|blockquote)\b/.test(trimmed)) {
        formatted.push(line);
      } else {
        formatted.push(`<p>${line}</p>`);
      }
    }
  }

  closeList();

  output = formatted.join("\n");
  return output.replace(
    /@@CODE_BLOCK_(\d+)@@/g,
    (_, index) => codeBlocks[Number(index)] || ""
  );
}

function modeLabel(mode, model) {
  const modelLabel = model ? ` · ${escapeHtml(model)}` : "";
  if (mode === "ai") return `AI answer · web grounded${modelLabel}`;
  if (mode === "local") return `Local AI${modelLabel}`;
  if (mode === "search") return "Web search summary";
  if (mode === "error") return "Something went wrong";
  return `Answer${modelLabel}`;
}

/* ---------- Messages ---------- */

function addBubble(role, text, options = {}) {
  const { mode, model, sources = [] } = options;
  const bubble = document.createElement("article");
  bubble.className = `chat-bubble ${role}`;

  if (role === "assistant") {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = `
      <strong><span class="assistant-avatar">✳</span> LPX</strong>
      <span class="mode-badge mode-${escapeHtml(mode || "ai")}">${modeLabel(mode, model)}</span>`;
    bubble.appendChild(meta);
  }

  const answer = document.createElement("div");
  answer.className = "answer";

  if (role === "assistant") {
    answer.innerHTML = renderMarkdown(text);
  } else {
    answer.innerHTML = escapeHtml(text).replace(/\n/g, "<br>");
  }

  bubble.appendChild(answer);

  // Keep actual API sources separate from the answer.
  const validSources = Array.isArray(sources) ? sources : [];
  const uniqueSources = [];
  const seenUrls = new Set();

  for (const source of validSources) {
    const url = String(source?.url || "").trim();
    if (!url || !/^https?:\/\//i.test(url) || seenUrls.has(url)) continue;
    seenUrls.add(url);
    uniqueSources.push(source);
  }

  if (role === "assistant" && uniqueSources.length) {
    const sourceSection = document.createElement("details");
    sourceSection.className = "chat-sources";
    sourceSection.open = false;

    const summary = document.createElement("summary");
    summary.textContent = `Sources (${uniqueSources.length})`;
    sourceSection.appendChild(summary);

    uniqueSources.forEach((source, index) => {
      const link = document.createElement("a");
      link.className = "source-link";
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";

      const title = document.createElement("strong");
      title.textContent = `${index + 1}. ${source.title || "Source"}`;

      const snippet = document.createElement("span");
      snippet.textContent = source.snippet || "";

      link.append(title, snippet);
      sourceSection.appendChild(link);
    });

    bubble.appendChild(sourceSection);
  }

  conversation.appendChild(bubble);
  return bubble;
}

function typingBubble() {
  const bubble = document.createElement("article");
  bubble.className = "chat-bubble assistant";
  bubble.innerHTML = `
    <div class="meta">
      <strong><span class="assistant-avatar">✳</span> LPX</strong>
      <span class="mode-badge">Working</span>
    </div>
    <div class="loading">
      <span class="spinner"></span>
      <span>Thinking and searching…</span>
    </div>`;
  conversation.appendChild(bubble);
  bubble.scrollIntoView({ behavior: "smooth", block: "end" });
  return bubble;
}

function enterChatMode() {
  hero?.classList.add("compact");
  document.body.classList.add("chatting");

  const spacer = $("#dock-spacer");
  if (spacer) spacer.hidden = false;
  if (chatDock) chatDock.hidden = false;
}

function showWelcome() {
  conversation.innerHTML = `
    <div class="welcome-card">
      <div class="welcome-icon">✳</div>
      <h2>How can I help you?</h2>
      <p>Ask a question, explore an idea, or upload a PDF to get started.</p>
    </div>`;
}

function setBusy(value) {
  busy = value;
  if (searchButton) searchButton.disabled = value;
  if (chatSend) chatSend.disabled = value;
}

/* ---------- Ask the backend ---------- */

async function ask(message, { first = false } = {}) {
  message = String(message || "").trim();
  if (busy || !message) return;

  setBusy(true);

  if (first) {
    conversation.innerHTML = "";
    conversationId = null;
    documentIds = [];
  }

  enterChatMode();
  addBubble("user", message);

  const pending = typingBubble();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        conversationId,
        useWeb: useWebToggle ? useWebToggle.checked : true,
        useModel: useModelToggle ? useModelToggle.checked : true,
        documentIds,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "The request could not be completed.");
    }

    conversationId = data.conversationId ?? conversationId;
    pending.remove();

    const answerText = data.answer || "No answer was returned.";

    addBubble("assistant", answerText, {
      mode: data.mode,
      model: data.model,
      sources: data.sources,
    });

    await loadHistory();
    markActiveConversation();
  } catch (error) {
    pending.remove();
    addBubble(
      "assistant",
      error.message || "Unable to connect. Please try again.",
      { mode: "error" }
    );
  } finally {
    setBusy(false);
    if (chatInput) chatInput.focus();
  }
}

/* ---------- PDF upload ---------- */

async function uploadPdf(file) {
  if (!file) return;

  const chip = $(".upload-chip");
  if (chip) chip.classList.add("busy");

  const formData = new FormData();
  formData.append("file", file);

  if (conversationId !== null) {
    formData.append("conversationId", conversationId);
  }

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "PDF upload failed.");
    }

    documentIds.push(data.documentId);
    enterChatMode();

    const note = document.createElement("div");
    note.className = "upload-note";
    note.textContent =
      `📎 ${data.filename} attached (${Number(data.chars || 0).toLocaleString()} characters).`;
    conversation.appendChild(note);
  } catch (error) {
    addBubble("assistant", error.message || "PDF upload failed.", {
      mode: "error",
    });
  } finally {
    if (chip) chip.classList.remove("busy");
    if (pdfInput) pdfInput.value = "";
  }
}

/* ---------- Conversation history ---------- */

function createSidebar() {
  if ($("#lpx-sidebar")) return;

  const sidebar = document.createElement("aside");
  sidebar.id = "lpx-sidebar";
  sidebar.className = "lpx-sidebar";

  sidebar.innerHTML = `
    <div class="sidebar-brand">
      <span class="brand-mark">✳</span>
      <span>LPX</span>
    </div>
    <button type="button" class="sidebar-new-chat" id="sidebar-new-chat">＋ New chat</button>
    <div class="sidebar-section-title">YOUR CHATS</div>
    <div id="history-list" class="history-list">
      <p class="history-empty">Your conversations will appear here.</p>
    </div>
    <div class="sidebar-bottom">
      <span class="sidebar-status-dot"></span>
      <span>Private local AI</span>
    </div>`;

  document.body.prepend(sidebar);
  document.body.classList.add("has-lpx-sidebar");

  $("#sidebar-new-chat").addEventListener("click", resetChat);
}

function getConversationTitle(item) {
  return item.title || item.name || item.first_message || item.preview ||
    item.last_message || `Conversation ${item.id ?? item.conversationId ?? ""}`;
}

function getConversationId(item) {
  return item.id ?? item.conversationId ?? item.conversation_id;
}

async function loadHistory() {
  const list = $("#history-list");
  if (!list) return;

  try {
    const response = await fetch("/api/conversations");
    const data = await response.json();
    if (!response.ok) throw new Error("Could not load chat history.");

    historyItems = Array.isArray(data.conversations) ? data.conversations : [];
    list.innerHTML = "";

    if (!historyItems.length) {
      list.innerHTML = '<p class="history-empty">No conversations yet.</p>';
      return;
    }

    historyItems.forEach((item) => {
      const id = getConversationId(item);
      if (id == null) return;

      const row = document.createElement("div");
      row.className = "history-row";
      row.dataset.id = String(id);

      const openButton = document.createElement("button");
      openButton.type = "button";
      openButton.className = "history-open";
      openButton.textContent = getConversationTitle(item);
      openButton.title = openButton.textContent;
      openButton.addEventListener("click", () => openConversation(id));

      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "history-delete";
      deleteButton.textContent = "×";
      deleteButton.title = "Delete conversation";
      deleteButton.setAttribute("aria-label", "Delete conversation");
      deleteButton.addEventListener("click", () => deleteConversation(id));

      row.append(openButton, deleteButton);
      list.appendChild(row);
    });

    markActiveConversation();
  } catch (error) {
    list.innerHTML =
      '<p class="history-empty">Chat history is temporarily unavailable.</p>';
    console.error(error);
  }
}

function markActiveConversation() {
  document.querySelectorAll(".history-row").forEach((row) => {
    row.classList.toggle(
      "active",
      conversationId !== null &&
      row.dataset.id === String(conversationId)
    );
  });
}

async function openConversation(id) {
  if (busy) return;

  try {
    const response = await fetch(
      `/api/conversations/${encodeURIComponent(id)}`
    );
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Could not open this conversation.");
    }

    conversationId = data.conversationId ?? id;
    documentIds = [];
    conversation.innerHTML = "";
    enterChatMode();

    const messages = Array.isArray(data.messages) ? data.messages : [];

    messages.forEach((message) => {
      const role = String(message.role || message.sender || "").toLowerCase();
      const text = message.content ?? message.message ??
        message.text ?? message.answer ?? "";
      const isUser = role === "user" || role === "human";

      addBubble(isUser ? "user" : "assistant", text, {
        mode: message.mode,
        model: message.model,
      });
    });

    if (!messages.length) showWelcome();
    markActiveConversation();
  } catch (error) {
    addBubble(
      "assistant",
      error.message || "Could not open this conversation.",
      { mode: "error" }
    );
  }
}

async function deleteConversation(id) {
  if (busy) return;

  try {
    const response = await fetch(
      `/api/conversations/${encodeURIComponent(id)}`,
      { method: "DELETE" }
    );
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Could not delete conversation.");
    }

    if (String(conversationId) === String(id)) resetChat();
    await loadHistory();
  } catch (error) {
    addBubble(
      "assistant",
      error.message || "Could not delete conversation.",
      { mode: "error" }
    );
  }
}

/* ---------- Reset and Ollama status ---------- */

function resetChat() {
  if (busy) return;

  conversationId = null;
  documentIds = [];
  conversation.innerHTML = "";
  document.body.classList.remove("chatting");
  hero?.classList.remove("compact");

  if (chatDock) chatDock.hidden = true;

  const spacer = $("#dock-spacer");
  if (spacer) spacer.hidden = true;

  showWelcome();
  markActiveConversation();
  questionInput?.focus();
}

async function checkStatus() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const models = Array.isArray(data.ollamaModels) ? data.ollamaModels : [];

    if (data.ollama && models.length > 0) {
      if (statusDot) statusDot.style.background = "#4ade80";
      if (statusText) {
        statusText.textContent = `Local AI ready · ${models[0]}`;
      }
    } else {
      if (statusDot) statusDot.style.background = "#f5b942";
      if (statusText) {
        statusText.textContent = "Local AI offline · Ollama needed";
        statusText.title =
          `Install a model with: ollama pull ${data.recommendedModel || "qwen3:0.6b"}`;
      }
    }
  } catch {
    if (statusDot) statusDot.style.background = "#fb7185";
    if (statusText) statusText.textContent = "Server offline";
  }
}

/* ---------- Event listeners ---------- */

searchForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = questionInput?.value.trim();
  if (message) {
    questionInput.value = "";
    ask(message, { first: true });
  }
});

chatDock?.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = chatInput?.value.trim();
  if (message) {
    chatInput.value = "";
    ask(message);
  }
});

document.querySelectorAll(".prompt-chip").forEach((button) => {
  button.addEventListener("click", () => {
    ask(button.textContent.trim(), { first: true });
  });
});

pdfInput?.addEventListener("change", () => {
  const file = pdfInput.files?.[0];
  if (file) {
    enterChatMode();
    uploadPdf(file);
  }
});

newChatButton?.addEventListener("click", resetChat);

document.addEventListener("click", async (event) => {
  const button = event.target.closest(".copy-code");
  if (!button) return;

  const codeBlock = button.closest(".code-block")?.querySelector("code");
  if (!codeBlock) return;

  try {
    await navigator.clipboard.writeText(codeBlock.textContent);
    button.textContent = "Copied!";
    setTimeout(() => { button.textContent = "Copy"; }, 1200);
  } catch {
    button.textContent = "Copy unavailable";
  }
});

createSidebar();
showWelcome();
loadHistory();
checkStatus();

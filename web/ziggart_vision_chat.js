import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CHAT_GATE_NODE_NAME = "ZiggartGGUFVisionPromptChatGate";
const STYLE_ELEMENT_ID = "ziggart-vision-chat-styles";

const activeNodeControllers = new Map();

function ensureStyles() {
    if (document.getElementById(STYLE_ELEMENT_ID)) {
        return;
    }
    const style = document.createElement("style");
    style.id = STYLE_ELEMENT_ID;
    style.textContent = `
        .zvc-root {
            box-sizing: border-box;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding: 8px;
            background: #141720;
            color: #e6ecf7;
            border: 1px solid #2b3244;
            border-radius: 8px;
            font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            font-size: 12px;
            overflow: hidden;
        }
        .zvc-root *, .zvc-modal * {
            box-sizing: border-box;
        }
        .zvc-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 6px;
            flex-wrap: wrap;
        }
        .zvc-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.01em;
            background: #23293a;
            color: #b7c3dc;
        }
        .zvc-badge.waiting {
            background: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.45);
        }
        .zvc-badge.thinking {
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.45);
        }
        .zvc-badge.approved {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.45);
        }
        .zvc-badge.cancelled {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.45);
        }
        .zvc-tools {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .zvc-select {
            background: #1d2230;
            color: #dce4f5;
            border: 1px solid #353e56;
            border-radius: 5px;
            padding: 2px 6px;
            font-size: 11px;
            max-width: 145px;
        }
        .zvc-label-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11px;
            color: #9aa7c2;
            font-weight: 600;
        }
        .zvc-textarea {
            width: 100%;
            resize: vertical;
            background: #0d1017;
            color: #eef3fc;
            border: 1px solid #30394f;
            border-radius: 6px;
            padding: 7px 8px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 11.5px;
            line-height: 1.4;
            outline: none;
        }
        .zvc-textarea:focus {
            border-color: #60a5fa;
            box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.35);
        }
        .zvc-prompt-box {
            min-height: 82px;
            max-height: 170px;
        }
        .zvc-reply-box {
            min-height: 46px;
            max-height: 96px;
            font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        }
        .zvc-chat-log {
            flex: 1 1 90px;
            min-height: 76px;
            overflow-y: auto;
            background: #0e1119;
            border: 1px solid #262d3f;
            border-radius: 6px;
            padding: 6px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .zvc-empty {
            color: #77839d;
            font-style: italic;
            font-size: 11px;
            padding: 6px 4px;
        }
        .zvc-msg {
            padding: 6px 8px;
            border-radius: 6px;
            border: 1px solid transparent;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 11.5px;
            line-height: 1.38;
        }
        .zvc-msg-head {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 3px;
            opacity: 0.85;
        }
        .zvc-msg.user {
            background: #1d2b44;
            border-color: #324a73;
            color: #e6f0ff;
            align-self: flex-end;
            max-width: 94%;
        }
        .zvc-msg.assistant.initial, .zvc-msg.assistant.refine {
            background: #172522;
            border-color: #294c42;
            color: #e2f7f0;
            align-self: flex-start;
            max-width: 96%;
        }
        .zvc-msg.assistant.discuss {
            background: #231e36;
            border-color: #42356b;
            color: #ece6ff;
            align-self: flex-start;
            max-width: 96%;
        }
        .zvc-msg.system {
            background: #1b202d;
            border-color: #313a4f;
            color: #aebad4;
            font-size: 11px;
            align-self: stretch;
        }
        .zvc-msg.system.error {
            background: rgba(127, 29, 29, 0.35);
            border-color: rgba(248, 113, 113, 0.5);
            color: #fecaca;
        }
        .zvc-error-bar {
            display: none;
            padding: 5px 8px;
            border-radius: 5px;
            font-size: 11px;
            background: rgba(185, 28, 28, 0.25);
            border: 1px solid rgba(248, 113, 113, 0.45);
            color: #fecaca;
        }
        .zvc-error-bar.info {
            background: rgba(59, 130, 246, 0.16);
            border-color: rgba(96, 165, 250, 0.4);
            color: #bfdbfe;
        }
        .zvc-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            align-items: center;
        }
        .zvc-btn {
            border: 1px solid #39435c;
            background: #22293b;
            color: #e6ecf7;
            border-radius: 6px;
            padding: 5px 9px;
            font-size: 11.5px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.12s ease, border-color 0.12s ease, opacity 0.12s ease;
        }
        .zvc-btn:hover:not(:disabled) {
            background: #2d364f;
            border-color: #56658a;
        }
        .zvc-btn:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }
        .zvc-btn.small {
            padding: 3px 7px;
            font-size: 11px;
        }
        .zvc-btn.refine {
            background: #21406b;
            border-color: #3b6fb5;
            color: #eff6ff;
        }
        .zvc-btn.refine:hover:not(:disabled) {
            background: #2b5287;
        }
        .zvc-btn.discuss {
            background: #3b2a66;
            border-color: #6c4ec2;
            color: #f5f3ff;
        }
        .zvc-btn.discuss:hover:not(:disabled) {
            background: #4c3782;
        }
        .zvc-btn.continue {
            background: #1a5c39;
            border-color: #2f9c62;
            color: #ecfdf5;
            margin-left: auto;
        }
        .zvc-btn.continue:hover:not(:disabled) {
            background: #22784b;
        }
        .zvc-btn.cancel {
            background: #5c1f27;
            border-color: #9e3543;
            color: #fff1f2;
        }
        .zvc-btn.cancel:hover:not(:disabled) {
            background: #782732;
        }
        .zvc-btn.stop {
            background: #6b4214;
            border-color: #c27821;
            color: #fef3c7;
        }
        .zvc-btn.stop:hover:not(:disabled) {
            background: #8a561a;
        }
        .zvc-msg.streaming {
            background: #182334;
            border-color: #3b6fb5;
            color: #dbeafe;
            align-self: flex-start;
            max-width: 96%;
        }
        .zvc-diff-box {
            display: none;
            max-height: 140px;
            overflow-y: auto;
            padding: 7px 9px;
            background: #0c1018;
            border: 1px solid #2b354d;
            border-radius: 6px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 11px;
            line-height: 1.45;
            white-space: pre-wrap;
        }
        .zvc-diff-add {
            background: rgba(34, 197, 94, 0.22);
            color: #86efac;
            border-radius: 3px;
            padding: 0 2px;
        }
        .zvc-diff-del {
            background: rgba(239, 68, 68, 0.22);
            color: #fca5a5;
            text-decoration: line-through;
            border-radius: 3px;
            padding: 0 2px;
        }
        .zvc-thumb-stage {
            position: relative;
            display: inline-block;
            max-width: 100%;
            max-height: 204px;
            cursor: crosshair;
            user-select: none;
        }
        .zvc-focus-rect {
            position: absolute;
            border: 2px solid #fbbf24;
            background: rgba(251, 191, 36, 0.18);
            pointer-events: none;
            display: none;
            border-radius: 3px;
        }
        .zvc-focus-chip {
            display: none;
            align-items: center;
            justify-content: space-between;
            gap: 6px;
            padding: 3px 8px;
            border-radius: 5px;
            background: rgba(251, 191, 36, 0.16);
            border: 1px solid rgba(251, 191, 36, 0.45);
            color: #fde68a;
            font-size: 11px;
        }

        /* Expandable Modal Overlay */
        .zvc-modal-backdrop {
            position: fixed;
            inset: 0;
            background: rgba(6, 8, 14, 0.78);
            backdrop-filter: blur(4px);
            z-index: 10050;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .zvc-modal {
            width: min(1120px, 94vw);
            height: min(820px, 90vh);
            background: #141720;
            color: #e6ecf7;
            border: 1px solid #354059;
            border-radius: 12px;
            box-shadow: 0 24px 64px rgba(0, 0, 0, 0.65);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            font-size: 13px;
        }
        .zvc-modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            background: #1a1f2c;
            border-bottom: 1px solid #2c3448;
            gap: 10px;
        }
        .zvc-modal-title {
            font-weight: 700;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .zvc-modal-body {
            flex: 1 1 auto;
            min-height: 0;
            display: grid;
            grid-template-columns: minmax(320px, 1.05fr) minmax(320px, 1fr);
            gap: 14px;
            padding: 14px 16px;
            overflow: hidden;
        }
        @media (max-width: 820px) {
            .zvc-modal-body {
                grid-template-columns: 1fr;
                overflow-y: auto;
            }
        }
        .zvc-modal-col {
            display: flex;
            flex-direction: column;
            gap: 8px;
            min-height: 0;
        }
        .zvc-thumb-wrap {
            display: none;
            align-items: center;
            justify-content: center;
            background: #0b0e15;
            border: 1px solid #262e40;
            border-radius: 8px;
            padding: 6px;
            max-height: 220px;
            overflow: hidden;
        }
        .zvc-thumb-wrap img {
            max-width: 100%;
            max-height: 204px;
            object-fit: contain;
            border-radius: 4px;
        }
        .zvc-modal .zvc-prompt-box {
            flex: 1 1 auto;
            max-height: none;
            min-height: 180px;
            font-size: 12.5px;
        }
        .zvc-modal .zvc-reply-box {
            min-height: 70px;
            max-height: 150px;
            font-size: 12.5px;
        }
        .zvc-modal .zvc-chat-log {
            flex: 1 1 auto;
            min-height: 180px;
        }
    `;
    document.head.appendChild(style);
}

function stopCanvasPropagation(el) {
    for (const evtName of ["mousedown", "pointerdown", "wheel", "keydown", "keyup"]) {
        el.addEventListener(evtName, (e) => {
            e.stopPropagation();
        });
    }
}

function formatBadge(state) {
    const status = state.status || "idle";
    const ver = state.active_version ? `v${state.active_version}/${state.version_count}` : "";
    const turns = state.turn_count ? `${state.turn_count} turn${state.turn_count === 1 ? "" : "s"}` : "";
    const meta = [ver, turns].filter(Boolean).join(" · ");

    if (status === "waiting") {
        return { cls: "waiting", text: `⏸ Waiting for review${meta ? ` (${meta})` : ""}` };
    }
    if (status === "thinking") {
        return { cls: "thinking", text: `⏳ VLM thinking...${meta ? ` (${meta})` : ""}` };
    }
    if (status === "approved") {
        return { cls: "approved", text: `✅ Sent to workflow${meta ? ` (${meta})` : ""}` };
    }
    if (status === "cancelled") {
        return { cls: "cancelled", text: "✖ Cancelled" };
    }
    return { cls: "idle", text: "Idle · Queue workflow to review" };
}

function renderWordDiffInto(container, prevText, currText) {
    container.innerHTML = "";
    const a = String(prevText || "").split(/(\s+)/);
    const b = String(currText || "").split(/(\s+)/);
    const m = a.length;
    const n = b.length;
    if (m * n > 250000) {
        container.textContent = currText;
        return;
    }
    const dp = Array.from({ length: m + 1 }, () => new Uint16Array(n + 1));
    for (let i = m - 1; i >= 0; i--) {
        for (let j = n - 1; j >= 0; j--) {
            dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
        }
    }
    let i = 0;
    let j = 0;
    while (i < m && j < n) {
        if (a[i] === b[j]) {
            container.appendChild(document.createTextNode(a[i]));
            i++;
            j++;
        } else if (dp[i + 1][j] >= dp[i][j + 1]) {
            const sp = document.createElement("span");
            sp.className = "zvc-diff-del";
            sp.textContent = a[i];
            container.appendChild(sp);
            i++;
        } else {
            const sp = document.createElement("span");
            sp.className = "zvc-diff-add";
            sp.textContent = b[j];
            container.appendChild(sp);
            j++;
        }
    }
    while (i < m) {
        const sp = document.createElement("span");
        sp.className = "zvc-diff-del";
        sp.textContent = a[i++];
        container.appendChild(sp);
    }
    while (j < n) {
        const sp = document.createElement("span");
        sp.className = "zvc-diff-add";
        sp.textContent = b[j++];
        container.appendChild(sp);
    }
}

function renderChatLogInto(container, chatLog, streamingState = null) {
    container.innerHTML = "";
    const hasLog = Array.isArray(chatLog) && chatLog.length > 0;
    const isStreaming = Boolean(streamingState?.active);
    if (!hasLog && !isStreaming) {
        const empty = document.createElement("div");
        empty.className = "zvc-empty";
        empty.textContent =
            "Queue the workflow to inspect the Vision prompt here. You can ask questions, request prompt refinements, edit manually, or continue.";
        container.appendChild(empty);
        return;
    }

    if (hasLog) {
        for (const item of chatLog) {
            const role = item.role || "system";
            const mode = item.mode || "";
            const div = document.createElement("div");
            div.className = `zvc-msg ${role} ${mode}`.trim();

            const head = document.createElement("div");
            head.className = "zvc-msg-head";
            let label = role === "user" ? "You" : role === "assistant" ? "VLM" : "System";
            if (mode === "initial") label += " · Initial Prompt";
            else if (mode === "refine") label += role === "user" ? " · Refine Request" : " · Updated Prompt";
            else if (mode === "discuss") label += role === "user" ? " · Question" : " · Answer";
            else if (mode === "undo") label += " · History";
            else if (mode === "stop") label += " · Stopped";
            else if (mode === "error") label += " · Error";
            if (item.version) label += ` (v${item.version})`;
            head.textContent = label;

            const body = document.createElement("div");
            body.textContent = item.content || "";

            div.appendChild(head);
            div.appendChild(body);
            container.appendChild(div);
        }
    }

    if (isStreaming) {
        const sDiv = document.createElement("div");
        sDiv.className = "zvc-msg streaming";
        sDiv.dataset.zvcStreaming = "1";
        const sHead = document.createElement("div");
        sHead.className = "zvc-msg-head";
        sHead.textContent = streamingState.mode === "discuss" ? "VLM · Streaming Answer..." : "VLM · Streaming Prompt...";
        const sBody = document.createElement("div");
        sBody.className = "zvc-streaming-body";
        sBody.textContent = streamingState.text || "…";
        sDiv.append(sHead, sBody);
        container.appendChild(sDiv);
    }

    container.scrollTop = container.scrollHeight;
}

function populateVersionSelect(selectEl, state) {
    selectEl.innerHTML = "";
    const versions = Array.isArray(state.versions) ? state.versions : [];
    if (versions.length === 0) {
        const opt = document.createElement("option");
        opt.value = "0";
        opt.textContent = "No versions";
        selectEl.appendChild(opt);
        selectEl.disabled = true;
        return;
    }
    for (const v of versions) {
        const opt = document.createElement("option");
        opt.value = String(v.index);
        opt.textContent = `v${v.version} · ${v.note || v.source}`;
        selectEl.appendChild(opt);
    }
    selectEl.value = String(state.active_version_index ?? versions.length - 1);
    selectEl.disabled = state.status !== "waiting" || versions.length <= 1;
}

function createVisionChatController(node) {
    ensureStyles();

    const state = {
        node_id: String(node.id ?? ""),
        session_token: "",
        status: "idle",
        error: "",
        current_prompt: "",
        active_version_index: 0,
        active_version: 0,
        version_count: 0,
        can_undo: false,
        versions: [],
        chat_log: [],
        thumbnail_uri: "",
        can_reply: false,
        turn_count: 0,
    };

    let localPromptDirty = false;
    let draftReply = "";
    let modalRefs = null;
    let escHandler = null;
    let showDiff = false;
    let focusBox = null; // [x0, y0, x1, y1] normalized 0..1
    const streamingState = { active: false, mode: "", text: "" };

    // Build inline DOM panel
    const root = document.createElement("div");
    root.className = "zvc-root";
    stopCanvasPropagation(root);

    const header = document.createElement("div");
    header.className = "zvc-header";

    const badgeEl = document.createElement("span");
    badgeEl.className = "zvc-badge";
    badgeEl.textContent = "Idle · Queue workflow to review";

    const tools = document.createElement("div");
    tools.className = "zvc-tools";

    const versionSelect = document.createElement("select");
    versionSelect.className = "zvc-select";
    versionSelect.title = "Switch between prompt versions";

    const undoBtn = document.createElement("button");
    undoBtn.type = "button";
    undoBtn.className = "zvc-btn small";
    undoBtn.textContent = "↩ Undo";
    undoBtn.title = "Revert to previous prompt version";

    const resetCtxBtn = document.createElement("button");
    resetCtxBtn.type = "button";
    resetCtxBtn.className = "zvc-btn small";
    resetCtxBtn.textContent = "🧹 Reset";
    resetCtxBtn.title = "Reset chat context back to original image + active prompt";

    const expandBtn = document.createElement("button");
    expandBtn.type = "button";
    expandBtn.className = "zvc-btn small";
    expandBtn.textContent = "⤢ Expand";
    expandBtn.title = "Open large review & chat modal";

    tools.append(versionSelect, undoBtn, resetCtxBtn, expandBtn);
    header.append(badgeEl, tools);

    const promptLabel = document.createElement("div");
    promptLabel.className = "zvc-label-row";
    promptLabel.innerHTML = "<span>Current Prompt (editable)</span><span></span>";

    const promptBox = document.createElement("textarea");
    promptBox.className = "zvc-textarea zvc-prompt-box";
    promptBox.placeholder = "Generated vision prompt will appear here when paused...";

    const errorBar = document.createElement("div");
    errorBar.className = "zvc-error-bar";

    const chatLabel = document.createElement("div");
    chatLabel.className = "zvc-label-row";
    chatLabel.textContent = "Discussion & Refinement Log";

    const chatLogEl = document.createElement("div");
    chatLogEl.className = "zvc-chat-log";

    const replyBox = document.createElement("textarea");
    replyBox.className = "zvc-textarea zvc-reply-box";
    replyBox.placeholder =
        "Reply to VLM (e.g. 'Look closer at the background — mention the red neon sign and silver rings')... Ctrl+Enter to Refine";

    const actionsRow = document.createElement("div");
    actionsRow.className = "zvc-actions";

    const refineBtn = document.createElement("button");
    refineBtn.type = "button";
    refineBtn.className = "zvc-btn refine";
    refineBtn.textContent = "✨ Refine Prompt";
    refineBtn.title = "Send feedback to the VLM and replace the current prompt with its updated version (Ctrl+Enter)";

    const discussBtn = document.createElement("button");
    discussBtn.type = "button";
    discussBtn.className = "zvc-btn discuss";
    discussBtn.textContent = "💬 Ask / Discuss";
    discussBtn.title = "Ask the VLM a question about the image without overwriting the current prompt (Ctrl+Shift+Enter)";

    const stopBtn = document.createElement("button");
    stopBtn.type = "button";
    stopBtn.className = "zvc-btn stop";
    stopBtn.textContent = "⏹ Stop";
    stopBtn.title = "Stop the active VLM generation without aborting the workflow";

    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "zvc-btn cancel";
    cancelBtn.textContent = "✖ Cancel";
    cancelBtn.title = "Abort the workflow run";

    const continueBtn = document.createElement("button");
    continueBtn.type = "button";
    continueBtn.className = "zvc-btn continue";
    continueBtn.textContent = "▶ Continue Workflow";
    continueBtn.title = "Approve the current prompt and resume downstream nodes";

    actionsRow.append(refineBtn, discussBtn, stopBtn, cancelBtn, continueBtn);
    root.append(header, promptLabel, promptBox, errorBar, chatLabel, chatLogEl, replyBox, actionsRow);

    async function sendAction(action, extra = {}) {
        const nodeId = String(node.id ?? state.node_id ?? "");
        const body = {
            node_id: nodeId,
            session_token: state.session_token || "",
            action,
            ...extra,
        };
        try {
            const res = await api.fetchApi("/ziggart/vision_chat/action", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const data = await res.json();
            if (data?.state) {
                applyServerState(data.state, { forcePromptSync: action !== "update_prompt" });
            }
            if (data && data.ok === false && data.error) {
                showError(data.error, false);
            }
            return data;
        } catch (err) {
            showError(`Request failed: ${err?.message || err}`, false);
            return null;
        }
    }

    function showError(msg, isInfo = false) {
        const targets = [errorBar];
        if (modalRefs?.errorBar) targets.push(modalRefs.errorBar);
        for (const el of targets) {
            if (!msg) {
                el.style.display = "none";
                el.textContent = "";
                el.classList.remove("info");
            } else {
                el.style.display = "block";
                el.textContent = msg;
                el.classList.toggle("info", Boolean(isInfo));
            }
        }
    }

    function syncPromptFromInput(value, originEl) {
        localPromptDirty = true;
        state.current_prompt = value;
        if (originEl !== promptBox) promptBox.value = value;
        if (modalRefs && originEl !== modalRefs.promptBox) {
            modalRefs.promptBox.value = value;
        }
    }

    function syncReplyFromInput(value, originEl) {
        draftReply = value;
        if (originEl !== replyBox) replyBox.value = value;
        if (modalRefs && originEl !== modalRefs.replyBox) {
            modalRefs.replyBox.value = value;
        }
    }

    async function commitManualEditIfDirty() {
        if (!localPromptDirty || state.status !== "waiting") return;
        localPromptDirty = false;
        await sendAction("update_prompt", { prompt: promptBox.value });
    }

    function updateFocusOverlay() {
        if (!modalRefs?.focusRect || !modalRefs?.focusChip) return;
        if (!focusBox) {
            modalRefs.focusRect.style.display = "none";
            modalRefs.focusChip.style.display = "none";
            return;
        }
        const [x0, y0, x1, y1] = focusBox;
        modalRefs.focusRect.style.display = "block";
        modalRefs.focusRect.style.left = `${(x0 * 100).toFixed(2)}%`;
        modalRefs.focusRect.style.top = `${(y0 * 100).toFixed(2)}%`;
        modalRefs.focusRect.style.width = `${((x1 - x0) * 100).toFixed(2)}%`;
        modalRefs.focusRect.style.height = `${((y1 - y0) * 100).toFixed(2)}%`;
        modalRefs.focusChip.style.display = "flex";
        modalRefs.focusChipText.textContent = `🎯 Region pinned [${x0.toFixed(2)}, ${y0.toFixed(2)} → ${x1.toFixed(2)}, ${y1.toFixed(2)}]`;
    }

    async function handleReply(mode) {
        const msg = draftReply.trim();
        if (!msg) {
            showError("Type a message or correction for the VLM first.", false);
            return;
        }
        const currentPromptVal = promptBox.value;
        localPromptDirty = false;
        const extraPayload = {
            message: msg,
            prompt: currentPromptVal,
        };
        if (focusBox) {
            extraPayload.focus_box = focusBox;
        }
        const result = await sendAction(mode, extraPayload);
        if (result?.ok) {
            syncReplyFromInput("", null);
            focusBox = null;
            updateFocusOverlay();
        }
    }

    async function handleStop() {
        await sendAction("stop");
    }

    async function handleResetContext() {
        localPromptDirty = false;
        await sendAction("reset_context");
    }

    async function handleContinue() {
        const currentPromptVal = promptBox.value;
        localPromptDirty = false;
        const result = await sendAction("continue", { prompt: currentPromptVal });
        if (result?.ok) {
            closeModal();
        }
    }

    async function handleCancel() {
        localPromptDirty = false;
        const result = await sendAction("cancel");
        if (result?.ok) {
            closeModal();
        }
    }

    function bindInputHandlers(pBox, rBox, vSel, uBtn, rstBtn, refBtn, discBtn, stpBtn, contBtn, cancBtn) {
        pBox.addEventListener("input", () => syncPromptFromInput(pBox.value, pBox));
        pBox.addEventListener("blur", () => {
            commitManualEditIfDirty();
        });
        rBox.addEventListener("input", () => syncReplyFromInput(rBox.value, rBox));
        rBox.addEventListener("keydown", (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                e.preventDefault();
                handleReply(e.shiftKey ? "discuss" : "refine");
            }
        });
        vSel.addEventListener("change", async () => {
            localPromptDirty = false;
            await sendAction("restore_version", { version_index: Number(vSel.value) });
        });
        uBtn.addEventListener("click", async () => {
            localPromptDirty = false;
            await sendAction("undo");
        });
        rstBtn.addEventListener("click", () => handleResetContext());
        refBtn.addEventListener("click", () => handleReply("refine"));
        discBtn.addEventListener("click", () => handleReply("discuss"));
        stpBtn.addEventListener("click", () => handleStop());
        contBtn.addEventListener("click", () => handleContinue());
        cancBtn.addEventListener("click", () => handleCancel());
    }

    bindInputHandlers(
        promptBox,
        replyBox,
        versionSelect,
        undoBtn,
        resetCtxBtn,
        refineBtn,
        discussBtn,
        stopBtn,
        continueBtn,
        cancelBtn
    );

    function renderTarget(refs) {
        const badge = formatBadge(state);
        refs.badgeEl.className = `zvc-badge ${badge.cls}`;
        refs.badgeEl.textContent = badge.text;

        const isWaiting = state.status === "waiting";
        const isThinking = state.status === "thinking";
        const isActive = isWaiting || isThinking;
        const canReplyNow = isWaiting && Boolean(state.can_reply);

        refs.promptBox.disabled = !isWaiting;
        refs.replyBox.disabled = !canReplyNow;
        refs.refineBtn.disabled = !canReplyNow;
        refs.discussBtn.disabled = !canReplyNow;
        refs.stopBtn.disabled = !isThinking;
        refs.resetCtxBtn.disabled = !isWaiting;
        refs.continueBtn.disabled = !isWaiting;
        refs.cancelBtn.disabled = !isActive;
        refs.undoBtn.disabled = !isWaiting || !state.can_undo;

        populateVersionSelect(refs.versionSelect, state);
        renderChatLogInto(refs.chatLogEl, state.chat_log, streamingState);

        if (refs.diffBox) {
            const versions = Array.isArray(state.versions) ? state.versions : [];
            const activeIdx = Number(state.active_version_index ?? 0);
            if (showDiff && activeIdx > 0 && versions[activeIdx - 1]?.prompt !== undefined) {
                refs.diffBox.style.display = "block";
                renderWordDiffInto(
                    refs.diffBox,
                    versions[activeIdx - 1].prompt || "",
                    state.current_prompt || ""
                );
            } else {
                refs.diffBox.style.display = "none";
            }
        }

        if (refs.thumbWrap && refs.thumbImg) {
            if (state.thumbnail_uri) {
                refs.thumbImg.src = state.thumbnail_uri;
                refs.thumbWrap.style.display = "flex";
            } else {
                refs.thumbWrap.style.display = "none";
            }
        }
    }

    function renderAll(forcePromptSync = false) {
        if (forcePromptSync || !localPromptDirty) {
            promptBox.value = state.current_prompt || "";
            if (modalRefs) modalRefs.promptBox.value = state.current_prompt || "";
        }
        replyBox.value = draftReply;
        if (modalRefs) modalRefs.replyBox.value = draftReply;

        renderTarget({
            badgeEl,
            promptBox,
            replyBox,
            versionSelect,
            undoBtn,
            resetCtxBtn,
            refineBtn,
            discussBtn,
            stopBtn,
            continueBtn,
            cancelBtn,
            chatLogEl,
        });
        if (modalRefs) {
            renderTarget(modalRefs);
            updateFocusOverlay();
        }

        if (state.error) {
            showError(state.error, false);
        } else if (state.status === "waiting" && !state.can_reply) {
            showError(
                "Manual review mode: connect vision_session from GGUF Vision Prompt (or a fallback VLM model) to enable Refine / Discuss replies.",
                true
            );
        } else {
            showError("", false);
        }
    }

    function applyServerDelta(deltaPayload) {
        if (!deltaPayload || typeof deltaPayload !== "object") return;
        if (deltaPayload.session_token && state.session_token && deltaPayload.session_token !== state.session_token) {
            return;
        }
        streamingState.active = true;
        streamingState.mode = deltaPayload.mode || "refine";
        streamingState.text = String(deltaPayload.partial_text || "");
        for (const el of [chatLogEl, modalRefs?.chatLogEl]) {
            if (!el) continue;
            let sBody = el.querySelector("[data-zvc-streaming='1'] .zvc-streaming-body");
            if (!sBody) {
                renderChatLogInto(el, state.chat_log, streamingState);
            } else {
                sBody.textContent = streamingState.text;
                el.scrollTop = el.scrollHeight;
            }
        }
    }

    function applyServerState(incoming, opts = {}) {
        if (!incoming || typeof incoming !== "object") return;
        const tokenChanged =
            incoming.session_token && incoming.session_token !== state.session_token;
        Object.assign(state, incoming);
        if (state.status !== "thinking") {
            streamingState.active = false;
            streamingState.text = "";
        }
        if (tokenChanged || opts.forcePromptSync) {
            localPromptDirty = false;
        }
        if (state.status === "approved" || state.status === "cancelled") {
            focusBox = null;
            closeModal();
        }
        renderAll(Boolean(tokenChanged || opts.forcePromptSync));
        app.graph?.setDirtyCanvas?.(true, false);
    }

    function closeModal() {
        if (!modalRefs) return;
        modalRefs.cleanupThumbDrag?.();
        if (escHandler) {
            window.removeEventListener("keydown", escHandler);
            escHandler = null;
        }
        modalRefs.backdrop.remove();
        modalRefs = null;
    }

    function openModal() {
        if (modalRefs) return;

        const backdrop = document.createElement("div");
        backdrop.className = "zvc-modal-backdrop";
        backdrop.addEventListener("click", (e) => {
            if (e.target === backdrop) closeModal();
        });

        const modal = document.createElement("div");
        modal.className = "zvc-modal";
        stopCanvasPropagation(modal);

        const mHeader = document.createElement("div");
        mHeader.className = "zvc-modal-header";

        const mTitle = document.createElement("div");
        mTitle.className = "zvc-modal-title";
        const mTitleText = document.createElement("span");
        mTitleText.textContent = "GGUF Vision Prompt Review & Chat";
        const mBadge = document.createElement("span");
        mBadge.className = "zvc-badge";
        mTitle.append(mTitleText, mBadge);

        const mTools = document.createElement("div");
        mTools.className = "zvc-tools";
        const mVersionSel = document.createElement("select");
        mVersionSel.className = "zvc-select";
        const mUndoBtn = document.createElement("button");
        mUndoBtn.type = "button";
        mUndoBtn.className = "zvc-btn small";
        mUndoBtn.textContent = "↩ Undo";
        const mResetBtn = document.createElement("button");
        mResetBtn.type = "button";
        mResetBtn.className = "zvc-btn small";
        mResetBtn.textContent = "🧹 Reset";
        mResetBtn.title = "Reset chat context to original image + active prompt";
        const mDiffBtn = document.createElement("button");
        mDiffBtn.type = "button";
        mDiffBtn.className = "zvc-btn small";
        mDiffBtn.textContent = "📊 Diff";
        mDiffBtn.title = "Toggle word-level diff against previous prompt version";
        mDiffBtn.addEventListener("click", () => {
            showDiff = !showDiff;
            renderAll(false);
        });
        const mCloseBtn = document.createElement("button");
        mCloseBtn.type = "button";
        mCloseBtn.className = "zvc-btn small";
        mCloseBtn.textContent = "✕ Collapse";
        mCloseBtn.addEventListener("click", closeModal);
        mTools.append(mVersionSel, mUndoBtn, mResetBtn, mDiffBtn, mCloseBtn);
        mHeader.append(mTitle, mTools);

        const mBody = document.createElement("div");
        mBody.className = "zvc-modal-body";

        // Left column: thumbnail (with region-pinning drag) + editable prompt + diff
        const leftCol = document.createElement("div");
        leftCol.className = "zvc-modal-col";

        const thumbWrap = document.createElement("div");
        thumbWrap.className = "zvc-thumb-wrap";
        const thumbStage = document.createElement("div");
        thumbStage.className = "zvc-thumb-stage";
        thumbStage.title = "Drag a box on the preview to pin a focus region for the next Refine / Discuss pass";
        const thumbImg = document.createElement("img");
        thumbImg.alt = "Vision input preview";
        thumbImg.draggable = false;
        const focusRect = document.createElement("div");
        focusRect.className = "zvc-focus-rect";
        thumbStage.append(thumbImg, focusRect);
        thumbWrap.appendChild(thumbStage);

        const focusChip = document.createElement("div");
        focusChip.className = "zvc-focus-chip";
        const focusChipText = document.createElement("span");
        const focusClearBtn = document.createElement("button");
        focusClearBtn.type = "button";
        focusClearBtn.className = "zvc-btn small";
        focusClearBtn.textContent = "Clear Region";
        focusClearBtn.addEventListener("click", () => {
            focusBox = null;
            updateFocusOverlay();
        });
        focusChip.append(focusChipText, focusClearBtn);

        let dragStart = null;
        thumbStage.addEventListener("mousedown", (e) => {
            if (!state.has_source_image || e.button !== 0) return;
            const rect = thumbImg.getBoundingClientRect();
            if (rect.width <= 4 || rect.height <= 4) return;
            const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const ny = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
            dragStart = { nx, ny, rect };
            focusBox = [nx, ny, nx, ny];
            updateFocusOverlay();
        });
        const onThumbMove = (e) => {
            if (!dragStart) return;
            const { nx, ny, rect } = dragStart;
            const cx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const cy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
            focusBox = [Math.min(nx, cx), Math.min(ny, cy), Math.max(nx, cx), Math.max(ny, cy)];
            updateFocusOverlay();
        };
        const onThumbUp = () => {
            if (!dragStart) return;
            dragStart = null;
            if (focusBox && ((focusBox[2] - focusBox[0]) < 0.02 || (focusBox[3] - focusBox[1]) < 0.02)) {
                focusBox = null;
                updateFocusOverlay();
            }
        };
        window.addEventListener("mousemove", onThumbMove);
        window.addEventListener("mouseup", onThumbUp);

        const mPromptLabel = document.createElement("div");
        mPromptLabel.className = "zvc-label-row";
        mPromptLabel.textContent = "Current Prompt (editable)";

        const mPromptBox = document.createElement("textarea");
        mPromptBox.className = "zvc-textarea zvc-prompt-box";
        mPromptBox.placeholder = "Generated vision prompt...";

        const mDiffBox = document.createElement("div");
        mDiffBox.className = "zvc-diff-box";

        const mErrorBar = document.createElement("div");
        mErrorBar.className = "zvc-error-bar";

        leftCol.append(thumbWrap, focusChip, mPromptLabel, mPromptBox, mDiffBox, mErrorBar);

        // Right column: chat log + reply controls
        const rightCol = document.createElement("div");
        rightCol.className = "zvc-modal-col";

        const mChatLabel = document.createElement("div");
        mChatLabel.className = "zvc-label-row";
        mChatLabel.textContent = "Discussion & Refinement History";

        const mChatLog = document.createElement("div");
        mChatLog.className = "zvc-chat-log";

        const mReplyBox = document.createElement("textarea");
        mReplyBox.className = "zvc-textarea zvc-reply-box";
        mReplyBox.placeholder =
            "Explain missed details or ask the VLM a question... (Ctrl+Enter = Refine, Ctrl+Shift+Enter = Ask)";

        const mActions = document.createElement("div");
        mActions.className = "zvc-actions";

        const mRefineBtn = document.createElement("button");
        mRefineBtn.type = "button";
        mRefineBtn.className = "zvc-btn refine";
        mRefineBtn.textContent = "✨ Refine Prompt";

        const mDiscussBtn = document.createElement("button");
        mDiscussBtn.type = "button";
        mDiscussBtn.className = "zvc-btn discuss";
        mDiscussBtn.textContent = "💬 Ask / Discuss";

        const mStopBtn = document.createElement("button");
        mStopBtn.type = "button";
        mStopBtn.className = "zvc-btn stop";
        mStopBtn.textContent = "⏹ Stop";

        const mCancelBtn = document.createElement("button");
        mCancelBtn.type = "button";
        mCancelBtn.className = "zvc-btn cancel";
        mCancelBtn.textContent = "✖ Cancel";

        const mContinueBtn = document.createElement("button");
        mContinueBtn.type = "button";
        mContinueBtn.className = "zvc-btn continue";
        mContinueBtn.textContent = "▶ Continue Workflow";

        mActions.append(mRefineBtn, mDiscussBtn, mStopBtn, mCancelBtn, mContinueBtn);
        rightCol.append(mChatLabel, mChatLog, mReplyBox, mActions);

        mBody.append(leftCol, rightCol);
        modal.append(mHeader, mBody);
        backdrop.appendChild(modal);
        document.body.appendChild(backdrop);

        modalRefs = {
            backdrop,
            badgeEl: mBadge,
            promptBox: mPromptBox,
            replyBox: mReplyBox,
            versionSelect: mVersionSel,
            undoBtn: mUndoBtn,
            resetCtxBtn: mResetBtn,
            refineBtn: mRefineBtn,
            discussBtn: mDiscussBtn,
            stopBtn: mStopBtn,
            continueBtn: mContinueBtn,
            cancelBtn: mCancelBtn,
            chatLogEl: mChatLog,
            diffBox: mDiffBox,
            errorBar: mErrorBar,
            thumbWrap,
            thumbImg,
            focusRect,
            focusChip,
            focusChipText,
            cleanupThumbDrag: () => {
                window.removeEventListener("mousemove", onThumbMove);
                window.removeEventListener("mouseup", onThumbUp);
            },
        };

        bindInputHandlers(
            mPromptBox,
            mReplyBox,
            mVersionSel,
            mUndoBtn,
            mResetBtn,
            mRefineBtn,
            mDiscussBtn,
            mStopBtn,
            mContinueBtn,
            mCancelBtn
        );

        escHandler = (e) => {
            if (e.key === "Escape") {
                closeModal();
            }
        };
        window.addEventListener("keydown", escHandler);

        renderAll(true);
    }

    expandBtn.addEventListener("click", openModal);

    async function hydrateFromServer() {
        const nodeId = String(node.id ?? "");
        if (!nodeId || nodeId === "-1") return;
        try {
            const res = await api.fetchApi(
                `/ziggart/vision_chat/state?node_id=${encodeURIComponent(nodeId)}`
            );
            if (!res.ok) return;
            const data = await res.json();
            if (data?.ok && data.state) {
                applyServerState(data.state, { forcePromptSync: true });
            }
        } catch (_err) {
            // Ignore transient fetch errors during startup.
        }
    }

    renderAll(true);

    return {
        root,
        applyServerState,
        applyServerDelta,
        hydrateFromServer,
        destroy() {
            closeModal();
        },
    };
}

api.addEventListener("ziggart-vision-chat-state", (event) => {
    const detail = event?.detail;
    if (!detail || detail.node_id === undefined) return;
    const key = String(detail.node_id);
    const ctrl = activeNodeControllers.get(key);
    if (ctrl) {
        ctrl.applyServerState(detail);
    }
});

api.addEventListener("ziggart-vision-chat-delta", (event) => {
    const detail = event?.detail;
    if (!detail || detail.node_id === undefined) return;
    const key = String(detail.node_id);
    const ctrl = activeNodeControllers.get(key);
    if (ctrl) {
        ctrl.applyServerDelta(detail);
    }
});

app.registerExtension({
    name: "Ziggart.VisionChatReview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== CHAT_GATE_NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            const controller = createVisionChatController(node);
            node._ziggartVisionChatController = controller;

            if (typeof node.addDOMWidget === "function") {
                node.addDOMWidget("vision_chat_panel", "ziggart_vision_chat", controller.root, {
                    serialize: false,
                    getMinHeight: () => 390,
                });
            }

            if (!node.size || node.size[0] < 460 || node.size[1] < 640) {
                node.setSize?.([Math.max(460, node.size?.[0] || 460), Math.max(660, node.size?.[1] || 660)]);
            }

            const registerId = () => {
                if (node.id !== undefined && node.id !== null && String(node.id) !== "-1") {
                    activeNodeControllers.set(String(node.id), controller);
                    controller.hydrateFromServer();
                }
            };
            setTimeout(registerId, 50);

            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (this._ziggartVisionChatController && this.id !== undefined && this.id !== null) {
                activeNodeControllers.set(String(this.id), this._ziggartVisionChatController);
                this._ziggartVisionChatController.hydrateFromServer();
            }
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this.id !== undefined && this.id !== null) {
                const existing = activeNodeControllers.get(String(this.id));
                if (existing === this._ziggartVisionChatController) {
                    activeNodeControllers.delete(String(this.id));
                }
            }
            this._ziggartVisionChatController?.destroy?.();
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };
    },
});

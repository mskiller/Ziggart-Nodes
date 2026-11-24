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

function renderChatLogInto(container, chatLog) {
    container.innerHTML = "";
    if (!Array.isArray(chatLog) || chatLog.length === 0) {
        const empty = document.createElement("div");
        empty.className = "zvc-empty";
        empty.textContent =
            "Queue the workflow to inspect the Vision prompt here. You can ask questions, request prompt refinements, edit manually, or continue.";
        container.appendChild(empty);
        return;
    }

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
        else if (mode === "error") label += " · Error";
        if (item.version) label += ` (v${item.version})`;
        head.textContent = label;

        const body = document.createElement("div");
        body.textContent = item.content || "";

        div.appendChild(head);
        div.appendChild(body);
        container.appendChild(div);
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

    const expandBtn = document.createElement("button");
    expandBtn.type = "button";
    expandBtn.className = "zvc-btn small";
    expandBtn.textContent = "⤢ Expand";
    expandBtn.title = "Open large review & chat modal";

    tools.append(versionSelect, undoBtn, expandBtn);
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

    actionsRow.append(refineBtn, discussBtn, cancelBtn, continueBtn);
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

    async function handleReply(mode) {
        const msg = draftReply.trim();
        if (!msg) {
            showError("Type a message or correction for the VLM first.", false);
            return;
        }
        const currentPromptVal = promptBox.value;
        localPromptDirty = false;
        const result = await sendAction(mode, {
            message: msg,
            prompt: currentPromptVal,
        });
        if (result?.ok) {
            syncReplyFromInput("", null);
        }
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

    function bindInputHandlers(pBox, rBox, vSel, uBtn, refBtn, discBtn, contBtn, cancBtn) {
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
        refBtn.addEventListener("click", () => handleReply("refine"));
        discBtn.addEventListener("click", () => handleReply("discuss"));
        contBtn.addEventListener("click", () => handleContinue());
        cancBtn.addEventListener("click", () => handleCancel());
    }

    bindInputHandlers(
        promptBox,
        replyBox,
        versionSelect,
        undoBtn,
        refineBtn,
        discussBtn,
        continueBtn,
        cancelBtn
    );

    function renderTarget(refs) {
        const badge = formatBadge(state);
        refs.badgeEl.className = `zvc-badge ${badge.cls}`;
        refs.badgeEl.textContent = badge.text;

        const isWaiting = state.status === "waiting";
        const isActive = isWaiting || state.status === "thinking";
        const canReplyNow = isWaiting && Boolean(state.can_reply);

        refs.promptBox.disabled = !isWaiting;
        refs.replyBox.disabled = !canReplyNow;
        refs.refineBtn.disabled = !canReplyNow;
        refs.discussBtn.disabled = !canReplyNow;
        refs.continueBtn.disabled = !isWaiting;
        refs.cancelBtn.disabled = !isActive;
        refs.undoBtn.disabled = !isWaiting || !state.can_undo;

        populateVersionSelect(refs.versionSelect, state);
        renderChatLogInto(refs.chatLogEl, state.chat_log);

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
            refineBtn,
            discussBtn,
            continueBtn,
            cancelBtn,
            chatLogEl,
        });
        if (modalRefs) {
            renderTarget(modalRefs);
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

    function applyServerState(incoming, opts = {}) {
        if (!incoming || typeof incoming !== "object") return;
        const tokenChanged =
            incoming.session_token && incoming.session_token !== state.session_token;
        Object.assign(state, incoming);
        if (tokenChanged || opts.forcePromptSync) {
            localPromptDirty = false;
        }
        if (state.status === "approved" || state.status === "cancelled") {
            closeModal();
        }
        renderAll(Boolean(tokenChanged || opts.forcePromptSync));
        app.graph?.setDirtyCanvas?.(true, false);
    }

    function closeModal() {
        if (!modalRefs) return;
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
        const mCloseBtn = document.createElement("button");
        mCloseBtn.type = "button";
        mCloseBtn.className = "zvc-btn small";
        mCloseBtn.textContent = "✕ Collapse";
        mCloseBtn.addEventListener("click", closeModal);
        mTools.append(mVersionSel, mUndoBtn, mCloseBtn);
        mHeader.append(mTitle, mTools);

        const mBody = document.createElement("div");
        mBody.className = "zvc-modal-body";

        // Left column: thumbnail + editable prompt
        const leftCol = document.createElement("div");
        leftCol.className = "zvc-modal-col";

        const thumbWrap = document.createElement("div");
        thumbWrap.className = "zvc-thumb-wrap";
        const thumbImg = document.createElement("img");
        thumbImg.alt = "Vision input preview";
        thumbWrap.appendChild(thumbImg);

        const mPromptLabel = document.createElement("div");
        mPromptLabel.className = "zvc-label-row";
        mPromptLabel.textContent = "Current Prompt (editable)";

        const mPromptBox = document.createElement("textarea");
        mPromptBox.className = "zvc-textarea zvc-prompt-box";
        mPromptBox.placeholder = "Generated vision prompt...";

        const mErrorBar = document.createElement("div");
        mErrorBar.className = "zvc-error-bar";

        leftCol.append(thumbWrap, mPromptLabel, mPromptBox, mErrorBar);

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

        const mCancelBtn = document.createElement("button");
        mCancelBtn.type = "button";
        mCancelBtn.className = "zvc-btn cancel";
        mCancelBtn.textContent = "✖ Cancel";

        const mContinueBtn = document.createElement("button");
        mContinueBtn.type = "button";
        mContinueBtn.className = "zvc-btn continue";
        mContinueBtn.textContent = "▶ Continue Workflow";

        mActions.append(mRefineBtn, mDiscussBtn, mCancelBtn, mContinueBtn);
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
            refineBtn: mRefineBtn,
            discussBtn: mDiscussBtn,
            continueBtn: mContinueBtn,
            cancelBtn: mCancelBtn,
            chatLogEl: mChatLog,
            errorBar: mErrorBar,
            thumbWrap,
            thumbImg,
        };

        bindInputHandlers(
            mPromptBox,
            mReplyBox,
            mVersionSel,
            mUndoBtn,
            mRefineBtn,
            mDiscussBtn,
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

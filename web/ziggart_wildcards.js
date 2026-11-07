import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const WILDCARD_PLACEHOLDER = "[Select a Wildcard to Insert or Resolve]";
const LORA_PLACEHOLDER = "[Select a LoRA to Insert]";
const NONE_OPTION = "[None]";

const WILDCARD_PREVIEW_NODES = new Set([
    "ZiggartWildcardProcessor",
    "ZiggartWildcardEncode",
    "ZiggartWildcardProcessorAllInOne",
    "ZiggartKrea2MasterPromptDirector",
]);

// Listen for backend WebSocket feedback when Queue is clicked or a node finishes executing
api.addEventListener("ziggart-node-feedback", (event) => {
    const detail = event.detail;
    if (!detail || detail.node_id === undefined) return;
    const node =
        app.graph?._nodes_by_id?.[detail.node_id] ||
        app.graph?.getNodeById?.(+detail.node_id);
    if (!node || !Array.isArray(node.widgets)) return;

    const w = node.widgets.find((widget) => widget.name === detail.widget_name);
    if (w) {
        w.value = detail.value;
        if (w.inputEl) {
            w.inputEl.value = detail.value;
        }
        app.graph?.setDirtyCanvas?.(true, true);
    }
});

function collectNodeWidgetInputs(node) {
    const inputs = {};
    if (!Array.isArray(node.widgets)) return inputs;
    for (const w of node.widgets) {
        if (!w || !w.name || w.type === "button") continue;
        inputs[w.name] = w.value;
    }
    return inputs;
}

app.registerExtension({
    name: "Ziggart.Wildcards",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const nodeName = nodeData.name;

        // -------------------------------------------------------------------
        // 1. Interactive 2D Spatial Layout Canvas for ZiggartSpatialLayoutDesigner
        // -------------------------------------------------------------------
        if (nodeName === "ZiggartSpatialLayoutDesigner") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;
                const colors = [
                    { fill: "rgba(255, 92, 124, 0.28)", stroke: "#ff6e8c" },
                    { fill: "rgba(72, 185, 255, 0.28)", stroke: "#60cdff" },
                    { fill: "rgba(110, 235, 145, 0.28)", stroke: "#82faa5" },
                    { fill: "rgba(255, 200, 85, 0.28)", stroke: "#ffd76e" },
                ];

                let activeDrag = null; // { idx, mode: 'move'|'resize', startX, startY, orig: [x1,y1,x2,y2] }

                const getW = (name) => node.widgets?.find((w) => w.name === name);
                const getRegionBox = (idx) => {
                    const p = `r${idx}_`;
                    return {
                        label: getW(p + "label")?.value || `Region ${idx}`,
                        x1: parseFloat(getW(p + "x1")?.value ?? 0.1),
                        y1: parseFloat(getW(p + "y1")?.value ?? 0.1),
                        x2: parseFloat(getW(p + "x2")?.value ?? 0.5),
                        y2: parseFloat(getW(p + "y2")?.value ?? 0.9),
                    };
                };
                const setRegionBox = (idx, x1, y1, x2, y2) => {
                    const p = `r${idx}_`;
                    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.round(v * 100) / 100));
                    const nx1 = clamp(Math.min(x1, x2 - 0.04), 0.0, 0.96);
                    const ny1 = clamp(Math.min(y1, y2 - 0.04), 0.0, 0.96);
                    const nx2 = clamp(Math.max(x2, nx1 + 0.04), 0.04, 1.0);
                    const ny2 = clamp(Math.max(y2, ny1 + 0.04), 0.04, 1.0);
                    if (getW(p + "x1")) getW(p + "x1").value = nx1;
                    if (getW(p + "y1")) getW(p + "y1").value = ny1;
                    if (getW(p + "x2")) getW(p + "x2").value = nx2;
                    if (getW(p + "y2")) getW(p + "y2").value = ny2;
                };

                const layoutWidget = {
                    name: "ziggart_spatial_canvas",
                    type: "custom_canvas",
                    y: 0,
                    computeSize: (width) => [width, 190],
                    draw: function (ctx, _node, widgetWidth, y) {
                        this.y = y;
                        const pad = 12;
                        const cw = Math.max(120, widgetWidth - pad * 2);
                        const ch = 172;
                        const ox = pad;
                        const oy = y + 6;
                        this._rect = { ox, oy, cw, ch };

                        ctx.save();
                        ctx.fillStyle = "#161820";
                        ctx.strokeStyle = "#34394c";
                        ctx.lineWidth = 1;
                        ctx.fillRect(ox, oy, cw, ch);
                        ctx.strokeRect(ox, oy, cw, ch);

                        // Subtle grid
                        ctx.strokeStyle = "#252938";
                        for (let i = 1; i < 4; i++) {
                            ctx.beginPath();
                            ctx.moveTo(ox + (cw * i) / 4, oy);
                            ctx.lineTo(ox + (cw * i) / 4, oy + ch);
                            ctx.stroke();
                            ctx.beginPath();
                            ctx.moveTo(ox, oy + (ch * i) / 4);
                            ctx.lineTo(ox + cw, oy + (ch * i) / 4);
                            ctx.stroke();
                        }

                        const count = Math.max(1, Math.min(4, parseInt(getW("active_regions")?.value ?? 2, 10)));
                        for (let i = 1; i <= count; i++) {
                            const b = getRegionBox(i);
                            const col = colors[(i - 1) % colors.length];
                            const rx = ox + b.x1 * cw;
                            const ry = oy + b.y1 * ch;
                            const rw = Math.max(8, (b.x2 - b.x1) * cw);
                            const rh = Math.max(8, (b.y2 - b.y1) * ch);

                            ctx.fillStyle = col.fill;
                            ctx.fillRect(rx, ry, rw, rh);
                            ctx.strokeStyle = col.stroke;
                            ctx.lineWidth = 2;
                            ctx.strokeRect(rx, ry, rw, rh);

                            // Bottom-right resize handle
                            ctx.fillStyle = col.stroke;
                            ctx.fillRect(rx + rw - 7, ry + rh - 7, 7, 7);

                            ctx.fillStyle = "#f0f4ff";
                            ctx.font = "11px sans-serif";
                            ctx.fillText(`#${i} ${b.label}`, rx + 5, ry + 14);
                        }
                        ctx.restore();
                    },
                    mouse: function (event, pos, _node) {
                        if (!this._rect) return false;
                        const { ox, oy, cw, ch } = this._rect;
                        const mx = (pos[0] - ox) / cw;
                        const my = (pos[1] - oy) / ch;
                        const count = Math.max(1, Math.min(4, parseInt(getW("active_regions")?.value ?? 2, 10)));

                        if (event.type === "pointerdown") {
                            for (let i = count; i >= 1; i--) {
                                const b = getRegionBox(i);
                                if (mx >= b.x1 && mx <= b.x2 && my >= b.y1 && my <= b.y2) {
                                    const nearCorner =
                                        Math.abs(mx - b.x2) * cw <= 12 && Math.abs(my - b.y2) * ch <= 12;
                                    activeDrag = {
                                        idx: i,
                                        mode: nearCorner ? "resize" : "move",
                                        startX: mx,
                                        startY: my,
                                        orig: [b.x1, b.y1, b.x2, b.y2],
                                    };
                                    return true;
                                }
                            }
                        } else if (event.type === "pointermove" && activeDrag) {
                            const dx = mx - activeDrag.startX;
                            const dy = my - activeDrag.startY;
                            const [ox1, oy1, ox2, oy2] = activeDrag.orig;
                            if (activeDrag.mode === "resize") {
                                setRegionBox(activeDrag.idx, ox1, oy1, ox2 + dx, oy2 + dy);
                            } else {
                                const w = ox2 - ox1;
                                const h = oy2 - oy1;
                                const nx1 = Math.max(0.0, Math.min(1.0 - w, ox1 + dx));
                                const ny1 = Math.max(0.0, Math.min(1.0 - h, oy1 + dy));
                                setRegionBox(activeDrag.idx, nx1, ny1, nx1 + w, ny1 + h);
                            }
                            app.graph?.setDirtyCanvas?.(true, true);
                            return true;
                        } else if (event.type === "pointerup" && activeDrag) {
                            activeDrag = null;
                            return true;
                        }
                        return false;
                    },
                };

                if (typeof this.addCustomWidget === "function") {
                    this.addCustomWidget(layoutWidget);
                }
                return r;
            };
            return;
        }

        // -------------------------------------------------------------------
        // 2. Wildcard Processors & Krea2 Master Director: Pre-Generate Button + Helpers
        // -------------------------------------------------------------------
        if (!WILDCARD_PREVIEW_NODES.has(nodeName)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            const wildcardTextWidget = this.widgets?.find((w) => w.name === "wildcard_text");
            const populatedTextWidget = this.widgets?.find((w) => w.name === "populated_text");
            const modeWidget = this.widgets?.find((w) => w.name === "mode");
            const seedWidget = this.widgets?.find((w) => w.name === "seed");

            if (wildcardTextWidget?.inputEl) {
                wildcardTextWidget.inputEl.placeholder =
                    "Wildcard Prompt (e.g. __Ziggart/scenes/all__ or {a|b})";
            }
            if (populatedTextWidget?.inputEl) {
                populatedTextWidget.inputEl.placeholder =
                    "Populated Prompt (Click '🎲 Pre-Generate Prompt' to test without Queue, or click Queue)";
            }

            let lastPreGeneratedSeed = null;

            const refreshPopulatedPreview = async (advanceSeedOnClick = false) => {
                if (!populatedTextWidget) return;
                if (!advanceSeedOnClick && modeWidget && modeWidget.value === "fixed") return;

                let currentSeed = seedWidget ? Number(seedWidget.value) || 0 : 0;
                if (advanceSeedOnClick && seedWidget) {
                    const controlWidget = node.widgets?.find(
                        (w) => w.name === "control_after_generate"
                    );
                    const ctrlMode = controlWidget?.value || "randomize";
                    if (lastPreGeneratedSeed === currentSeed) {
                        if (ctrlMode === "increment") {
                            currentSeed += 1;
                        } else if (ctrlMode === "decrement") {
                            currentSeed = Math.max(0, currentSeed - 1);
                        } else if (ctrlMode !== "fixed") {
                            currentSeed = Math.floor(Math.random() * 2147483647);
                        }
                        seedWidget.value = currentSeed;
                    }
                    lastPreGeneratedSeed = currentSeed;
                }

                const inputs = collectNodeWidgetInputs(node);
                const text = wildcardTextWidget ? wildcardTextWidget.value || "" : "";

                try {
                    const res = await api.fetchApi("/ziggart/wildcards/process", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            class_type: nodeName,
                            text,
                            seed: currentSeed,
                            inputs,
                        }),
                    });
                    if (res.ok) {
                        const data = await res.json();
                        if (data.populated_text !== undefined) {
                            populatedTextWidget.value = data.populated_text;
                            if (populatedTextWidget.inputEl) {
                                populatedTextWidget.inputEl.value = data.populated_text;
                            }
                            app.graph?.setDirtyCanvas?.(true, true);
                        }
                    }
                } catch (_err) {
                    // Backend will still resolve during node execution
                }
            };

            // Add the "🎲 Pre-Generate Prompt (No Queue)" button widget
            if (typeof this.addWidget === "function" && populatedTextWidget) {
                this.addWidget(
                    "button",
                    "🎲 Pre-Generate Prompt (Test Without Queue)",
                    null,
                    () => {
                        refreshPopulatedPreview(true);
                    }
                );
            }

            // Wire 'select_to_add_wildcard' to append into wildcard_text on selection
            const selectWcWidget = this.widgets?.find((w) => w.name === "select_to_add_wildcard");
            if (selectWcWidget && wildcardTextWidget) {
                const origCallback = selectWcWidget.callback;
                selectWcWidget.callback = function (value) {
                    if (origCallback) origCallback.apply(this, arguments);
                    if (value && value !== WILDCARD_PLACEHOLDER && value !== NONE_OPTION) {
                        const current = (wildcardTextWidget.value || "").trim();
                        wildcardTextWidget.value = current ? `${current}, ${value}` : value;
                        if (wildcardTextWidget.inputEl) {
                            wildcardTextWidget.inputEl.value = wildcardTextWidget.value;
                        }
                        selectWcWidget.value = WILDCARD_PLACEHOLDER;
                        refreshPopulatedPreview(false);
                    }
                };
            }

            // On ZiggartWildcardProcessorAllInOne, when combine_selected_pillars is False,
            // selecting from any pillar dropdown appends it directly into wildcard_text
            if (nodeName === "ZiggartWildcardProcessorAllInOne" && wildcardTextWidget) {
                const combineWidget = this.widgets?.find((w) => w.name === "combine_selected_pillars");
                const pillarNames = [
                    "select_macro_yaml_theme",
                    "select_character_anatomy",
                    "select_outfit_pose",
                    "select_env_vehicle",
                    "select_style_camera_light",
                    "select_mixed_other",
                ];
                for (const pName of pillarNames) {
                    const pw = this.widgets?.find((w) => w.name === pName);
                    if (!pw) continue;
                    const origPillarCb = pw.callback;
                    pw.callback = function (value) {
                        if (origPillarCb) origPillarCb.apply(this, arguments);
                        if (
                            combineWidget &&
                            combineWidget.value === false &&
                            value &&
                            value !== NONE_OPTION &&
                            value !== WILDCARD_PLACEHOLDER
                        ) {
                            const current = (wildcardTextWidget.value || "").trim();
                            wildcardTextWidget.value = current ? `${current}, ${value}` : value;
                            if (wildcardTextWidget.inputEl) {
                                wildcardTextWidget.inputEl.value = wildcardTextWidget.value;
                            }
                            pw.value = NONE_OPTION;
                        }
                        refreshPopulatedPreview(false);
                    };
                }
            }

            // Wire 'select_to_add_lora' on ZiggartWildcardEncode
            const selectLoraWidget = this.widgets?.find((w) => w.name === "select_to_add_lora");
            if (selectLoraWidget && wildcardTextWidget) {
                const origLoraCallback = selectLoraWidget.callback;
                selectLoraWidget.callback = function (value) {
                    if (origLoraCallback) origLoraCallback.apply(this, arguments);
                    if (value && value !== LORA_PLACEHOLDER && value !== NONE_OPTION) {
                        const stem = value.replace(/\.(safetensors|pt|ckpt|bin)$/i, "");
                        const tag = `<lora:${stem}:1.0:1.0>`;
                        const current = (wildcardTextWidget.value || "").trim();
                        wildcardTextWidget.value = current ? `${current} ${tag}` : tag;
                        if (wildcardTextWidget.inputEl) {
                            wildcardTextWidget.inputEl.value = wildcardTextWidget.value;
                        }
                        selectLoraWidget.value = LORA_PLACEHOLDER;
                    }
                };
            }

            return r;
        };
    },
});

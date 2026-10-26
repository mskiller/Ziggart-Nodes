import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const WILDCARD_PLACEHOLDER = "[Select a Wildcard to Insert or Resolve]";
const LORA_PLACEHOLDER = "[Select a LoRA to Insert]";
const NONE_OPTION = "[None]";

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

app.registerExtension({
    name: "Ziggart.Wildcards",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const nodeName = nodeData.name;
        if (
            nodeName !== "ZiggartWildcardProcessor" &&
            nodeName !== "ZiggartWildcardEncode" &&
            nodeName !== "ZiggartWildcardProcessorAllInOne"
        ) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
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
                    "Populated Prompt (Automatically filled when you click Queue)";
            }

            const refreshPopulatedPreview = async () => {
                if (!wildcardTextWidget || !populatedTextWidget) return;
                if (modeWidget && modeWidget.value === "fixed") return;
                const text = wildcardTextWidget.value || "";
                if (!text.trim()) return;
                try {
                    const res = await api.fetchApi("/ziggart/wildcards/process", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            text,
                            seed: seedWidget ? seedWidget.value : 0,
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
                        refreshPopulatedPreview();
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
                        refreshPopulatedPreview();
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

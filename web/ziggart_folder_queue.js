import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function setStatusWidgetValue(node, text) {
    const w = getWidget(node, "status_text");
    if (!w) return;
    w.value = text;
    if (w.inputEl) {
        w.inputEl.value = text;
    }
    app.graph?.setDirtyCanvas?.(true, true);
}

async function fetchFolderScan(node) {
    const folderPath = getWidget(node, "folder_path")?.value || "";
    if (!folderPath.trim()) {
        setStatusWidgetValue(
            node,
            "Enter a folder_path and click '🔍 Scan Folder & Count Queues'."
        );
        return null;
    }

    const payload = {
        folder_path: folderPath,
        image_index: Number(getWidget(node, "image_index")?.value ?? 0) || 0,
        sort_method: getWidget(node, "sort_method")?.value || "natural_alphabetical",
        include_subfolders: Boolean(getWidget(node, "include_subfolders")?.value),
        skip_existing_txt: Boolean(getWidget(node, "skip_existing_txt")?.value),
        loop_index: Boolean(getWidget(node, "loop_index")?.value),
        filename_filter: getWidget(node, "filename_filter")?.value || "",
        seed: Number(getWidget(node, "seed")?.value ?? 42) || 42,
    };

    try {
        const res = await api.fetchApi("/ziggart/folder_queue/scan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) {
            return null;
        }
        const data = await res.json();
        if (data.status_text !== undefined) {
            setStatusWidgetValue(node, data.status_text);
        }
        return data;
    } catch (_err) {
        return null;
    }
}

app.registerExtension({
    name: "Ziggart.FolderImageQueue",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ZiggartFolderImageQueueLoader") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            // Default control_after_generate on image_index to 'increment'
            setTimeout(() => {
                const ctrlWidget = getWidget(node, "control_after_generate");
                if (ctrlWidget && (!ctrlWidget.value || ctrlWidget.value === "randomize")) {
                    ctrlWidget.value = "increment";
                }
            }, 0);

            const folderWidget = getWidget(node, "folder_path");
            if (folderWidget?.inputEl) {
                folderWidget.inputEl.placeholder =
                    "e.g. J:\\dataset\\images (supports jpg, png, webp, bmp, tiff, etc.)";
            }

            // Auto-scan when folder settings change
            const watchedNames = [
                "folder_path",
                "image_index",
                "sort_method",
                "include_subfolders",
                "skip_existing_txt",
                "loop_index",
                "filename_filter",
                "seed",
            ];
            for (const wName of watchedNames) {
                const w = getWidget(node, wName);
                if (!w) continue;
                const origCb = w.callback;
                w.callback = function () {
                    if (origCb) origCb.apply(this, arguments);
                    fetchFolderScan(node);
                };
            }

            // Button 1: Scan Folder & Count Queues
            if (typeof this.addWidget === "function") {
                this.addWidget(
                    "button",
                    "🔍 Scan Folder & Count Queues",
                    null,
                    async () => {
                        await fetchFolderScan(node);
                    }
                );

                // Button 2: Reset Index to 0
                this.addWidget(
                    "button",
                    "🔄 Reset Index to 0",
                    null,
                    async () => {
                        const idxWidget = getWidget(node, "image_index");
                        if (idxWidget) {
                            idxWidget.value = 0;
                        }
                        await fetchFolderScan(node);
                    }
                );

                // Button 3: Queue All Remaining Images (1-by-1)
                let isBatchQueuing = false;
                this.addWidget(
                    "button",
                    "▶️ Queue All Remaining Images (1-by-1)",
                    null,
                    async () => {
                        if (isBatchQueuing) return;
                        isBatchQueuing = true;
                        try {
                            const scan = await fetchFolderScan(node);
                            if (!scan || !scan.ok) {
                                return;
                            }
                            const totalNeeded = Number(scan.queues_needed || 0);
                            if (totalNeeded <= 0) {
                                return;
                            }

                            const idxWidget = getWidget(node, "image_index");
                            let startIdx = Number(idxWidget?.value ?? 0) || 0;
                            if (startIdx < 0 || startIdx >= totalNeeded) {
                                startIdx = 0;
                            }

                            const runsToQueue = totalNeeded - startIdx;
                            for (let idx = startIdx; idx < totalNeeded; idx++) {
                                if (idxWidget) {
                                    idxWidget.value = idx;
                                }
                                setStatusWidgetValue(
                                    node,
                                    `🚀 Enqueuing [${idx + 1}/${totalNeeded}] (Index ${idx})...`
                                );
                                await app.queuePrompt(0, 1);
                            }

                            setStatusWidgetValue(
                                node,
                                `✅ Queued ${runsToQueue} run(s) (Index ${startIdx} → ${totalNeeded - 1}) into ComfyUI!\n` +
                                    `📊 Total Images in Queue Batch: ${runsToQueue} / ${totalNeeded}`
                            );
                        } finally {
                            isBatchQueuing = false;
                        }
                    }
                );
            }

            return r;
        };
    },
});

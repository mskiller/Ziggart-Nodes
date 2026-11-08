import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Ziggart 3D VRM Pose Editor & Live Webcam "Replace Me!" Mocap Studio
 * Interactive 60 FPS Canvas Widget with Verlet Spring-Bone Hair Physics,
 * Interactive Bone Gizmos, Microphone Lip-Sync, Webcam Face/Body Tracking,
 * and Pop-Out Broadcast Window for OBS / Discord / Zoom.
 */

const STUDIO_NODE_NAME = "ZiggartVRMPoseAndMocapStudio";

// Canonical 3D humanoid rest skeleton for interactive viewport + physics preview
const REST_BONES = {
    hips: { parent: null, pos: [0.0, -0.05, 0.0] },
    spine: { parent: "hips", pos: [0.0, 0.12, 0.0] },
    chest: { parent: "spine", pos: [0.0, 0.28, 0.0] },
    neck: { parent: "chest", pos: [0.0, 0.42, 0.0] },
    head: { parent: "neck", pos: [0.0, 0.56, 0.0] },
    leftUpperArm: { parent: "chest", pos: [0.16, 0.38, 0.0] },
    leftLowerArm: { parent: "leftUpperArm", pos: [0.33, 0.37, 0.0] },
    leftHand: { parent: "leftLowerArm", pos: [0.48, 0.36, 0.0] },
    rightUpperArm: { parent: "chest", pos: [-0.16, 0.38, 0.0] },
    rightLowerArm: { parent: "rightUpperArm", pos: [-0.33, 0.37, 0.0] },
    rightHand: { parent: "rightLowerArm", pos: [-0.48, 0.36, 0.0] },
    leftUpperLeg: { parent: "hips", pos: [0.09, -0.08, 0.0] },
    leftLowerLeg: { parent: "leftUpperLeg", pos: [0.09, -0.40, 0.0] },
    leftFoot: { parent: "leftLowerLeg", pos: [0.09, -0.70, 0.03] },
    rightUpperLeg: { parent: "hips", pos: [-0.09, -0.08, 0.0] },
    rightLowerLeg: { parent: "rightUpperLeg", pos: [-0.09, -0.40, 0.0] },
    rightFoot: { parent: "rightLowerLeg", pos: [-0.09, -0.70, 0.03] },
};

function degToRad(d) {
    return (d * Math.PI) / 180.0;
}

function rotateVec3EulerDeg(v, rxDeg, ryDeg, rzDeg) {
    const rx = degToRad(rxDeg);
    const ry = degToRad(ryDeg);
    const rz = degToRad(rzDeg);
    const cx = Math.cos(rx), sx = Math.sin(rx);
    const cy = Math.cos(ry), sy = Math.sin(ry);
    const cz = Math.cos(rz), sz = Math.sin(rz);

    // Rotate X
    let x1 = v[0];
    let y1 = v[1] * cx - v[2] * sx;
    let z1 = v[1] * sx + v[2] * cx;
    // Rotate Y
    let x2 = x1 * cy + z1 * sy;
    let y2 = y1;
    let z2 = -x1 * sy + z1 * cy;
    // Rotate Z
    let x3 = x2 * cz - y2 * sz;
    let y3 = x2 * sz + y2 * cz;
    let z3 = z2;
    return [x3, y3, z3];
}

function createHairSpringStrands(numStrands = 18, segs = 6) {
    const strands = [];
    for (let i = 0; i < numStrands; i++) {
        const angle = (i / numStrands) * Math.PI * 2.0 - Math.PI;
        const isFront = Math.abs(angle) < 0.75;
        const len = isFront ? 0.18 : 0.42;
        const rootOffset = [
            Math.sin(angle) * 0.13,
            0.10 + Math.cos(angle * 2.0) * 0.02,
            Math.cos(angle) * 0.13,
        ];
        const pts = [];
        const prevPts = [];
        for (let k = 0; k <= segs; k++) {
            const t = k / segs;
            const p = [
                rootOffset[0] * (1.0 + 0.35 * t),
                0.56 + rootOffset[1] - t * len,
                rootOffset[2] * (1.0 + 0.25 * t),
            ];
            pts.push([...p]);
            prevPts.push([...p]);
        }
        strands.push({
            angle,
            isFront,
            segLen: len / segs,
            rootOffset,
            pts,
            prevPts,
            hue: 265 + (i % 5) * 10,
        });
    }
    return strands;
}

app.registerExtension({
    name: "Ziggart.AvatarStudio",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== STUDIO_NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            const getW = (name) => node.widgets?.find((w) => w.name === name);

            const studioState = {
                liveWebcamActive: false,
                videoEl: null,
                audioCtx: null,
                analyser: null,
                mediaStream: null,
                popoutWin: null,
                selectedBone: "head",
                draggingBone: false,
                orbitDragging: false,
                lastMouse: [0, 0],
                rotations: {
                    hips: [0, 0, 0],
                    chest: [0, 0, 0],
                    head: [0, 0, 0],
                    leftUpperArm: [15, 20, -45],
                    leftLowerArm: [35, 0, -15],
                    rightUpperArm: [45, -15, 25],
                    rightLowerArm: [85, 10, 35],
                },
                expressions: {
                    aa: 0.0,
                    blink: 0.0,
                    happy: 0.6,
                    chibi_big_head: 0.0,
                },
                hairStrands: createHairSpringStrands(20, 6),
                particles: [],
                vrmLoadedPath: "",
                prevFrameLuma: null,
                offscreenTrackCanvas: document.createElement("canvas"),
                offscreenRenderCanvas: document.createElement("canvas"),
                animFrameId: null,
            };
            studioState.offscreenTrackCanvas.width = 96;
            studioState.offscreenTrackCanvas.height = 72;
            studioState.offscreenRenderCanvas.width = 512;
            studioState.offscreenRenderCanvas.height = 512;

            // Listen for backend VRM path notifications
            api.addEventListener("ziggart-avatar-loaded", (ev) => {
                const detail = ev.detail || {};
                const sess = getW("session_id")?.value || "default";
                if (!detail.session_id || detail.session_id === sess) {
                    studioState.vrmLoadedPath = detail.vrm_path || "";
                    app.graph?.setDirtyCanvas?.(true, true);
                }
            });

            // Sync current pose + expressions + snapshot to Python backend
            const syncToBackend = async () => {
                const sess = getW("session_id")?.value || "default";
                let frameUri = "";
                try {
                    frameUri = studioState.offscreenRenderCanvas.toDataURL("image/png");
                } catch (_e) {
                    frameUri = "";
                }
                const payload = {
                    session_id: sess,
                    rotations: studioState.rotations,
                    expressions: studioState.expressions,
                    frame_data_uri: frameUri,
                };
                const jsonW = getW("webcam_pose_json");
                if (jsonW) {
                    jsonW.value = JSON.stringify({
                        rotations: studioState.rotations,
                        expressions: studioState.expressions,
                    });
                }
                try {
                    await api.fetchApi("/ziggart/avatar/sync_capture", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
                    });
                } catch (_err) {
                    // Ignore network error during offline preview
                }
            };

            // Real-time Webcam + Microphone Tracker (Optical Centroid + Face Box + WebAudio Formant LipSync)
            const updateLiveWebcamTracking = () => {
                if (!studioState.liveWebcamActive || !studioState.videoEl) return;
                const vid = studioState.videoEl;
                if (vid.readyState < 2) return;

                // 1. Microphone Amplitude -> Mouth Open (`aa`) Lip-Sync
                if (studioState.analyser) {
                    const bins = new Uint8Array(studioState.analyser.frequencyBinCount);
                    studioState.analyser.getByteFrequencyData(bins);
                    let sum = 0;
                    for (let i = 2; i < Math.min(36, bins.length); i++) sum += bins[i];
                    const avg = sum / 34.0 / 255.0;
                    const targetMouth = Math.min(1.0, Math.max(0.0, (avg - 0.06) * 2.8));
                    studioState.expressions.aa = studioState.expressions.aa * 0.55 + targetMouth * 0.45;
                    if (getW("mouth_open_aa")) {
                        getW("mouth_open_aa").value = Math.round(studioState.expressions.aa * 100) / 100;
                    }
                }

                // 2. Fast Optical Face/Head/Hand Centroid Tracker on 96x72 downsampled webcam frame
                const tCtx = studioState.offscreenTrackCanvas.getContext("2d", { willReadFrequently: true });
                const tw = 96, th = 72;
                tCtx.save();
                // Mirror webcam horizontally so moving left/right feels like a mirror
                tCtx.translate(tw, 0);
                tCtx.scale(-1, 1);
                tCtx.drawImage(vid, 0, 0, tw, th);
                tCtx.restore();

                const imgData = tCtx.getImageData(0, 0, tw, th).data;
                let skinSumX = 0, skinSumY = 0, skinCount = 0;
                let leftHandY = 0, leftHandCnt = 0;
                let rightHandY = 0, rightHandCnt = 0;

                for (let y = 0; y < th; y++) {
                    for (let x = 0; x < tw; x++) {
                        const idx = (y * tw + x) * 4;
                        const r = imgData[idx], g = imgData[idx + 1], b = imgData[idx + 2];
                        // Warm skin-tone / face luminance heuristic
                        const isSkin = r > 75 && g > 45 && b > 30 && r > g && r > b && (r - Math.min(g, b)) > 18;
                        if (isSkin) {
                            if (x > tw * 0.22 && x < tw * 0.78 && y < th * 0.75) {
                                skinSumX += x;
                                skinSumY += y;
                                skinCount++;
                            } else if (x <= tw * 0.22) {
                                leftHandY += y;
                                leftHandCnt++;
                            } else if (x >= tw * 0.78) {
                                rightHandY += y;
                                rightHandCnt++;
                            }
                        }
                    }
                }

                if (skinCount > 35) {
                    const cx = skinSumX / skinCount / tw; // [0..1]
                    const cy = skinSumY / skinCount / th; // [0..1]
                    const targetYaw = (cx - 0.5) * 75.0;
                    const targetPitch = (cy - 0.42) * 55.0;
                    const prevHead = studioState.rotations.head || [0, 0, 0];
                    const newPitch = prevHead[0] * 0.70 + targetPitch * 0.30;
                    const newYaw = prevHead[1] * 0.70 + targetYaw * 0.30;
                    const newRoll = (cx - 0.5) * -28.0;
                    studioState.rotations.head = [newPitch, newYaw, newRoll];
                    studioState.rotations.chest = [newPitch * 0.25, newYaw * 0.35, newRoll * 0.30];

                    if (getW("head_yaw")) getW("head_yaw").value = Math.round(newYaw);
                    if (getW("head_pitch")) getW("head_pitch").value = Math.round(newPitch);
                    if (getW("head_roll")) getW("head_roll").value = Math.round(newRoll);
                }

                // Hand raise tracking -> raises left/right avatar arms & spawns fun emote particles!
                if (leftHandCnt > 18) {
                    const ly = leftHandY / leftHandCnt / th;
                    const armRaise = Math.max(-45, Math.min(70, (0.6 - ly) * 140));
                    studioState.rotations.rightUpperArm = [25, -10, armRaise];
                    studioState.rotations.rightLowerArm = [75, 10, 30];
                    if (ly < 0.38 && Math.random() < 0.18) {
                        studioState.particles.push({
                            x: (Math.random() - 0.5) * 0.6,
                            y: 0.65,
                            vx: (Math.random() - 0.5) * 0.01,
                            vy: 0.012 + Math.random() * 0.01,
                            life: 1.0,
                            symbol: Math.random() > 0.5 ? "✨" : "✌️",
                        });
                    }
                }
                if (rightHandCnt > 18) {
                    const ry = rightHandY / rightHandCnt / th;
                    const armRaise = Math.max(-70, Math.min(45, -(0.6 - ry) * 140));
                    studioState.rotations.leftUpperArm = [25, 10, armRaise];
                    studioState.rotations.leftLowerArm = [75, -10, -30];
                }

                // Natural periodic eye blink every ~3.8 seconds
                const tBlink = (performance.now() * 0.001) % 3.8;
                studioState.expressions.blink = tBlink < 0.14 ? Math.sin((tBlink / 0.14) * Math.PI) : 0.0;
            };

            // Step Verlet Spring-Bone Hair Physics & Render 3D Avatar Scene
            const renderAvatarScene = (ctx, width, height) => {
                const bgMode = getW("background_mode")?.value || "dark_studio";
                const azDeg = parseFloat(getW("camera_azimuth")?.value ?? 12.0);
                const elDeg = parseFloat(getW("camera_elevation")?.value ?? 10.0);

                // Read manual widget sliders when not overridden by live webcam
                if (!studioState.liveWebcamActive) {
                    const hy = parseFloat(getW("head_yaw")?.value ?? 0.0);
                    const hp = parseFloat(getW("head_pitch")?.value ?? 0.0);
                    const hr = parseFloat(getW("head_roll")?.value ?? 0.0);
                    studioState.rotations.head = [hp, hy, hr];
                    studioState.expressions.aa = parseFloat(getW("mouth_open_aa")?.value ?? 0.0);
                    studioState.expressions.blink = parseFloat(getW("eye_blink")?.value ?? 0.0);
                    studioState.expressions.happy = parseFloat(getW("emotion_happy")?.value ?? 0.5);
                    studioState.expressions.chibi_big_head = parseFloat(getW("chibi_scale")?.value ?? 0.0);
                }

                ctx.save();
                // 1. Background Fill or Live Webcam "Replace Me" Composite
                if (bgMode === "green_screen") {
                    ctx.fillStyle = "#00ff00";
                    ctx.fillRect(0, 0, width, height);
                } else if (bgMode === "white_studio") {
                    ctx.fillStyle = "#f3f5f9";
                    ctx.fillRect(0, 0, width, height);
                } else if (bgMode === "webcam_replace_me" && studioState.liveWebcamActive && studioState.videoEl) {
                    ctx.save();
                    ctx.translate(width, 0);
                    ctx.scale(-1, 1);
                    ctx.drawImage(studioState.videoEl, 0, 0, width, height);
                    ctx.restore();
                    // Subtle vignette dim so 3D avatar pops over real room background
                    ctx.fillStyle = "rgba(12, 15, 24, 0.22)";
                    ctx.fillRect(0, 0, width, height);
                } else {
                    const grad = ctx.createLinearGradient(0, 0, 0, height);
                    grad.addColorStop(0, "#171a26");
                    grad.addColorStop(1, "#0e1017");
                    ctx.fillRect(0, 0, width, height);
                    ctx.fillStyle = grad;
                    ctx.fillRect(0, 0, width, height);
                }

                // 2. Forward Kinematics of Humanoid Skeleton
                const worldBones = {};
                const boneNames = Object.keys(REST_BONES);
                for (const bName of boneNames) {
                    const spec = REST_BONES[bName];
                    if (!spec.parent) {
                        worldBones[bName] = [...spec.pos];
                    } else {
                        const pRest = REST_BONES[spec.parent].pos;
                        const pWorld = worldBones[spec.parent];
                        const rel = [
                            spec.pos[0] - pRest[0],
                            spec.pos[1] - pRest[1],
                            spec.pos[2] - pRest[2],
                        ];
                        const pRot = studioState.rotations[spec.parent] || [0, 0, 0];
                        const rRel = rotateVec3EulerDeg(rel, pRot[0], pRot[1], pRot[2]);
                        worldBones[bName] = [
                            pWorld[0] + rRel[0],
                            pWorld[1] + rRel[1],
                            pWorld[2] + rRel[2],
                        ];
                    }
                }

                const headPos = worldBones.head;
                const headRot = studioState.rotations.head || [0, 0, 0];
                const chibiScale = 1.0 + (studioState.expressions.chibi_big_head || 0.0) * 0.45;

                // 3. Project 3D point -> 2D Viewport Canvas
                const project3D = (pt) => {
                    const camRot = rotateVec3EulerDeg(pt, elDeg, azDeg, 0.0);
                    const zDist = Math.max(0.3, 2.15 - camRot[2]);
                    const focal = 1.45;
                    const sx = width * 0.5 + (camRot[0] * focal / zDist) * Math.min(width, height) * 0.52;
                    const sy = height * 0.54 - (camRot[1] * focal / zDist) * Math.min(width, height) * 0.52;
                    return [sx, sy, zDist];
                };

                // Draw perspective floor grid in studio mode
                if (bgMode === "dark_studio") {
                    ctx.strokeStyle = "rgba(96, 205, 255, 0.14)";
                    ctx.lineWidth = 1;
                    for (let g = -0.6; g <= 0.61; g += 0.2) {
                        const a = project3D([g, -0.74, -0.6]);
                        const b = project3D([g, -0.74, 0.6]);
                        ctx.beginPath();
                        ctx.moveTo(a[0], a[1]);
                        ctx.lineTo(b[0], b[1]);
                        ctx.stroke();
                        const c = project3D([-0.6, -0.74, g]);
                        const d = project3D([0.6, -0.74, g]);
                        ctx.beginPath();
                        ctx.moveTo(c[0], c[1]);
                        ctx.lineTo(d[0], d[1]);
                        ctx.stroke();
                    }
                }

                // 4. Update & Draw Verlet Spring-Bone HairDeck Strands (Back layer first)
                const drawStrandLayer = (frontPass) => {
                    for (const strand of studioState.hairStrands) {
                        if (strand.isFront !== frontPass) continue;
                        const scaledRoot = [
                            strand.rootOffset[0] * chibiScale,
                            strand.rootOffset[1] * chibiScale,
                            strand.rootOffset[2] * chibiScale,
                        ];
                        const rotRoot = rotateVec3EulerDeg(scaledRoot, headRot[0], headRot[1], headRot[2]);
                        strand.pts[0] = [
                            headPos[0] + rotRoot[0],
                            headPos[1] + rotRoot[1],
                            headPos[2] + rotRoot[2],
                        ];

                        // Verlet integration + distance + stiffness constraint along strand chain
                        for (let k = 1; k < strand.pts.length; k++) {
                            const cur = strand.pts[k];
                            const prev = strand.prevPts[k];
                            const vx = (cur[0] - prev[0]) * 0.82;
                            const vy = (cur[1] - prev[1]) * 0.82 - 0.0045; // Gravity
                            const vz = (cur[2] - prev[2]) * 0.82;
                            strand.prevPts[k] = [...cur];

                            // Desired rest target relative to parent joint
                            const ideal = [
                                strand.pts[k - 1][0] + rotRoot[0] * 0.15,
                                strand.pts[k - 1][1] - strand.segLen,
                                strand.pts[k - 1][2] + rotRoot[2] * 0.15,
                            ];
                            let nx = cur[0] + vx + (ideal[0] - cur[0]) * 0.28;
                            let ny = cur[1] + vy + (ideal[1] - cur[1]) * 0.28;
                            let nz = cur[2] + vz + (ideal[2] - cur[2]) * 0.28;

                            // Enforce segment length
                            const dx = nx - strand.pts[k - 1][0];
                            const dy = ny - strand.pts[k - 1][1];
                            const dz = nz - strand.pts[k - 1][2];
                            const d = Math.hypot(dx, dy, dz) || 1e-5;
                            strand.pts[k] = [
                                strand.pts[k - 1][0] + (dx / d) * strand.segLen,
                                strand.pts[k - 1][1] + (dy / d) * strand.segLen,
                                strand.pts[k - 1][2] + (dz / d) * strand.segLen,
                            ];
                        }

                        // Draw stylized ribbon hair card
                        ctx.beginPath();
                        for (let k = 0; k < strand.pts.length; k++) {
                            const sp = project3D(strand.pts[k]);
                            if (k === 0) ctx.moveTo(sp[0], sp[1]);
                            else ctx.lineTo(sp[0], sp[1]);
                        }
                        ctx.strokeStyle = `hsla(${strand.hue}, 78%, ${frontPass ? 66 : 48}%, 0.92)`;
                        ctx.lineWidth = frontPass ? 5.5 * chibiScale : 7.0 * chibiScale;
                        ctx.lineCap = "round";
                        ctx.stroke();
                    }
                };

                drawStrandLayer(false);

                // 5. Draw Stylized 3D Avatar Body & Limbs
                const limbPairs = [
                    ["hips", "spine", 16, "#4d597a"],
                    ["spine", "chest", 20, "#5e6d94"],
                    ["chest", "neck", 12, "#e8c3b0"],
                    ["chest", "leftUpperArm", 11, "#5e6d94"],
                    ["leftUpperArm", "leftLowerArm", 10, "#4d597a"],
                    ["leftLowerArm", "leftHand", 9, "#f2ccb8"],
                    ["chest", "rightUpperArm", 11, "#5e6d94"],
                    ["rightUpperArm", "rightLowerArm", 10, "#4d597a"],
                    ["rightLowerArm", "rightHand", 9, "#f2ccb8"],
                    ["hips", "leftUpperLeg", 13, "#373f59"],
                    ["leftUpperLeg", "leftLowerLeg", 11, "#2f364d"],
                    ["leftLowerLeg", "leftFoot", 10, "#23283b"],
                    ["hips", "rightUpperLeg", 13, "#373f59"],
                    ["rightUpperLeg", "rightLowerLeg", 11, "#2f364d"],
                    ["rightLowerLeg", "rightFoot", 10, "#23283b"],
                ];

                for (const [bA, bB, thickness, color] of limbPairs) {
                    const pA = project3D(worldBones[bA]);
                    const pB = project3D(worldBones[bB]);
                    ctx.beginPath();
                    ctx.moveTo(pA[0], pA[1]);
                    ctx.lineTo(pB[0], pB[1]);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = thickness;
                    ctx.lineCap = "round";
                    ctx.stroke();
                }

                // 6. Draw 3D Avatar Head + Facial BlendShapes (Eyes, Blink, Lip-Sync Mouth, Blush)
                const headScreen = project3D(headPos);
                const headRadiusPx = (Math.min(width, height) * 0.115 * chibiScale) / Math.max(0.5, headScreen[2] * 0.55);

                ctx.beginPath();
                ctx.arc(headScreen[0], headScreen[1], headRadiusPx, 0, Math.PI * 2);
                ctx.fillStyle = "#ffe0d1";
                ctx.fill();
                ctx.lineWidth = 2;
                ctx.strokeStyle = "#d9a996";
                ctx.stroke();

                // Eyes & Mouth projected on front of head sphere
                const eyeBlink = studioState.expressions.blink || 0.0;
                const happy = studioState.expressions.happy || 0.0;
                const mouthOpen = studioState.expressions.aa || 0.0;

                for (const sx of [-1, 1]) {
                    const eye3D = rotateVec3EulerDeg(
                        [sx * 0.052 * chibiScale, 0.02 * chibiScale, 0.11 * chibiScale],
                        headRot[0],
                        headRot[1],
                        headRot[2]
                    );
                    const ep = project3D([headPos[0] + eye3D[0], headPos[1] + eye3D[1], headPos[2] + eye3D[2]]);
                    const ew = headRadiusPx * 0.22;
                    const eh = Math.max(1.5, headRadiusPx * 0.24 * (1.0 - eyeBlink * 0.92));

                    ctx.save();
                    ctx.beginPath();
                    ctx.ellipse(ep[0], ep[1], ew, eh, 0, 0, Math.PI * 2);
                    ctx.fillStyle = "#2b1d38";
                    ctx.fill();
                    if (eyeBlink < 0.6) {
                        ctx.beginPath();
                        ctx.arc(ep[0] - ew * 0.25, ep[1] - eh * 0.25, ew * 0.32, 0, Math.PI * 2);
                        ctx.fillStyle = "#ffffff";
                        ctx.fill();
                    }
                    ctx.restore();
                }

                // Mouth (`aa` lip-sync + `happy` smile curve)
                const mouth3D = rotateVec3EulerDeg(
                    [0.0, -0.042 * chibiScale, 0.115 * chibiScale],
                    headRot[0],
                    headRot[1],
                    headRot[2]
                );
                const mp = project3D([headPos[0] + mouth3D[0], headPos[1] + mouth3D[1], headPos[2] + mouth3D[2]]);
                const mw = headRadiusPx * (0.24 + happy * 0.12);
                const mh = Math.max(2.0, headRadiusPx * (0.06 + mouthOpen * 0.34));
                ctx.beginPath();
                ctx.ellipse(mp[0], mp[1], mw, mh, 0, 0, Math.PI * 2);
                ctx.fillStyle = mouthOpen > 0.08 ? "#d94b68" : "#b85d6f";
                ctx.fill();

                // Front HairDeck bangs layer (with live spring physics)
                drawStrandLayer(true);

                // 7. Interactive Bone Gizmo Handles
                this._boneScreenMap = {};
                for (const bName of ["head", "chest", "leftLowerArm", "rightLowerArm", "hips"]) {
                    const sp = project3D(worldBones[bName]);
                    this._boneScreenMap[bName] = sp;
                    const isSel = studioState.selectedBone === bName;
                    ctx.beginPath();
                    ctx.arc(sp[0], sp[1], isSel ? 6 : 4, 0, Math.PI * 2);
                    ctx.fillStyle = isSel ? "#00f2ff" : "rgba(255, 255, 255, 0.65)";
                    ctx.fill();
                    ctx.strokeStyle = "#10141f";
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }

                // 8. Floating Fun Emote Particles
                for (let i = studioState.particles.length - 1; i >= 0; i--) {
                    const pt = studioState.particles[i];
                    pt.x += pt.vx;
                    pt.y += pt.vy;
                    pt.life -= 0.025;
                    if (pt.life <= 0) {
                        studioState.particles.splice(i, 1);
                        continue;
                    }
                    const sp = project3D([pt.x, pt.y, 0.15]);
                    ctx.font = "16px sans-serif";
                    ctx.globalAlpha = Math.max(0, pt.life);
                    ctx.fillText(pt.symbol, sp[0], sp[1]);
                    ctx.globalAlpha = 1.0;
                }

                ctx.restore();
            };

            // Animation loop when Live Webcam or Pop-Out Broadcast Window is active
            const startLiveLoop = () => {
                if (studioState.animFrameId) return;
                const tick = () => {
                    if (!studioState.liveWebcamActive && (!studioState.popoutWin || studioState.popoutWin.closed)) {
                        studioState.animFrameId = null;
                        return;
                    }
                    updateLiveWebcamTracking();

                    // Render to offscreen 512x512 canvas for backend sync & Pop-Out Broadcast window
                    const offCtx = studioState.offscreenRenderCanvas.getContext("2d");
                    renderAvatarScene(offCtx, 512, 512);

                    if (studioState.popoutWin && !studioState.popoutWin.closed) {
                        const popCanvas = studioState.popoutWin.document.getElementById("ziggart_broadcast_canvas");
                        if (popCanvas) {
                            const pCtx = popCanvas.getContext("2d");
                            pCtx.drawImage(studioState.offscreenRenderCanvas, 0, 0, popCanvas.width, popCanvas.height);
                        }
                    }

                    app.graph?.setDirtyCanvas?.(true, false);
                    studioState.animFrameId = requestAnimationFrame(tick);
                };
                studioState.animFrameId = requestAnimationFrame(tick);
            };

            const toggleLiveWebcam = async () => {
                if (studioState.liveWebcamActive) {
                    studioState.liveWebcamActive = false;
                    if (studioState.mediaStream) {
                        studioState.mediaStream.getTracks().forEach((t) => t.stop());
                        studioState.mediaStream = null;
                    }
                    await syncToBackend();
                    app.graph?.setDirtyCanvas?.(true, true);
                    return;
                }

                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { width: 640, height: 480, frameRate: 30 },
                        audio: true,
                    });
                    studioState.mediaStream = stream;
                    const video = document.createElement("video");
                    video.srcObject = stream;
                    video.muted = true;
                    video.playsInline = true;
                    await video.play();
                    studioState.videoEl = video;

                    // WebAudio Microphone Analyser for real-time vowel lip-sync
                    const AudioContextCls = window.AudioContext || window.webkitAudioContext;
                    if (AudioContextCls && stream.getAudioTracks().length > 0) {
                        const aCtx = new AudioContextCls();
                        const src = aCtx.createMediaStreamSource(stream);
                        const analyser = aCtx.createAnalyser();
                        analyser.fftSize = 128;
                        src.connect(analyser);
                        studioState.audioCtx = aCtx;
                        studioState.analyser = analyser;
                    }

                    studioState.liveWebcamActive = true;
                    const modeW = getW("studio_mode");
                    if (modeW) modeW.value = "Live Webcam Mocap Capture (Browser Studio)";
                    startLiveLoop();
                } catch (err) {
                    alert("Webcam / Microphone access denied or unavailable: " + err.message);
                }
            };

            const openBroadcastWindow = () => {
                if (studioState.popoutWin && !studioState.popoutWin.closed) {
                    studioState.popoutWin.focus();
                    return;
                }
                const win = window.open("", "ZiggartLiveAvatarBroadcast", "width=720,height=720,menubar=no,toolbar=no");
                if (!win) return;
                win.document.write(`
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Ziggart Live 3D Avatar Broadcast (Capture in OBS / Discord / Zoom)</title>
                        <style>
                            body { margin: 0; background: #0e1017; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; font-family: sans-serif; color: #e2e8f0; overflow: hidden; }
                            canvas { width: 100vmin; height: 100vmin; object-fit: contain; }
                        </style>
                    </head>
                    <body>
                        <canvas id="ziggart_broadcast_canvas" width="720" height="720"></canvas>
                    </body>
                    </html>
                `);
                win.document.close();
                studioState.popoutWin = win;
                startLiveLoop();
            };

            // Add Interactive Control Buttons to the Node
            if (typeof this.addWidget === "function") {
                this.addWidget("button", "🎥 Toggle Live Webcam & Mic Mocap (Replace Me!)", null, () => {
                    toggleLiveWebcam();
                });
                this.addWidget("button", "🗗 Pop-Out Broadcast Window (OBS / Zoom / Discord)", null, () => {
                    openBroadcastWindow();
                });
                this.addWidget("button", "📸 Capture Pose & Frame to Workflow", null, () => {
                    const offCtx = studioState.offscreenRenderCanvas.getContext("2d");
                    renderAvatarScene(offCtx, 512, 512);
                    syncToBackend();
                });
            }

            // Add Embedded Interactive 3D Viewport Canvas Widget
            const viewportWidget = {
                name: "ziggart_vrm_viewport",
                type: "custom_canvas",
                y: 0,
                computeSize: (width) => [width, 250],
                draw: function (ctx, _node, widgetWidth, y) {
                    this.y = y;
                    const pad = 10;
                    const cw = Math.max(160, widgetWidth - pad * 2);
                    const ch = 232;
                    const ox = pad;
                    const oy = y + 6;
                    this._rect = { ox, oy, cw, ch };

                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(ox, oy, cw, ch);
                    ctx.clip();
                    ctx.translate(ox, oy);

                    renderAvatarScene.call(this, ctx, cw, ch);

                    // HUD Status Overlay
                    ctx.fillStyle = "rgba(15, 18, 28, 0.78)";
                    ctx.fillRect(6, 6, cw - 12, 22);
                    ctx.fillStyle = studioState.liveWebcamActive ? "#4ade80" : "#60cdff";
                    ctx.font = "bold 11px sans-serif";
                    const statusTxt = studioState.liveWebcamActive
                        ? "● LIVE WEBCAM MOCAP (60 FPS) — Move head, speak, or raise hands!"
                        : `3D Pose Editor | Bone: [${studioState.selectedBone}] (Drag joint to rotate, Drag bg to orbit)`;
                    ctx.fillText(statusTxt, 12, 21);

                    ctx.restore();
                    ctx.strokeStyle = studioState.liveWebcamActive ? "#4ade80" : "#34394c";
                    ctx.lineWidth = 1.5;
                    ctx.strokeRect(ox, oy, cw, ch);
                },
                mouse: function (event, pos, _node) {
                    if (!this._rect) return false;
                    const { ox, oy, cw, ch } = this._rect;
                    const lx = pos[0] - ox;
                    const ly = pos[1] - oy;
                    if (lx < 0 || lx > cw || ly < 0 || ly > ch) {
                        if (event.type === "pointerup") {
                            studioState.draggingBone = false;
                            studioState.orbitDragging = false;
                        }
                        return false;
                    }

                    if (event.type === "pointerdown") {
                        studioState.lastMouse = [lx, ly];
                        // Check if user clicked on a bone gizmo handle
                        const map = this._boneScreenMap || {};
                        for (const [bName, sp] of Object.entries(map)) {
                            if (Math.hypot(lx - sp[0], ly - sp[1]) <= 14) {
                                studioState.selectedBone = bName;
                                studioState.draggingBone = true;
                                app.graph?.setDirtyCanvas?.(true, false);
                                return true;
                            }
                        }
                        studioState.orbitDragging = true;
                        return true;
                    } else if (event.type === "pointermove") {
                        const dx = lx - studioState.lastMouse[0];
                        const dy = ly - studioState.lastMouse[1];
                        studioState.lastMouse = [lx, ly];

                        if (studioState.draggingBone) {
                            const b = studioState.selectedBone;
                            const cur = studioState.rotations[b] || [0, 0, 0];
                            studioState.rotations[b] = [
                                Math.max(-90, Math.min(90, cur[0] + dy * 0.6)),
                                Math.max(-90, Math.min(90, cur[1] + dx * 0.6)),
                                cur[2],
                            ];
                            if (b === "head") {
                                if (getW("head_pitch")) getW("head_pitch").value = Math.round(studioState.rotations.head[0]);
                                if (getW("head_yaw")) getW("head_yaw").value = Math.round(studioState.rotations.head[1]);
                            }
                            app.graph?.setDirtyCanvas?.(true, false);
                            return true;
                        } else if (studioState.orbitDragging) {
                            const azW = getW("camera_azimuth");
                            const elW = getW("camera_elevation");
                            if (azW) azW.value = Math.round(((parseFloat(azW.value) || 0) + dx * 0.8) % 360);
                            if (elW) elW.value = Math.max(-60, Math.min(60, Math.round((parseFloat(elW.value) || 0) + dy * 0.5)));
                            app.graph?.setDirtyCanvas?.(true, false);
                            return true;
                        }
                    } else if (event.type === "pointerup") {
                        if (studioState.draggingBone || studioState.orbitDragging) {
                            studioState.draggingBone = false;
                            studioState.orbitDragging = false;
                            const offCtx = studioState.offscreenRenderCanvas.getContext("2d");
                            renderAvatarScene(offCtx, 512, 512);
                            syncToBackend();
                            return true;
                        }
                    }
                    return false;
                },
            };

            if (typeof this.addCustomWidget === "function") {
                this.addCustomWidget(viewportWidget);
            }
            return r;
        };
    },
});

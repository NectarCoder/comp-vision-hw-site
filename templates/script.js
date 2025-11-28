const initialActiveTab = document.querySelector('.tab-content.active');
let activeTabId = initialActiveTab ? initialActiveTab.id : null;
const tabChangeListeners = [];

function registerTabChangeListener(fn) {
    if (typeof fn === 'function') {
        tabChangeListeners.push(fn);
    }
}

function notifyTabChangeListeners(prev, next) {
    tabChangeListeners.forEach((fn) => {
        try {
            fn(prev, next);
        } catch (err) {
            console.error('Tab change callback failed', err);
        }
    });
}

function switchTab(tabId) {
    const previousTab = activeTabId;
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    document.getElementById(tabId).classList.add('active');

    const buttons = Array.from(document.querySelectorAll('.tab-btn'));
    buttons.forEach(btn => {
        const isTarget = btn.getAttribute('onclick') && btn.getAttribute('onclick').includes(tabId);
        if (isTarget) {
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            btn.setAttribute('tabindex', '0');
            try { btn.focus(); } catch (e) { }
        } else {
            btn.classList.remove('active');
            btn.setAttribute('aria-selected', 'false');
            btn.setAttribute('tabindex', '-1');
        }
    });

    activeTabId = tabId;
    if (previousTab !== tabId) {
        notifyTabChangeListeners(previousTab, tabId);
    }
}

(function attachTabKeyboard() {
    const tabButtons = Array.from(document.querySelectorAll('.tab-btn'));
    if (!tabButtons.length) return;

    function focusTab(index) {
        const btn = tabButtons[index];
        if (!btn) return;
        btn.click();
    }

    tabButtons.forEach((btn, idx) => {
        btn.addEventListener('keydown', (e) => {
            const key = e.key;
            if (key === 'ArrowRight') {
                e.preventDefault();
                focusTab((idx + 1) % tabButtons.length);
            } else if (key === 'ArrowLeft') {
                e.preventDefault();
                focusTab((idx - 1 + tabButtons.length) % tabButtons.length);
            } else if (key === 'Home') {
                e.preventDefault();
                focusTab(0);
            } else if (key === 'End') {
                e.preventDefault();
                focusTab(tabButtons.length - 1);
            } else if (key === 'Enter' || key === ' ') {
                e.preventDefault();
                btn.click();
            }
        });
    });
})();

/* ---------- Source explorer (collapsed panels) ---------- */
(function () {
    function togglePanel(toggleBtn) {
        const panelId = toggleBtn.getAttribute('aria-controls');
        const panel = document.getElementById(panelId);
        const expanded = toggleBtn.getAttribute('aria-expanded') === 'true';

        toggleBtn.setAttribute('aria-expanded', String(!expanded));

        // Support both source and video toggle buttons. Choose the
        // correct label text depending on the button type.
        const isVideo = toggleBtn.classList.contains('video-toggle');
        const isInstructions = toggleBtn.classList.contains('instructions-toggle');
        // toggle open state using a CSS class so transitions (opacity/height)
        // can animate. Keep ARIA attributes in sync.
        if (expanded) {
            panel.classList.remove('open');
            panel.setAttribute('aria-hidden', 'true');
            toggleBtn.setAttribute('aria-expanded', 'false');
        } else {
            panel.classList.add('open');
            panel.setAttribute('aria-hidden', 'false');
            toggleBtn.setAttribute('aria-expanded', 'true');
        }

        let collapsedLabel = 'View CLI source code ▾';
        let expandedLabel = 'View CLI source code ▴';
        if (isVideo) {
            collapsedLabel = 'Watch the video ▾';
            expandedLabel = 'Watch the video ▴';
        } else if (isInstructions) {
            collapsedLabel = 'View module instructions ▾';
            expandedLabel = 'View module instructions ▴';
        }

        toggleBtn.innerText = expanded ? collapsedLabel : expandedLabel;
    }

    const LANGUAGE_MAP = {
        py: 'python',
        txt: 'plaintext',
        md: 'markdown'
    };

    function cleanupLanguageClasses(el) {
        el.classList.remove('hljs');
        Array.from(el.classList)
            .filter(cls => cls.startsWith('language-'))
            .forEach(cls => el.classList.remove(cls));
    }

    function handleFileSelection(btn) {
        // Fetch the file from /source/<filename> and load it into the code pane.
        const moduleId = btn.dataset.module; // e.g. "a1"
        const codeEl = document.querySelector(`#source-code-${moduleId} code`);
        if (!codeEl) return;

        const sourcePath = btn.dataset.sourcePath;
        if (!sourcePath) {
            codeEl.textContent = '[Error] No source path configured for this file.';
            return;
        }

        // Visual loading state
        cleanupLanguageClasses(codeEl);
        codeEl.textContent = 'Loading source…';
        codeEl.parentElement.scrollIntoView({ behavior: 'smooth', block: 'start' });

        fetch(`/source/${encodeURIComponent(sourcePath)}`)
            .then(resp => {
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                return resp.text();
            })
            .then(text => {
                const extMatch = sourcePath.split('.').pop()?.toLowerCase() || '';
                const mappedLanguage = LANGUAGE_MAP[extMatch];

                cleanupLanguageClasses(codeEl);
                const highlightAvailable = typeof window !== 'undefined' && window.hljs;

                if (highlightAvailable && (mappedLanguage ? hljs.getLanguage(mappedLanguage) : true)) {
                    try {
                        let highlighted;
                        if (mappedLanguage && hljs.getLanguage(mappedLanguage)) {
                            highlighted = hljs.highlight(text, { language: mappedLanguage, ignoreIllegals: true }).value;
                            codeEl.innerHTML = highlighted;
                            codeEl.classList.add('hljs', `language-${mappedLanguage}`);
                        } else if (hljs.highlightAuto) {
                            highlighted = hljs.highlightAuto(text).value;
                            codeEl.innerHTML = highlighted;
                            codeEl.classList.add('hljs');
                        } else {
                            codeEl.textContent = text;
                        }
                    } catch (err) {
                        console.warn('Highlight.js failed, showing plain text.', err);
                        codeEl.textContent = text;
                    }
                } else {
                    codeEl.textContent = text;
                }

                // keep the loaded code visible and scroll to the top
                codeEl.parentElement.scrollTop = 0;

                // Populate the raw-view and download links in the same source panel
                try {
                    const panel = codeEl.closest('.source-panel');
                    if (panel) {
                        const viewBtn = panel.querySelector('.view-raw-btn');
                        const downloadBtn = panel.querySelector('.download-btn');
                        const rawUrl = `/source/${encodeURI(sourcePath)}`; // preserve path separators
                        if (viewBtn) {
                            viewBtn.href = rawUrl;
                            viewBtn.removeAttribute('aria-disabled');
                            viewBtn.removeAttribute('hidden');
                            const toolbar = panel.querySelector('.source-toolbar');
                            if (toolbar) {
                                toolbar.removeAttribute('hidden');
                                toolbar.setAttribute('aria-hidden', 'false');
                            }
                        }
                        if (downloadBtn) {
                            // Add a query param to force the server to send Content-Disposition
                            downloadBtn.href = rawUrl + '?download=1';
                            downloadBtn.removeAttribute('aria-disabled');
                            downloadBtn.removeAttribute('hidden');
                        }
                    }
                } catch (err) {
                    // non-fatal; toolbar actions are optional
                    console.warn('Failed to enable source toolbar', err);
                }
            })
            .catch(err => {
                codeEl.textContent = `[Error] Could not load source: ${err.message}`;
                try {
                    const panel = codeEl.closest('.source-panel');
                    if (panel) {
                        const toolbar = panel.querySelector('.source-toolbar');
                        if (toolbar) {
                            toolbar.setAttribute('hidden', '');
                            toolbar.setAttribute('aria-hidden', 'true');
                        }
                    }
                } catch (err) { /* ignore */ }
            });
    }

    document.addEventListener('click', (e) => {
        const t = e.target;
        if (t.classList && (t.classList.contains('source-toggle') || t.classList.contains('video-toggle') || t.classList.contains('instructions-toggle'))) {
            togglePanel(t);
            return;
        }

        if (t.classList && t.classList.contains('source-file')) {
            handleFileSelection(t);
            return;
        }
    });

    // keyboard support for toggles & files (Enter/Space)
    document.addEventListener('keydown', (e) => {
        const target = e.target;
        if (!target) return;

        if (target.classList && (target.classList.contains('source-toggle') || target.classList.contains('video-toggle') || target.classList.contains('instructions-toggle'))) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                togglePanel(target);
            }
        }

        if (target.classList && target.classList.contains('source-file')) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                handleFileSelection(target);
            }
        }
    });

})();

async function runModule(id) {
    const inputElem = document.getElementById(`input-${id}`);
    const outputElem = document.getElementById(`output-${id}`);
    const textData = inputElem.value;

    outputElem.innerText = "Processing on Azure VM...";

    try {
        const response = await fetch(`/api/${id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: textData })
        });

        const data = await response.json();

        if (response.ok) {
            outputElem.innerText = data.result;
        } else {
            outputElem.innerText = "Error: " + (data.error || "Unknown server error");
        }
    } catch (error) {
        outputElem.innerText = "Network Error: Could not connect to Flask backend.\nIs app.py running?";
        console.error(error);
    }
}

document.addEventListener('DOMContentLoaded', initModule2Flow);
document.addEventListener('DOMContentLoaded', initModule2Part1Flow);
document.addEventListener('DOMContentLoaded', initModule2Part3Flow);
document.addEventListener('DOMContentLoaded', initModule4Flow);

(function () {
    const themeToggle = document.getElementById('theme-toggle');
    const iconEl = document.getElementById('theme-icon');

    const sunSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M12 4V2M12 22v-2M20 12h2M2 12h2M18.36 5.64l1.41-1.41M4.22 19.78l1.41-1.41M18.36 18.36l1.41 1.41M4.22 4.22l1.41 1.41M12 8a4 4 0 100 8 4 4 0 000-8z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    const moonSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';

    function updateIcon(theme) {
        if (theme === 'dark') {
            iconEl.innerHTML = sunSvg;
            themeToggle.setAttribute('aria-pressed', 'true');
        } else {
            iconEl.innerHTML = moonSvg;
            themeToggle.setAttribute('aria-pressed', 'false');
        }
    }

    function applyTheme(theme) {
        if (theme === 'dark') document.body.classList.add('dark');
        else document.body.classList.remove('dark');
        updateIcon(theme);
    }

    const stored = localStorage.getItem('site-theme');
    const initial = stored || 'dark';
    applyTheme(initial);

    themeToggle.addEventListener('click', () => {
        const isDark = document.body.classList.toggle('dark');
        const nextTheme = isDark ? 'dark' : 'light';
        localStorage.setItem('site-theme', nextTheme);
        updateIcon(nextTheme);
    });
})();

document.addEventListener('DOMContentLoaded', initModule1Flow);
document.addEventListener('DOMContentLoaded', initModule56Flow);
document.addEventListener('DOMContentLoaded', initModule56Part2Showcase);

function initModule56Flow() {
    const form = document.getElementById('module5-6-part1-form');
    if (!form) return;

    const fileInput = document.getElementById('module5-6-video-input');
    const useSampleCheckbox = document.getElementById('module5-6-use-sample');
    const runBtn = document.getElementById('module5-6-run-btn');
    const resetBtn = document.getElementById('module5-6-reset-btn');
    const statusLine = document.getElementById('module5-6-part1-status');
    const inputVideo = document.getElementById('module5-6-input-video');
    const inputPlaceholder = document.getElementById('module5-6-input-placeholder');
    const outputVideo = document.getElementById('module5-6-output-video');
    const outputPlaceholder = document.getElementById('module5-6-output-placeholder');
    const dictSelect = document.getElementById('module5-6-dict');
    const paddingInput = document.getElementById('module5-6-padding');

    if (!fileInput || !runBtn || !resetBtn || !statusLine || !inputVideo || !outputVideo) return;

    let hasResults = false;
    let createdBlobUrl = null;
    let samplePreviewToken = 0;

    const setStatus = (message, variant = 'info') => {
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const clearVideoElement = (vidEl, placeholder) => {
        if (!vidEl) return;
        try { vidEl.pause(); } catch (e) { /* ignore invalid states */ }
        try { vidEl.removeAttribute('src'); } catch (e) { /* ignore */ }
        // Remove any dynamically injected <source> tags so the element fully resets
        try {
            Array.from(vidEl.querySelectorAll('source')).forEach(src => src.remove());
        } catch (e) { /* ignore */ }
        try { vidEl.load(); } catch (e) { /* ignore */ }
        vidEl.hidden = true;
        if (placeholder) placeholder.hidden = false;
    };

    const showVideo = (vidEl, placeholder, dataUrlOrBlobUrl) => {
        if (!dataUrlOrBlobUrl) {
            clearVideoElement(vidEl, placeholder);
            return;
        }
        // set a proper <source> child so the MIME type can be respected
        try {
            // Make sure we clear previous sources
            Array.from(vidEl.querySelectorAll('source')).forEach(s => s.remove());
            // clear previous inline src if present
            try { vidEl.removeAttribute('src'); } catch (e) { /* ignore */ }
            const srcEl = document.createElement('source');
            srcEl.src = dataUrlOrBlobUrl;
            // if this is a data URL, we can extract the mime
            if (typeof dataUrlOrBlobUrl === 'string' && dataUrlOrBlobUrl.startsWith('data:')) {
                const mime = dataUrlOrBlobUrl.split(';')[0].replace('data:', '') || 'video/mp4';
                srcEl.type = mime;
            }
            vidEl.appendChild(srcEl);
        } catch (err) {
            // fall back to setting src directly
            try { vidEl.src = dataUrlOrBlobUrl; } catch (_) { /* ignore */ }
        }
        vidEl.hidden = false;
        if (placeholder) placeholder.hidden = true;
        try { vidEl.load(); } catch (e) { /* ignore */ }
    };

    const hasSelected = () => Boolean((fileInput.files && fileInput.files.length) || (useSampleCheckbox && useSampleCheckbox.checked));

    fileInput.addEventListener('change', () => {
        const file = fileInput.files?.[0];
        if (!file) {
            runBtn.disabled = !hasSelected();
            return;
        }
        // create a local blob url for preview
        const url = URL.createObjectURL(file);
        showVideo(inputVideo, inputPlaceholder, url);
        runBtn.disabled = false;
        resetBtn.disabled = false;
        setStatus('Ready to run the tracker on the selected video.', 'info');
    });

    if (useSampleCheckbox) {
        useSampleCheckbox.addEventListener('change', async () => {
            const checked = useSampleCheckbox.checked;
            const requestToken = ++samplePreviewToken;

            if (checked) {
                fileInput.value = '';
                fileInput.disabled = true;
                runBtn.disabled = false;
                resetBtn.disabled = false;
                setStatus('Loading sample video preview…', 'info');
                try {
                    const resp = await fetch('/api/a56/part1/sample');
                    const data = await resp.json();
                    if (!resp.ok) throw new Error(data.error || 'Failed to load sample');
                    if (requestToken !== samplePreviewToken) return; // user toggled off mid-fetch
                    showVideo(inputVideo, inputPlaceholder, data.video);
                    setStatus(`Using sample ${data.filename}. Ready to run.`, 'info');
                } catch (err) {
                    if (requestToken !== samplePreviewToken) return;
                    fileInput.disabled = false;
                    useSampleCheckbox.checked = false;
                    setStatus(`Could not load sample: ${err.message}`, 'error');
                }
            } else {
                fileInput.disabled = false;
                clearVideoElement(inputVideo, inputPlaceholder);
                runBtn.disabled = !hasSelected();
                resetBtn.disabled = !hasResults;
                setStatus('Choose a video to get started.', 'info');
            }
        });
    }

        resetBtn.addEventListener('click', () => {
        form.reset();
        clearVideoElement(inputVideo, inputPlaceholder);
        clearVideoElement(outputVideo, outputPlaceholder);
        runBtn.disabled = true;
        resetBtn.disabled = true;
        hasResults = false;
            // hide download link and revoke any created blob url
            const dlArea = document.getElementById('module5-6-download-area');
            const dlLink = document.getElementById('module5-6-download-link');
            if (dlArea) dlArea.style.display = 'none';
            if (dlLink) dlLink.href = '#';
            if (createdBlobUrl) {
                try { URL.revokeObjectURL(createdBlobUrl); } catch (_) { /* ignore */ }
                createdBlobUrl = null;
            }
        setStatus('Choose a video to get started.', 'info');
    });

    form.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        if (!hasSelected()) {
            setStatus('Please select a video or choose the example video.', 'error');
            return;
        }

        const formData = new FormData();
        if (useSampleCheckbox && useSampleCheckbox.checked) {
            formData.append('sample', 'aruco-marker.mp4');
        } else {
            formData.append('video', fileInput.files[0]);
        }
        formData.append('dict', dictSelect.value);
        formData.append('padding', paddingInput.value);

        runBtn.disabled = true;
        setStatus('Uploading video and running the ArUco tracker…', 'info');
        clearVideoElement(outputVideo, outputPlaceholder);
        // hide existing download link and revoke any previously created blob
        const dlArea = document.getElementById('module5-6-download-area');
        const dlLink = document.getElementById('module5-6-download-link');
        if (dlArea) dlArea.style.display = 'none';
        if (dlLink) dlLink.href = '#';
        if (createdBlobUrl) {
            try { URL.revokeObjectURL(createdBlobUrl); } catch (_) { /* ignore */ }
            createdBlobUrl = null;
        }

        try {
            const response = await fetch('/api/a56/part1', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Process failed');

            if (data.outputVideo) {
                // If we received a data URL, convert it to a blob URL for more compatibility
                try {
                    let videoUrl = data.outputVideo;
                        if (typeof videoUrl === 'string' && videoUrl.startsWith('data:')) {
                        const parts = videoUrl.split(',');
                        const meta = parts[0];
                        const base64 = parts[1];
                        const mimeMatch = meta.match(/data:([^;]+)/);
                        const mime = mimeMatch ? mimeMatch[1] : 'video/mp4';
                        const binary = atob(base64);
                        const len = binary.length;
                        const u8 = new Uint8Array(len);
                        for (let i = 0; i < len; i++) u8[i] = binary.charCodeAt(i);
                        const blob = new Blob([u8], { type: mime });
                        // revoke a previous blob url if present
                        if (createdBlobUrl) try { URL.revokeObjectURL(createdBlobUrl); } catch (_) { /* ignore */ }
                        videoUrl = URL.createObjectURL(blob);
                        createdBlobUrl = videoUrl;
                        // show the download link for the blob (create an object URL for download) — we'll reuse videoUrl
                        const dlArea = document.getElementById('module5-6-download-area');
                        const dlLink = document.getElementById('module5-6-download-link');
                        if (dlArea && dlLink) {
                            dlLink.href = videoUrl;
                            dlLink.download = data.outputFilename || 'processed.mp4';
                            dlArea.style.display = 'block';
                        }
                    }
                    showVideo(outputVideo, outputPlaceholder, videoUrl);
                } catch (err) {
                    console.warn('Failed to convert data URL to blob; using raw URL fallback', err);
                    showVideo(outputVideo, outputPlaceholder, data.outputVideo);
                }
                setStatus(`Process complete for ${data.originalFilename || 'uploaded video'}.`, 'success');
                hasResults = true;
            } else {
                setStatus('Processing completed but no output video was returned.', 'warning');
            }
            resetBtn.disabled = false;
        } catch (err) {
            setStatus(err.message || 'Unexpected error occurred.', 'error');
        } finally {
            runBtn.disabled = false;
        }
    });
}

function initModule56Part2Showcase() {
    const section = document.getElementById('module5-6-part2');
    const originalVideo = document.getElementById('module5-6-part2-original-video');
    const overlayVideo = document.getElementById('module5-6-part2-overlay-video');
    const overlayPlayer = document.getElementById('module5-6-part2-overlay-player');
    const overlayCanvas = document.getElementById('module5-6-part2-overlay-canvas');
    const statusLine = document.getElementById('module5-6-part2-status');
    const reloadBtn = document.getElementById('module5-6-part2-reload');
    const originalPlaceholder = document.getElementById('module5-6-part2-original-placeholder');
    const overlayHint = document.getElementById('module5-6-part2-overlay-hint');
    const frameLabel = document.getElementById('module5-6-part2-frame');
    const boxLabel = document.getElementById('module5-6-part2-box');
    const framesMetric = document.getElementById('module5-6-part2-metric-frames');
    const resolutionMetric = document.getElementById('module5-6-part2-metric-resolution');
    const fpsMetric = document.getElementById('module5-6-part2-metric-fps');
    const durationMetric = document.getElementById('module5-6-part2-metric-duration');
    const maskNote = document.getElementById('module5-6-part2-mask-note');

    if (!section || !originalVideo || !overlayVideo || !overlayCanvas || !statusLine || !overlayPlayer) {
        return;
    }

    const overlayCtx = overlayCanvas.getContext('2d');
    if (!overlayCtx) {
        return;
    }
    const originalFrame = originalVideo.closest('.result-image');
    let boxes = [];
    let fps = 30;
    let videoFrameCount = 0;
    let maskFrameCount = 0;
    let assetsLoaded = false;
    let isLoading = false;
    let overlayReady = false;
    let animationFrameId = null;

    const setStatus = (message, variant = 'info') => {
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const clampFrameIndex = () => {
        if (!boxes.length || !overlayVideo) return 0;
        const idx = Math.round((overlayVideo.currentTime || 0) * fps);
        const maxIdx = boxes.length - 1;
        return Math.min(Math.max(idx, 0), maxIdx >= 0 ? maxIdx : 0);
    };

    const stopAnimation = () => {
        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    };

    const drawFrame = () => {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        if (!overlayReady || !boxes.length) {
            frameLabel.textContent = '--';
            boxLabel.textContent = '--';
            return;
        }

        const idx = clampFrameIndex();
        const entry = boxes[idx] || null;
        const frameTotal = videoFrameCount || boxes.length;
        frameLabel.textContent = frameTotal ? `${idx + 1} / ${frameTotal}` : `${idx + 1}`;

        if (!entry || !entry.box) {
            boxLabel.textContent = 'No mask';
            return;
        }

        const [x1, y1, x2, y2] = entry.box;
        const width = Math.max(0, x2 - x1);
        const height = Math.max(0, y2 - y1);
        const strokeWidth = Math.max(2, overlayCanvas.width * 0.0035);

        overlayCtx.lineWidth = strokeWidth;
        overlayCtx.strokeStyle = '#29d3d3';
        overlayCtx.fillStyle = 'rgba(41, 211, 211, 0.22)';
        overlayCtx.strokeRect(x1, y1, width, height);
        overlayCtx.fillRect(x1, y1, width, height);

        // subtle corner markers for easier spatial awareness
        overlayCtx.strokeStyle = '#0df0ff';
        overlayCtx.lineWidth = Math.max(1, strokeWidth * 0.6);
        const marker = Math.min(24, width * 0.2, height * 0.2);
        if (marker > 0) {
            overlayCtx.beginPath();
            overlayCtx.moveTo(x1, y1 + marker);
            overlayCtx.lineTo(x1, y1);
            overlayCtx.lineTo(x1 + marker, y1);
            overlayCtx.moveTo(x2 - marker, y1);
            overlayCtx.lineTo(x2, y1);
            overlayCtx.lineTo(x2, y1 + marker);
            overlayCtx.moveTo(x1, y2 - marker);
            overlayCtx.lineTo(x1, y2);
            overlayCtx.lineTo(x1 + marker, y2);
            overlayCtx.moveTo(x2 - marker, y2);
            overlayCtx.lineTo(x2, y2);
            overlayCtx.lineTo(x2, y2 - marker);
            overlayCtx.stroke();
        }

        boxLabel.textContent = `${width}px × ${height}px`;
    };

    const startAnimation = () => {
        stopAnimation();
        const tick = () => {
            drawFrame();
            animationFrameId = requestAnimationFrame(tick);
        };
        animationFrameId = requestAnimationFrame(tick);
    };

    const resetVideos = () => {
        stopAnimation();
        overlayReady = false;
        frameLabel.textContent = '--';
        boxLabel.textContent = '--';
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        overlayCanvas.width = overlayCanvas.height = 0;
        overlayVideo.removeAttribute('src');
        originalVideo.removeAttribute('src');
        try { overlayVideo.load(); } catch (e) { /* ignore */ }
        try { originalVideo.load(); } catch (e) { /* ignore */ }
        overlayPlayer.classList.remove('ready');
        if (overlayHint) {
            overlayHint.textContent = 'Loading overlay…';
            overlayHint.classList.remove('ready');
            overlayHint.hidden = false;
        }
        if (originalFrame) originalFrame.dataset.empty = 'true';
        originalVideo.hidden = true;
        if (originalPlaceholder) originalPlaceholder.hidden = false;
    };

    const populateMetrics = (data) => {
        framesMetric.textContent = `${data.maskFrameCount ?? '--'} / ${data.frameCount ?? '--'}`;
        resolutionMetric.textContent = data.frameWidth && data.frameHeight ? `${data.frameWidth} × ${data.frameHeight}` : '-- × --';
        fpsMetric.textContent = data.fps ? `${Number(data.fps).toFixed(2)} fps` : '-- fps';
        durationMetric.textContent = data.durationSeconds ? `${Number(data.durationSeconds).toFixed(2)} s` : '-- s';
        if (maskNote) {
            const maskName = data.maskFilename || 'iphone-moving-masks.npz';
            maskNote.innerHTML = `Masks: <code>${maskName}</code>`;
        }
    };

    const handleLoadedMetadata = () => {
        const width = overlayVideo.videoWidth || 0;
        const height = overlayVideo.videoHeight || 0;
        if (!width || !height) return;
        overlayCanvas.width = width;
        overlayCanvas.height = height;
        overlayReady = true;
        overlayPlayer.classList.add('ready');
        if (overlayHint) {
            overlayHint.textContent = 'Press play to visualize the SAM2 track.';
            overlayHint.classList.add('ready');
            overlayHint.hidden = false;
        }
        drawFrame();
    };

    overlayVideo.addEventListener('loadedmetadata', handleLoadedMetadata);
    overlayVideo.addEventListener('play', () => {
        if (!overlayReady) return;
        startAnimation();
    });
    overlayVideo.addEventListener('pause', stopAnimation);
    overlayVideo.addEventListener('ended', stopAnimation);
    overlayVideo.addEventListener('seeking', () => drawFrame());
    overlayVideo.addEventListener('timeupdate', () => {
        if (overlayVideo.paused) {
            drawFrame();
        }
    });

    if (originalVideo) {
        originalVideo.addEventListener('loadeddata', () => {
            originalVideo.hidden = false;
            if (originalPlaceholder) originalPlaceholder.hidden = true;
            if (originalFrame) originalFrame.dataset.empty = 'false';
        });
    }

    const loadAssets = async () => {
        if (isLoading) return;
        isLoading = true;
        reloadBtn && (reloadBtn.disabled = true);
        setStatus('Loading SAM2 assets…', 'info');
        resetVideos();

        try {
            const response = await fetch('/api/a56/part2/assets');
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to load assets');

            boxes = Array.isArray(data.boxes) ? data.boxes : [];
            fps = Number(data.fps) || 30;
            videoFrameCount = Number(data.frameCount) || boxes.length;
            maskFrameCount = Number(data.maskFrameCount) || boxes.length;
            populateMetrics(data);

            const videoUrl = data.videoUrl;
            if (!videoUrl) {
                throw new Error('Video URL missing in response');
            }

            overlayVideo.src = videoUrl;
            originalVideo.src = videoUrl;
            overlayVideo.load();
            originalVideo.load();
            assetsLoaded = true;
            setStatus('Assets ready. Scrub or press play to inspect the tracker.', 'success');
        } catch (err) {
            boxes = [];
            assetsLoaded = false;
            setStatus(err.message || 'Failed to load Module 5 & 6 Part 2 assets.', 'error');
        } finally {
            isLoading = false;
            if (reloadBtn) reloadBtn.disabled = false;
        }
    };

    reloadBtn && reloadBtn.addEventListener('click', loadAssets);

    registerTabChangeListener((prev, next) => {
        if (prev === 'a56' && overlayVideo) {
            try { overlayVideo.pause(); } catch (e) { /* ignore */ }
        }
        if (next === 'a56' && !assetsLoaded && !isLoading) {
            loadAssets();
        }
    });

    loadAssets();
}

/*
 * Global reset button - clears state across modules and UI.
 * Click the existing reset/clear buttons (if available) so module-specific
 * handlers run, then fall back to explicit form resets and cleanup.
 */
function initGlobalReset() {
    const globalResetBtn = document.getElementById('global-reset-btn');
    if (!globalResetBtn) return;

    globalResetBtn.addEventListener('click', async () => {
        const confirmed = confirm('Reset all modules and clear results? This will clear images, points, and outputs across every module.');
        if (!confirmed) return;

        // Module 1: reuse existing reset handler by clicking the button
        const m1Reset = document.getElementById('module1-reset');
        if (m1Reset) try { m1Reset.click(); } catch (err) { /* ignore */ }

        // Module 2 Part 1: clear UI + server results
        const m2part1Clear = document.getElementById('module2-part1-clear-btn');
        if (m2part1Clear && !m2part1Clear.disabled) {
            try { m2part1Clear.click(); } catch (err) { /* ignore */ }
        } else if (m2part1Clear) {
            // If the button exists but is disabled, still attempt to clear server-side results
            try { await fetch('/api/a2/part1/results', { method: 'DELETE' }); } catch (_) { /* ignore errors */ }
        }

        // Module 2 Part 2: clear form and outputs
        const m2ResetBtn = document.getElementById('module2-reset-btn');
        if (m2ResetBtn && !m2ResetBtn.disabled) {
            try { m2ResetBtn.click(); } catch (err) { /* ignore */ }
        } else {
            // fallback to reset the form elements
            const f2 = document.getElementById('module2-part2-form');
            if (f2 && typeof f2.reset === 'function') f2.reset();
            const original = document.getElementById('module2-original-preview');
            const blurred = document.getElementById('module2-blurred-preview');
            const restored = document.getElementById('module2-restored-preview');
            if (original) { original.src = ''; original.hidden = true; }
            if (blurred) { blurred.src = ''; blurred.hidden = true; }
            if (restored) { restored.src = ''; restored.hidden = true; }
        }

        // Module 2 Part 3: clear results
        const m2part3Reset = document.getElementById('module2-part3-reset-btn');
        if (m2part3Reset && !m2part3Reset.disabled) {
            try { m2part3Reset.click(); } catch (err) { /* ignore */ }
        } else {
            const f3 = document.getElementById('module2-part3-form');
            if (f3 && typeof f3.reset === 'function') f3.reset();
            const detImg = document.getElementById('module2-part3-detections');
            const blurImg = document.getElementById('module2-part3-blurred');
            const log = document.getElementById('module2-part3-detection-log');
            const count = document.getElementById('module2-part3-detection-count');
            const templatesGrid = document.getElementById('module2-part3-templates-grid');
            if (detImg) { detImg.src = ''; detImg.hidden = true; }
            if (blurImg) { blurImg.src = ''; blurImg.hidden = true; }
            if (log) { log.innerHTML = ''; log.dataset.empty = 'true'; }
            if (count) { count.textContent = '0'; }
            if (templatesGrid) { templatesGrid.innerHTML = ''; }
        }

        ['module4-part1-form', 'module4-part2-form'].forEach((formId) => {
            const formEl = document.getElementById(formId);
            if (formEl && typeof formEl.__module4Reset === 'function') {
                try { formEl.__module4Reset({ clearServer: true }); } catch (err) { /* ignore */ }
            }
        });

        // Clear any run outputs in simple modules (3, 4, 5-6, 7) and additional UI elements
        ['a3', 'a4', 'a56', 'a7'].forEach((id) => {
            const input = document.getElementById(`input-${id}`);
            const output = document.getElementById(`output-${id}`);
            if (input && 'value' in input) input.value = '';
            if (output && 'textContent' in output) output.textContent = '// Output will appear here...';
        });

        // Also clear Module 5 & 6 specific video inputs and playback elements
        const m56File = document.getElementById('module5-6-video-input');
        const m56InVideo = document.getElementById('module5-6-input-video');
        const m56OutVideo = document.getElementById('module5-6-output-video');
        const m56InPlaceholder = document.getElementById('module5-6-input-placeholder');
        const m56OutPlaceholder = document.getElementById('module5-6-output-placeholder');
        if (m56File) m56File.value = '';
        if (m56InVideo) {
            try { m56InVideo.pause(); m56InVideo.removeAttribute('src'); m56InVideo.load(); } catch (e) { /* ignore */ }
            if (m56InPlaceholder) { m56InPlaceholder.hidden = false; }
            m56InVideo.hidden = true;
        }
        if (m56OutVideo) {
            try { m56OutVideo.pause(); m56OutVideo.removeAttribute('src'); m56OutVideo.load(); } catch (e) { /* ignore */ }
            if (m56OutPlaceholder) { m56OutPlaceholder.hidden = false; }
            m56OutVideo.hidden = true;
        }
        // hide download area
        const m56Download = document.getElementById('module5-6-download-area');
        if (m56Download) m56Download.style.display = 'none';
        const m56UseSample = document.getElementById('module5-6-use-sample');
        if (m56UseSample) {
            try { m56UseSample.checked = false; } catch (e) { /* ignore */ }
            try { m56File.disabled = false; } catch (e) { /* ignore */ }
        }

        // Ensure Module 1 inputs are cleared (extra cleanup to cover all cases)
        const refFile = document.getElementById('ref-image-input');
        const testFile = document.getElementById('test-image-input');
        if (refFile) refFile.value = '';
        if (testFile) testFile.value = '';
        // Remove any 'autofilled' highlights that may remain
        document.querySelectorAll('.autofilled').forEach(el => el.classList.remove('autofilled'));
        // reset test expected value hint if present
        const expectedEl = document.getElementById('test-expected-value');
        if (expectedEl) expectedEl.textContent = '--';

        // Provide a small visual confirmation in the current tab (call the status areas)
        try { document.getElementById('ref-status').textContent = 'All modules reset.'; document.getElementById('test-status').textContent = 'All modules reset.'; } catch(e) { }
    });
}

document.addEventListener('DOMContentLoaded', initGlobalReset);

function initModule2Flow() {
    const form = document.getElementById('module2-part2-form');
    const fileInput = document.getElementById('module2-image-input');
    const useSampleCheckbox = document.getElementById('module2-use-sample');
    const runBtn = document.getElementById('module2-run-btn');
    const resetBtn = document.getElementById('module2-reset-btn');
    const statusLine = document.getElementById('module2-part2-status');
    const originalImg = document.getElementById('module2-original-preview');
    const originalPlaceholder = document.getElementById('module2-original-placeholder');
    const blurredImg = document.getElementById('module2-blurred-preview');
    const restoredImg = document.getElementById('module2-restored-preview');
    const blurredPlaceholder = document.getElementById('module2-blurred-placeholder');
    const restoredPlaceholder = document.getElementById('module2-restored-placeholder');

    if (!form || !fileInput || !runBtn || !resetBtn || !statusLine || !blurredImg || !restoredImg || !originalImg) {
        return;
    }

    const resultFrames = [
        { img: originalImg, placeholder: originalPlaceholder, isOriginal: true },
        { img: blurredImg, placeholder: blurredPlaceholder, isOriginal: false },
        { img: restoredImg, placeholder: restoredPlaceholder, isOriginal: false }
    ];

    const hasFileSelected = () => Boolean((fileInput.files && fileInput.files.length) || (useSampleCheckbox && useSampleCheckbox.checked));
    const hasResults = () => resultFrames.some(({ img, isOriginal }) => !isOriginal && img.dataset.loaded === 'true');

    const setStatus = (message, variant = 'info') => {
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const showImage = (imgEl, placeholderEl, dataUrl) => {
        if (!imgEl) return;
        imgEl.src = dataUrl;
        imgEl.hidden = false;
        imgEl.dataset.loaded = 'true';
        const wrapper = imgEl.closest('.result-image');
        if (wrapper) wrapper.dataset.empty = 'false';
        if (placeholderEl) placeholderEl.hidden = true;
    };

    const clearResults = (opts = { keepOriginal: false }) => {
        resultFrames.forEach(({ img, placeholder, isOriginal }) => {
            if (isOriginal && opts.keepOriginal) {
                return;
            }
            if (!img) return;
            img.removeAttribute('src');
            img.hidden = true;
            img.dataset.loaded = 'false';
            const wrapper = img.closest('.result-image');
            if (wrapper) wrapper.dataset.empty = 'true';
            if (placeholder) {
                placeholder.hidden = false;
                placeholder.textContent = isOriginal ? 'No image selected.' : 'No output yet.';
            }
        });
    };

    const showOriginal = (file) => {
        if (!file) {
            clearResults({ keepOriginal: false });
            return;
        }
        const reader = new FileReader();
        reader.onload = (evt) => {
            showImage(originalImg, originalPlaceholder, evt.target?.result || '');
        };
        reader.onerror = () => {
            clearResults({ keepOriginal: false });
            setStatus('Could not preview that image locally.', 'error');
        };
        reader.readAsDataURL(file);
    };

    const showOriginalFromUrl = (dataUrl) => {
        if (!dataUrl) {
            clearResults();
            return;
        }
        showImage(originalImg, originalPlaceholder, dataUrl);
    };

    clearResults();
    setStatus('Choose an image to get started.', 'info');

    fileInput.addEventListener('change', () => {
        const hasFile = hasFileSelected();
        runBtn.disabled = !hasFile;
        resetBtn.disabled = !hasFile && !hasResults();
        if (hasFile) {
            // If the source is a file, prefer that preview
            if (fileInput.files && fileInput.files.length) {
                showOriginal(fileInput.files[0]);
            }
            setStatus('Ready to run the process.', 'info');
        } else if (!hasResults()) {
            clearResults();
            setStatus('Choose an image to get started.', 'info');
        }
    });

    if (useSampleCheckbox) {
        useSampleCheckbox.addEventListener('change', async () => {
            const checked = useSampleCheckbox.checked;
            if (checked) {
                // disable file input
                fileInput.value = '';
                fileInput.disabled = true;
                resetBtn.disabled = false;
                // fetch the sample preview
                setStatus('Loading example image preview…', 'info');
                try {
                    const resp = await fetch('/api/a2/part2/sample');
                    const data = await resp.json();
                    if (!resp.ok) throw new Error(data.error || 'Failed to load sample');
                    showOriginalFromUrl(data.image);
                    setStatus(`Using sample ${data.filename}. Ready to run.`, 'info');
                    runBtn.disabled = false;
                } catch (err) {
                    fileInput.disabled = false;
                    useSampleCheckbox.checked = false;
                    setStatus(`Could not load sample: ${err.message}`, 'error');
                }
            } else {
                fileInput.disabled = false;
                clearResults();
                setStatus('Choose an image to get started.', 'info');
            }
        });
    }

    resetBtn.addEventListener('click', () => {
        form.reset();
        clearResults();
        runBtn.disabled = true;
        resetBtn.disabled = true;
        if (useSampleCheckbox) useSampleCheckbox.checked = false;
        if (fileInput) fileInput.disabled = false;
        setStatus('Choose an image to get started.', 'info');
    });

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!hasFileSelected()) {
            setStatus('Please select an image before running the process.', 'error');
            return;
        }

        const formData = new FormData();
        if (useSampleCheckbox && useSampleCheckbox.checked) {
            // signal server to use the pre-configured sample
            formData.append('sample', 'tree.jpg');
        } else {
            formData.append('image', fileInput.files[0]);
        }

        runBtn.disabled = true;
        resetBtn.disabled = true;
        setStatus('Uploading image and running the Gaussian blur + inverse filtering process...', 'info');

        try {
            const response = await fetch('/api/a2/part2', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Process failed');
            }

            showImage(blurredImg, blurredPlaceholder, data.blurredImage);
            showImage(restoredImg, restoredPlaceholder, data.restoredImage);
            resetBtn.disabled = false;
            const extraBits = [];
            if (Number.isFinite(data.kernelSize)) {
                extraBits.push(`kernel ${data.kernelSize}`);
            }
            if (Number.isFinite(data.sigma)) {
                extraBits.push(`sigma ${Number(data.sigma).toFixed(2)}`);
            }
            const meta = extraBits.length ? ` (${extraBits.join(', ')})` : '';
            setStatus(`Process complete for ${data.originalFilename || 'uploaded image'}${meta}.`, 'success');
        } catch (err) {
            clearResults({ keepOriginal: true });
            setStatus(err.message || 'Unexpected error occurred.', 'error');
        } finally {
            if (hasFileSelected()) {
                runBtn.disabled = false;
            }
            if (hasFileSelected() || hasResults()) {
                resetBtn.disabled = false;
            }
        }
    });
}

function initModule2Part1Flow() {
    const form = document.getElementById('module2-part1-form');
    if (!form) return;

    const thresholdInput = document.getElementById('module2-part1-threshold');
    const runBtn = document.getElementById('module2-part1-run-btn');
    const clearBtn = document.getElementById('module2-part1-clear-btn');
    const statusLine = document.getElementById('module2-part1-status');
    const sceneImg = document.getElementById('module2-part1-scene');
    const scenePlaceholder = document.getElementById('module2-part1-scene-placeholder');
    const gallery = document.getElementById('module2-part1-gallery');
    const galleryPlaceholder = document.getElementById('module2-part1-gallery-placeholder');
    const matchCountEl = document.getElementById('module2-part1-match-count');
    const totalCountEl = document.getElementById('module2-part1-total');

    if (!thresholdInput || !runBtn || !clearBtn || !statusLine || !gallery || !matchCountEl || !totalCountEl) {
        return;
    }

    let sceneLoaded = false;
    let hasResults = false;

    const setStatus = (message, variant = 'info') => {
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const setSceneImage = (dataUrl, alt = 'Reference scene for template matching') => {
        if (!sceneImg) return;
        if (dataUrl) {
            sceneImg.src = dataUrl;
            sceneImg.alt = alt;
            sceneImg.hidden = false;
            sceneImg.removeAttribute('aria-hidden');
            scenePlaceholder && (scenePlaceholder.hidden = true);
            const wrapper = sceneImg.closest('.result-image');
            if (wrapper) wrapper.dataset.empty = 'false';
            sceneLoaded = true;
        } else {
            sceneImg.hidden = true;
            if (scenePlaceholder) scenePlaceholder.hidden = false;
            const wrapper = sceneImg.closest('.result-image');
            if (wrapper) wrapper.dataset.empty = 'true';
        }
    };

    const templateLabel = (name) => {
        if (!name) return 'Unknown template';
        return name
            .replace(/^template_/i, '')
            .replace(/_/g, ' ')
            .replace(/\.png$/i, '')
            .trim() || name;
    };

    const renderMatches = (matches = []) => {
        gallery.innerHTML = '';
        if (!matches.length) {
            gallery.dataset.empty = 'true';
            if (galleryPlaceholder) galleryPlaceholder.hidden = false;
            return;
        }

        gallery.dataset.empty = 'false';
        if (galleryPlaceholder) galleryPlaceholder.hidden = true;
        matches.forEach((match) => {
            const card = document.createElement('article');
            card.className = 'match-card';
            card.dataset.result = match.matched ? 'hit' : 'miss';

            const header = document.createElement('div');
            header.className = 'match-card__header';

            const labelSpan = document.createElement('span');
            labelSpan.className = 'match-chip';
            labelSpan.textContent = match.matched ? 'Match' : 'No match';

            const title = document.createElement('strong');
            title.textContent = templateLabel(match.template);

            header.appendChild(labelSpan);
            header.appendChild(title);
            card.appendChild(header);

            if (match.matched && match.outputImage) {
                const img = document.createElement('img');
                img.src = match.outputImage;
                img.alt = `Detected ${templateLabel(match.template)}`;
                img.loading = 'lazy';
                img.className = 'match-card__image';
                card.appendChild(img);
            }

            const message = document.createElement('p');
            message.className = 'match-card__message';
            if (match.error) {
                message.textContent = `Error: ${match.error}`;
            } else if (match.matched) {
                message.textContent = 'Correlation above threshold – see highlighted regions.';
            } else {
                message.textContent = 'No detections cleared the chosen threshold.';
            }
            card.appendChild(message);

            gallery.appendChild(card);
        });
    };

    const updateSummary = (matched = 0, total = 0) => {
        matchCountEl.textContent = matched;
        totalCountEl.textContent = total;
    };

    const clearUI = (opts = { resetSummary: true }) => {
        renderMatches([]);
        hasResults = false;
        clearBtn.disabled = true;
        if (opts.resetSummary) {
            updateSummary(0, 0);
        }
    };

    const clearServerResults = async (options = { silent: false, keepalive: false }) => {
        try {
            const usePost = Boolean(options.keepalive);
            const fetchConfig = {
                method: usePost ? 'POST' : 'DELETE',
                keepalive: Boolean(options.keepalive),
            };
            if (usePost) {
                fetchConfig.headers = { 'Content-Type': 'application/json' };
                fetchConfig.body = JSON.stringify({ cleanup: true });
            }
            await fetch('/api/a2/part1/results', fetchConfig);
        } catch (err) {
            if (!options.silent) {
                console.warn('Failed to clear Module 2 Part 1 outputs on server.', err);
            }
        }
    };

    const loadScenePreview = async () => {
        if (sceneLoaded) return;
        try {
            const response = await fetch('/api/a2/part1/scene');
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Unable to load scene.');
            setSceneImage(data.image, `Reference scene (${data.filename || 'scene.jpg'})`);
            if (scenePlaceholder) scenePlaceholder.textContent = 'Scene preview ready.';
        } catch (err) {
            setSceneImage(null);
            setStatus(err.message, 'error');
        }
    };

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const thresholdValue = parseFloat(thresholdInput.value);
        if (Number.isNaN(thresholdValue) || thresholdValue < 0 || thresholdValue > 1) {
            setStatus('Please enter a threshold between 0.0 and 1.0.', 'error');
            return;
        }

        setStatus('Running correlation across all templates…', 'info');
        runBtn.disabled = true;
        thresholdInput.disabled = true;

        try {
            const response = await fetch('/api/a2/part1', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: thresholdValue }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Correlation run failed.');

            if (data.scene?.image) {
                setSceneImage(data.scene.image, `Reference scene (${data.scene.filename || 'scene'})`);
            }
            renderMatches(data.matches || []);
            const matched = data.summary?.matched ?? 0;
            const total = data.summary?.total ?? (data.matches ? data.matches.length : 0);
            updateSummary(matched, total);
            hasResults = true;
            clearBtn.disabled = false;

            if (matched > 0) {
                setStatus(`Detected ${matched} object(s) using threshold ${thresholdValue.toFixed(2)}.`, 'success');
            } else {
                setStatus(`No templates passed the threshold ${thresholdValue.toFixed(2)}.`, 'warning');
            }
        } catch (err) {
            clearUI({ resetSummary: false });
            setStatus(err.message || 'Unexpected error while running correlation.', 'error');
        } finally {
            runBtn.disabled = false;
            thresholdInput.disabled = false;
        }
    });

    clearBtn.addEventListener('click', async () => {
        clearUI({ resetSummary: true });
        await clearServerResults({ silent: true });
        setStatus('Cleared generated match files.', 'info');
    });

    registerTabChangeListener((prev, next) => {
        if (prev === 'a2' && next !== 'a2') {
            clearUI({ resetSummary: true });
            clearServerResults({ silent: true });
        } else if (next === 'a2') {
            loadScenePreview();
        }
    });

    window.addEventListener('beforeunload', () => {
        clearServerResults({ silent: true, keepalive: true });
    });

    loadScenePreview();
}

function initModule2Part3Flow() {
    const form = document.getElementById('module2-part3-form');
    if (!form) return;

    const thresholdInput = document.getElementById('module2-part3-threshold');
    const blurInput = document.getElementById('module2-part3-blur');
    const runBtn = document.getElementById('module2-part3-run-btn');
    const resetBtn = document.getElementById('module2-part3-reset-btn');
    const statusLine = document.getElementById('module2-part3-status');
    const detectionCountEl = document.getElementById('module2-part3-detection-count');
    const templateCountEl = document.getElementById('module2-part3-template-count');
    const detectionLog = document.getElementById('module2-part3-detection-log');

    const sceneImg = document.getElementById('module2-part3-scene');
    const scenePlaceholder = document.getElementById('module2-part3-scene-placeholder');
    const detectionsImg = document.getElementById('module2-part3-detections');
    const detectionsPlaceholder = document.getElementById('module2-part3-detections-placeholder');
    const blurredImg = document.getElementById('module2-part3-blurred');
    const blurredPlaceholder = document.getElementById('module2-part3-blurred-placeholder');

    const templatesGrid = document.getElementById('module2-part3-templates-grid');
    const templatesPlaceholder = document.getElementById('module2-part3-templates-placeholder');

    if (!thresholdInput || !blurInput || !runBtn || !resetBtn || !statusLine || !detectionCountEl || !templateCountEl || !detectionLog || !sceneImg || !detectionsImg || !blurredImg || !templatesGrid) {
        return;
    }

    let referencesLoaded = false;
    let hasResults = false;

    const setStatus = (message, variant = 'info') => {
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const setImageState = (imgEl, placeholderEl, dataUrl) => {
        if (!imgEl) return;
        const wrapper = imgEl.closest('.result-image');
        if (dataUrl) {
            imgEl.src = dataUrl;
            imgEl.hidden = false;
            imgEl.dataset.loaded = 'true';
            if (wrapper) wrapper.dataset.empty = 'false';
            if (placeholderEl) placeholderEl.hidden = true;
        } else {
            imgEl.removeAttribute('src');
            imgEl.hidden = true;
            imgEl.dataset.loaded = 'false';
            if (wrapper) wrapper.dataset.empty = 'true';
            if (placeholderEl) placeholderEl.hidden = false;
        }
    };

    const formatLabel = (label) => {
        if (!label) return 'Unknown template';
        return label.replace(/^template_/i, '').replace(/_/g, ' ').trim() || label;
    };

    const renderDetections = (detections = []) => {
        detectionLog.dataset.empty = detections.length ? 'false' : 'true';
        detectionLog.innerHTML = '';
        if (!detections.length) {
            detectionLog.textContent = 'Run the detector to list every match, including correlation scores.';
            return;
        }

        detections.forEach((det, index) => {
            const entry = document.createElement('p');
            entry.className = 'module2-part3-log__item';
            const prettyLabel = formatLabel(det.label);
            const score = Number(det.score).toFixed(2);
            entry.innerHTML = `<strong>${index + 1}. ${prettyLabel}</strong> – score ${score}`;
            detectionLog.appendChild(entry);
        });
    };

    const renderTemplates = (templates = []) => {
        templateCountEl.textContent = templates.length;
        templatesGrid.innerHTML = '';
        if (!templates.length) {
            templatesGrid.dataset.empty = 'true';
            if (templatesPlaceholder) templatesPlaceholder.hidden = false;
            return;
        }
        templatesGrid.dataset.empty = 'false';
        if (templatesPlaceholder) templatesPlaceholder.hidden = true;

        templates.forEach((tpl) => {
            const card = document.createElement('figure');
            card.className = 'template-card';

            const img = document.createElement('img');
            img.src = tpl.image;
            img.alt = tpl.filename || 'Template image';
            img.loading = 'lazy';
            card.appendChild(img);

            const caption = document.createElement('figcaption');
            caption.textContent = formatLabel(tpl.label || tpl.filename);
            card.appendChild(caption);

            templatesGrid.appendChild(card);
        });
    };

    const clearResults = () => {
        hasResults = false;
        setImageState(detectionsImg, detectionsPlaceholder, null);
        setImageState(blurredImg, blurredPlaceholder, null);
        detectionCountEl.textContent = '0';
        renderDetections([]);
        resetBtn.disabled = true;
    };

    const loadReferences = async () => {
        if (referencesLoaded) return;
        try {
            const response = await fetch('/api/a2/part3/references');
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Unable to load references.');

            if (data.scene?.image) {
                setImageState(sceneImg, scenePlaceholder, data.scene.image);
            }
            renderTemplates(data.templates || []);
            setStatus('Reference assets loaded. Set your parameters and run detection.', 'info');
            referencesLoaded = true;
        } catch (err) {
            setStatus(err.message || 'Failed to load Module 2 Part 3 references.', 'error');
        }
    };

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const thresholdValue = parseFloat(thresholdInput.value);
        const blurValue = parseFloat(blurInput.value);

        if (Number.isNaN(thresholdValue) || thresholdValue < 0 || thresholdValue > 1) {
            setStatus('Please enter a threshold between 0.0 and 1.0.', 'error');
            return;
        }

        if (Number.isNaN(blurValue) || blurValue <= 0 || blurValue > 25) {
            setStatus('Blur multiplier must be between 0.1 and 25.', 'error');
            return;
        }

        runBtn.disabled = true;
        resetBtn.disabled = true;
        setStatus('Running multi-template detection…', 'info');

        try {
            const response = await fetch('/api/a2/part3', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: thresholdValue, blurMultiplier: blurValue })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Detection failed.');

            detectionCountEl.textContent = data.summary?.detected ?? data.detections?.length ?? 0;
            if (Number.isFinite(data.summary?.templatesTested)) {
                templateCountEl.textContent = data.summary.templatesTested;
            }
            if (data.detectionsImage) {
                setImageState(detectionsImg, detectionsPlaceholder, data.detectionsImage);
            } else {
                setImageState(detectionsImg, detectionsPlaceholder, null);
            }
            if (data.blurredImage) {
                setImageState(blurredImg, blurredPlaceholder, data.blurredImage);
            } else {
                setImageState(blurredImg, blurredPlaceholder, null);
            }
            renderDetections(data.detections || []);
            setStatus(data.message || 'Detection complete.', data.summary?.detected ? 'success' : 'warning');
            hasResults = true;
            resetBtn.disabled = false;
        } catch (err) {
            clearResults();
            setStatus(err.message || 'Unexpected error while running detection.', 'error');
        } finally {
            runBtn.disabled = false;
        }
    });

    resetBtn.addEventListener('click', () => {
        form.reset();
        clearResults();
        setStatus('Outputs cleared. Adjust the parameters and run again.', 'info');
    });

    registerTabChangeListener((prev, next) => {
        if (prev === 'a2' && next !== 'a2' && hasResults) {
            clearResults();
            setStatus('Outputs cleared after leaving Module 2.', 'info');
        }
        if (next === 'a2') {
            loadReferences();
        }
    });

    loadReferences();
}

const MODULE4_MIN_IMAGES = 8;

function initModule4Flow() {
    const configs = [
        {
            key: 'part1',
            formId: 'module4-part1-form',
            inputId: 'module4-part1-input',
            selectionId: 'module4-part1-selection',
            runBtnId: 'module4-part1-run',
            resetBtnId: 'module4-part1-reset',
            statusId: 'module4-part1-status',
            outputImgId: 'module4-part1-output',
            placeholderId: 'module4-part1-placeholder',
            countId: 'module4-part1-count',
            sizeId: 'module4-part1-size',
            downloadId: 'module4-part1-download',
            endpoint: '/api/a4/part1',
            clearEndpoint: '/api/a4/part1/results',
            readyMessage: 'Ready to stitch with the OpenCV pipeline.',
            uploadMessage: 'Uploading images and running the OpenCV stitcher…',
            downloadFilename: 'module4-part1-panorama.png'
        },
        {
            key: 'part2',
            formId: 'module4-part2-form',
            inputId: 'module4-part2-input',
            selectionId: 'module4-part2-selection',
            runBtnId: 'module4-part2-run',
            resetBtnId: 'module4-part2-reset',
            statusId: 'module4-part2-status',
            outputImgId: 'module4-part2-output',
            placeholderId: 'module4-part2-placeholder',
            countId: 'module4-part2-count',
            sizeId: 'module4-part2-size',
            downloadId: 'module4-part2-download',
            endpoint: '/api/a4/part2',
            clearEndpoint: '/api/a4/part2/results',
            readyMessage: 'Ready to run the scratch-built stitcher.',
            uploadMessage: 'Uploading images and running the custom SIFT pipeline…',
            downloadFilename: 'module4-part2-panorama.png'
        }
    ];

    configs.forEach((cfg) => setupModule4Part(cfg));
}

function setupModule4Part(config) {
    const form = document.getElementById(config.formId);
    if (!form) return;

    const fileInput = document.getElementById(config.inputId);
    const selectionEl = document.getElementById(config.selectionId);
    const runBtn = document.getElementById(config.runBtnId);
    const resetBtn = document.getElementById(config.resetBtnId);
    const statusLine = document.getElementById(config.statusId);
    const outputImg = document.getElementById(config.outputImgId);
    const placeholder = document.getElementById(config.placeholderId);
    const countEl = document.getElementById(config.countId);
    const sizeEl = document.getElementById(config.sizeId);
    const downloadBtn = document.getElementById(config.downloadId);

    if (!fileInput || !selectionEl || !runBtn || !resetBtn || !statusLine || !outputImg || !placeholder || !countEl || !sizeEl) {
        return;
    }

    const selectionDefault = selectionEl.innerHTML;
    const placeholderDefault = placeholder.textContent;
    const defaultStatus = statusLine.textContent || 'Upload 8+ portrait frames to enable stitching.';
    const readyMessage = config.readyMessage || 'Ready to stitch.';
    const uploadMessage = config.uploadMessage || 'Uploading images and stitching…';
    const state = { hasResults: false, downloadUrl: null };

    const setStatus = (message, variant = 'info') => {
        if (!statusLine) return;
        statusLine.textContent = message;
        statusLine.dataset.variant = variant;
    };

    const resetSelection = () => {
        selectionEl.dataset.empty = 'true';
        selectionEl.innerHTML = selectionDefault || '<p>No images selected yet.</p>';
    };

    const updateSelection = () => {
        const files = Array.from(fileInput.files || []);
        if (!files.length) {
            resetSelection();
            runBtn.disabled = true;
            resetBtn.disabled = !state.hasResults;
            return;
        }

        selectionEl.dataset.empty = 'false';
        selectionEl.innerHTML = '';
        const list = document.createElement('ol');
        list.className = 'module4-file-list';
        files.forEach((file, index) => {
            const item = document.createElement('li');
            item.textContent = `${index + 1}. ${file.name}`;
            list.appendChild(item);
        });
        selectionEl.appendChild(list);

        runBtn.disabled = files.length < MODULE4_MIN_IMAGES;
        if (!state.hasResults) {
            resetBtn.disabled = false;
        }
    };

    const clearOutput = () => {
        outputImg.removeAttribute('src');
        outputImg.hidden = true;
        placeholder.hidden = false;
        if (placeholderDefault) placeholder.textContent = placeholderDefault;
        countEl.textContent = '0';
        sizeEl.textContent = '-- × --';
        if (downloadBtn) downloadBtn.hidden = true;
        state.downloadUrl = null;
        state.hasResults = false;
    };

    const performReset = ({ clearServer = true, keepStatus = false } = {}) => {
        form.reset();
        try { fileInput.value = ''; } catch (err) { /* ignore readonly errors */ }
        clearOutput();
        updateSelection();
        runBtn.disabled = true;
        resetBtn.disabled = true;
        if (!keepStatus) setStatus(defaultStatus, 'info');
        if (clearServer && config.clearEndpoint) {
            fetch(config.clearEndpoint, { method: 'DELETE', keepalive: true }).catch(() => { /* best effort */ });
        }
    };

    const handleDownload = () => {
        if (!state.downloadUrl) return;
        const link = document.createElement('a');
        link.href = state.downloadUrl;
        link.download = config.downloadFilename || `${config.key || 'module4'}-panorama.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const updateButtonsAfterSubmit = () => {
        const fileCount = fileInput.files ? fileInput.files.length : 0;
        runBtn.disabled = fileCount < MODULE4_MIN_IMAGES;
        resetBtn.disabled = !(fileCount || state.hasResults);
    };

    const showResult = (payload, fileCount) => {
        if (payload.panorama) {
            outputImg.src = payload.panorama;
            outputImg.hidden = false;
            placeholder.hidden = true;
            state.downloadUrl = payload.panorama;
            if (downloadBtn) downloadBtn.hidden = false;
        }
        countEl.textContent = Number(payload.count || fileCount || 0).toString();
        if (Number.isFinite(payload.width) && Number.isFinite(payload.height)) {
            sizeEl.textContent = `${payload.width} × ${payload.height}`;
        } else {
            sizeEl.textContent = '-- × --';
        }
        state.hasResults = true;
        resetBtn.disabled = false;
        setStatus(payload.message || 'Panorama generated successfully.', 'success');
    };

    fileInput.addEventListener('change', () => {
        updateSelection();
        const count = fileInput.files ? fileInput.files.length : 0;
        if (count >= MODULE4_MIN_IMAGES) {
            setStatus(readyMessage, 'info');
        } else {
            setStatus(defaultStatus, 'info');
        }
    });

    resetBtn.addEventListener('click', () => {
        performReset({ clearServer: true });
    });

    if (downloadBtn) {
        downloadBtn.addEventListener('click', handleDownload);
    }

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const files = Array.from(fileInput.files || []);
        if (files.length < MODULE4_MIN_IMAGES) {
            setStatus(`Please select at least ${MODULE4_MIN_IMAGES} portrait images.`, 'error');
            return;
        }

        const formData = new FormData();
        files.forEach((file) => formData.append('images', file));

        runBtn.disabled = true;
        resetBtn.disabled = true;
        setStatus(uploadMessage, 'info');

        try {
            const response = await fetch(config.endpoint, { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Stitching failed.');
            showResult(data, files.length);
        } catch (err) {
            setStatus(err.message || 'Unexpected error while stitching.', 'error');
        } finally {
            updateButtonsAfterSubmit();
        }
    });

    registerTabChangeListener((prev, next) => {
        if (prev === 'a4' && next !== 'a4') {
            performReset({ clearServer: state.hasResults || (fileInput.files && fileInput.files.length > 0) });
        }
    });

    form.__module4Reset = (options) => performReset(options || { clearServer: true });

    updateSelection();
    setStatus(defaultStatus, 'info');
}

function initModule1Flow() {
    const refCanvas = document.getElementById('ref-canvas');
    const testCanvas = document.getElementById('test-canvas');
    if (!refCanvas || !testCanvas) return;

    const ctxRef = refCanvas.getContext('2d');
    const ctxTest = testCanvas.getContext('2d');

    const elements = {
        ref: {
            canvas: refCanvas,
            ctx: ctxRef,
            wrapper: document.querySelector('#module1-ref-card .canvas-wrapper'),
            empty: document.getElementById('ref-canvas-empty'),
            fileInput: document.getElementById('ref-image-input'),
            clearBtn: document.getElementById('ref-clear-btn'),
            resetBtn: document.getElementById('ref-reset-points'),
            pixelLabel: document.getElementById('ref-pixel-width'),
            realWidthInput: document.getElementById('ref-real-width'),
            distanceInput: document.getElementById('ref-distance'),
            calcBtn: document.getElementById('ref-calc-btn'),
            status: document.getElementById('ref-status'),
            card: document.getElementById('module1-ref-card')
        },
        test: {
            canvas: testCanvas,
            ctx: ctxTest,
            wrapper: document.querySelector('#module1-test-card .canvas-wrapper'),
            empty: document.getElementById('test-canvas-empty'),
            fileInput: document.getElementById('test-image-input'),
            clearBtn: document.getElementById('test-clear-btn'),
            resetBtn: document.getElementById('test-reset-points'),
            pixelLabel: document.getElementById('test-pixel-width'),
            distanceInput: document.getElementById('test-distance'),
            calcBtn: document.getElementById('test-calc-btn'),
            status: document.getElementById('test-status'),
            card: document.getElementById('module1-test-card')
        },
        summary: {
            focal: document.getElementById('summary-focal'),
            refPx: document.getElementById('summary-ref-px'),
            refDist: document.getElementById('summary-ref-dist'),
            testDist: document.getElementById('summary-test-dist'),
            testPx: document.getElementById('summary-test-px'),
            realWidth: document.getElementById('summary-real-width'),
            banner: document.getElementById('module1-result-banner'),
            reset: document.getElementById('module1-reset')
        }
    };

    // new control: Use the sample images provided in resources/m1
    elements.ref.useSampleBtn = document.getElementById('module1-use-sample-btn');

    const MAX_HEIGHT = 800;
    const MAX_WIDTH = 900;

    const state = {
        focalLength: null,
        reference: {
            image: null,
            meta: null,
            points: [],
            pixelWidth: null,
            realWidth: null,
            distance: null
        },
        test: {
            image: null,
            meta: null,
            points: [],
            pixelWidth: null,
            distance: null,
            realWidth: null
        }
    };

    const numberFormat = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

    elements.ref.fileInput.addEventListener('change', (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        loadImage('reference', file);
    });

    elements.test.fileInput.addEventListener('change', (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        loadImage('test', file);
    });

    elements.ref.clearBtn.addEventListener('click', () => resetImage('reference'));
    elements.test.clearBtn.addEventListener('click', () => resetImage('test'));

    elements.ref.resetBtn.addEventListener('click', () => resetPoints('reference'));
    elements.test.resetBtn.addEventListener('click', () => resetPoints('test'));

    elements.ref.realWidthInput.addEventListener('input', checkRefReady);
    elements.ref.distanceInput.addEventListener('input', checkRefReady);
    elements.test.distanceInput.addEventListener('input', checkTestReady);

    elements.ref.calcBtn.addEventListener('click', submitFocalLength);
    elements.test.calcBtn.addEventListener('click', submitRealWidth);
    elements.summary.reset.addEventListener('click', resetModule);

    elements.ref.canvas.addEventListener('click', (event) => handleCanvasClick('reference', event));
    elements.test.canvas.addEventListener('click', (event) => handleCanvasClick('test', event));

    function loadImage(kind, file) {
        const reader = new FileReader();
        reader.onload = (evt) => {
            const img = new Image();
            img.onload = () => {
                const target = kind === 'reference' ? elements.ref : elements.test;
                const { displayWidth, displayHeight } = calculateDisplaySize(img);

                const canvas = target.canvas;
                canvas.width = displayWidth;
                canvas.height = displayHeight;

                const ctx = target.ctx;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

                const bucket = state[kind === 'reference' ? 'reference' : 'test'];
                bucket.image = img;
                bucket.meta = {
                    displayWidth,
                    displayHeight,
                    naturalWidth: img.width,
                    naturalHeight: img.height
                };
                bucket.points = [];
                bucket.pixelWidth = null;

                target.wrapper.dataset.state = 'ready';
                target.empty.textContent = 'Click two points to draw the measurement line.';
                target.clearBtn.disabled = false;
                target.resetBtn.disabled = true;

                if (kind === 'reference') {
                    updateStatus('reference', 'Image loaded. Mark the known width.', 'info');
                } else {
                    updateStatus('test', 'Image loaded. Mark the test object.', 'info');
                    elements.test.distanceInput.disabled = false;
                }

                updatePixelMetric(kind);
                checkRefReady();
                checkTestReady();
            };
            img.onerror = () => {
                updateStatus(kind, 'Unable to load that file. Please try a different image.', 'error');
            };
            img.src = evt.target.result;
        };
        reader.readAsDataURL(file);
    }

    function loadImageFromDataUrl(kind, dataUrl, filename) {
        const img = new Image();
        img.onload = () => {
            const target = kind === 'reference' ? elements.ref : elements.test;
            const { displayWidth, displayHeight } = calculateDisplaySize(img);

            const canvas = target.canvas;
            canvas.width = displayWidth;
            canvas.height = displayHeight;

            const ctx = target.ctx;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

            const bucket = state[kind === 'reference' ? 'reference' : 'test'];
            bucket.image = img;
            bucket.meta = {
                displayWidth,
                displayHeight,
                naturalWidth: img.width,
                naturalHeight: img.height
            };
            bucket.points = [];
            bucket.pixelWidth = null;

            target.wrapper.dataset.state = 'ready';
            target.empty.textContent = 'Click two points to draw the measurement line.';
            target.clearBtn.disabled = false;
            target.resetBtn.disabled = true;

            if (kind === 'reference') {
                updateStatus('reference', 'Image loaded. Mark the known width.', 'info');
            } else {
                updateStatus('test', 'Image loaded. Mark the test object.', 'info');
                elements.test.distanceInput.disabled = false;
            }

            updatePixelMetric(kind);
            checkRefReady();
            checkTestReady();
        };
        img.onerror = () => updateStatus(kind, 'Unable to load that file.', 'error');
        img.src = dataUrl;
    }

    function calculateDisplaySize(img) {
        const widthRatio = MAX_WIDTH / img.width;
        const heightRatio = MAX_HEIGHT / img.height;
        const scale = Math.min(1, widthRatio, heightRatio);
        const displayWidth = Math.round(img.width * scale);
        const displayHeight = Math.round(img.height * scale);
        return { displayWidth, displayHeight };
    }

    function handleCanvasClick(kind, event) {
        const bucket = state[kind === 'reference' ? 'reference' : 'test'];
        if (!bucket.image || bucket.points.length === 2) return;

        const elementsGroup = kind === 'reference' ? elements.ref : elements.test;
        const rect = elementsGroup.canvas.getBoundingClientRect();
        const clickPoint = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };

        const meta = bucket.meta;
        if (!meta || !rect.width || !rect.height) return;

        const scaleX = meta.naturalWidth / rect.width;
        const scaleY = meta.naturalHeight / rect.height;
        const displayScaleX = meta.displayWidth / rect.width;
        const displayScaleY = meta.displayHeight / rect.height;

        const imagePoint = {
            x: clickPoint.x * scaleX,
            y: clickPoint.y * scaleY
        };

        const canvasPoint = {
            x: clickPoint.x * displayScaleX,
            y: clickPoint.y * displayScaleY
        };

        bucket.points.push({ canvas: canvasPoint, image: imagePoint });

        drawCanvas(kind);

        if (bucket.points.length === 2) {
            const [p1, p2] = bucket.points;
            bucket.pixelWidth = Math.hypot(p2.image.x - p1.image.x, p2.image.y - p1.image.y);
            updatePixelMetric(kind);
            elementsGroup.resetBtn.disabled = false;

            if (kind === 'reference') {
                updateStatus('reference', 'Select the real width and distance, then calculate the focal length.', 'success');
            } else {
                updateStatus('test', 'Enter the camera distance to compute the real-world size.', 'success');
            }
        } else {
            updateStatus(kind, 'Point recorded. Select one more point.', 'info');
        }

        checkRefReady();
        checkTestReady();
    }

    function drawCanvas(kind) {
        const target = kind === 'reference' ? elements.ref : elements.test;
        const bucket = state[kind === 'reference' ? 'reference' : 'test'];
        const ctx = target.ctx;
        const canvas = target.canvas;
        if (!bucket.image || !bucket.meta) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(bucket.image, 0, 0, bucket.meta.displayWidth, bucket.meta.displayHeight);

        bucket.points.forEach(({ canvas: point }, index) => {
            ctx.fillStyle = '#ff3b81';
            ctx.strokeStyle = '#1d1f27';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = '#101217';
            ctx.font = '12px Inter, sans-serif';
            ctx.fillText(index === 0 ? 'A' : 'B', point.x + 8, point.y - 8);
        });

        if (bucket.points.length === 2) {
            ctx.strokeStyle = '#31c48d';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(bucket.points[0].canvas.x, bucket.points[0].canvas.y);
            ctx.lineTo(bucket.points[1].canvas.x, bucket.points[1].canvas.y);
            ctx.stroke();
        }
    }

    function resetPoints(kind) {
        const bucket = state[kind === 'reference' ? 'reference' : 'test'];
        const target = kind === 'reference' ? elements.ref : elements.test;
        bucket.points = [];
        bucket.pixelWidth = null;
        drawCanvas(kind);
        target.pixelLabel.textContent = '--';
        target.resetBtn.disabled = true;

        if (kind === 'reference') {
            updateStatus('reference', 'Click two new points to measure the reference width.', 'info');
        } else {
            updateStatus('test', 'Click two new points to measure the test object.', 'info');
        }

        checkRefReady();
        checkTestReady();
    }

    function resetImage(kind) {
        const bucket = state[kind === 'reference' ? 'reference' : 'test'];
        const target = kind === 'reference' ? elements.ref : elements.test;
        bucket.image = null;
        bucket.meta = null;
        resetPoints(kind);
        target.canvas.width = target.canvas.height = 0;
        target.wrapper.dataset.state = 'empty';
        target.empty.textContent = 'Upload an image to begin';
        target.fileInput.value = '';
        target.clearBtn.disabled = true;
        if (kind === 'test') {
            elements.test.distanceInput.value = '';
            elements.test.distanceInput.disabled = true;
        }

        checkRefReady();
        checkTestReady();
    }

    function updatePixelMetric(kind) {
        const target = kind === 'reference' ? elements.ref : elements.test;
        const bucket = state[kind === 'reference' ? 'reference' : 'test'];
        target.pixelLabel.textContent = bucket.pixelWidth ? numberFormat.format(bucket.pixelWidth) + ' px' : '--';
    }

    function checkRefReady() {
        const hasImage = Boolean(state.reference.image);
        const hasPoints = typeof state.reference.pixelWidth === 'number';
        const realWidth = parseFloat(elements.ref.realWidthInput.value);
        const distance = parseFloat(elements.ref.distanceInput.value);
        const numericReady = realWidth > 0 && distance > 0;
        const enable = hasImage && hasPoints && numericReady;
        elements.ref.calcBtn.disabled = !enable;
        return enable;
    }

    function checkTestReady() {
        const unlocked = Boolean(state.focalLength);
        const hasImage = Boolean(state.test.image);
        const hasPoints = typeof state.test.pixelWidth === 'number';
        const distance = parseFloat(elements.test.distanceInput.value);
        const enable = unlocked && hasImage && hasPoints && distance > 0;
        elements.test.calcBtn.disabled = !enable;
        elements.test.resetBtn.disabled = !hasImage || state.test.points.length === 0;
        return enable;
    }

    async function submitFocalLength() {
        if (!checkRefReady()) return;
        toggleLoading(elements.ref.calcBtn, true, 'Calculating...');
        updateStatus('reference', 'Calculating focal length...', 'info');

        const payload = {
            pixelWidth: state.reference.pixelWidth,
            realWidth: parseFloat(elements.ref.realWidthInput.value),
            distance: parseFloat(elements.ref.distanceInput.value)
        };

        try {
            const res = await fetch('/api/a1/focal-length', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Unable to compute focal length.');

            state.focalLength = data.focalLength;
            state.reference.pixelWidth = data.refPixelWidth;
            state.reference.realWidth = data.refRealWidth;
            state.reference.distance = data.refDistance;

            updateStatus('reference', `Focal length stored: ${numberFormat.format(state.focalLength)} px`, 'success');
            unlockTestStep();
            updateSummary();
        } catch (err) {
            updateStatus('reference', err.message, 'error');
        } finally {
            toggleLoading(elements.ref.calcBtn, false);
            checkRefReady();
        }
    }

    async function submitRealWidth() {
        if (!checkTestReady()) return;
        toggleLoading(elements.test.calcBtn, true, 'Calculating...');
        updateStatus('test', 'Calculating real-world width...', 'info');

        const payload = {
            pixelWidth: state.test.pixelWidth,
            distance: parseFloat(elements.test.distanceInput.value),
            focalLength: state.focalLength
        };

        try {
            const res = await fetch('/api/a1/real-width', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Unable to compute real width.');

            state.test.pixelWidth = data.testPixelWidth;
            state.test.distance = data.testDistance;
            state.test.realWidth = data.realWidth;

            updateStatus('test', 'Measurement complete!', 'success');
            updateSummary();
            elements.summary.banner.textContent = `Calculated width: ${numberFormat.format(state.test.realWidth)} cm`;
            elements.summary.banner.classList.add('ready');
        } catch (err) {
            updateStatus('test', err.message, 'error');
        } finally {
            toggleLoading(elements.test.calcBtn, false);
            checkTestReady();
        }
    }

    function unlockTestStep() {
        elements.test.card.classList.remove('locked');
        elements.test.card.removeAttribute('aria-disabled');
        elements.test.fileInput.disabled = false;
        elements.test.clearBtn.disabled = false;
        elements.test.distanceInput.disabled = false;
        updateStatus('test', 'Upload a test image captured by the same camera.', 'info');
    }

    function resetModule() {
        resetImage('reference');
        resetImage('test');
        elements.ref.realWidthInput.value = '';
        elements.ref.distanceInput.value = '';
        elements.test.distanceInput.value = '';
        elements.test.fileInput.disabled = true;
        elements.test.clearBtn.disabled = true;
        elements.test.card.classList.add('locked');
        elements.test.card.setAttribute('aria-disabled', 'true');
        state.reference.realWidth = null;
        state.reference.distance = null;
        state.test.realWidth = null;
        state.test.distance = null;
        state.focalLength = null;
        elements.summary.banner.textContent = 'Run the steps to see the measurement summary.';
        elements.summary.banner.classList.remove('ready');
        updateStatus('reference', 'Upload a reference image to begin calibration.', 'info');
        updateStatus('test', 'Complete Step 1 to unlock measurement.', 'info');
        updateSummary();
        // remove autofill indication
        elements.ref.realWidthInput.classList.remove('autofilled');
        elements.ref.distanceInput.classList.remove('autofilled');
        elements.test.distanceInput.classList.remove('autofilled');
        const expectedEl = document.getElementById('test-expected-value');
        if (expectedEl) expectedEl.textContent = '--';
    }

    function updateSummary() {
        elements.summary.focal.textContent = state.focalLength ? numberFormat.format(state.focalLength) : '--';
        elements.summary.refPx.textContent = state.reference.pixelWidth ? numberFormat.format(state.reference.pixelWidth) : '--';
        elements.summary.refDist.textContent = state.reference.distance ? numberFormat.format(state.reference.distance) : '--';
        elements.summary.testDist.textContent = state.test.distance ? numberFormat.format(state.test.distance) : '--';
        elements.summary.testPx.textContent = state.test.pixelWidth ? numberFormat.format(state.test.pixelWidth) : '--';
        elements.summary.realWidth.textContent = state.test.realWidth ? numberFormat.format(state.test.realWidth) : '--';
    }

    function toggleLoading(button, isLoading, label = 'Processing...') {
        if (isLoading) {
            button.dataset.originalLabel = button.textContent;
            button.textContent = label;
            button.disabled = true;
        } else {
            const original = button.dataset.originalLabel || label;
            button.textContent = original;
            button.disabled = false;
        }
    }

    function updateStatus(kind, message, variant) {
        const target = kind === 'reference' ? elements.ref.status : elements.test.status;
        target.textContent = message;
        target.dataset.variant = variant;
    }

    resetModule();

    // When the 'Use example images' button is clicked, fetch the sample images from server
    if (elements.ref.useSampleBtn) {
        elements.ref.useSampleBtn.addEventListener('click', async () => {
            try {
                elements.ref.useSampleBtn.disabled = true;
                updateStatus('reference', 'Loading example images…', 'info');
                const resp = await fetch('/api/a1/samples');
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.error || 'Failed to load samples');

                if (data.reference?.image) {
                    loadImageFromDataUrl('reference', data.reference.image, data.reference.filename);
                }
                if (data.test?.image) {
                    loadImageFromDataUrl('test', data.test.image, data.test.filename);
                }

                // Autofill fields from the measurements file
                const addAutofilled = (elem, value) => {
                    if (!elem || typeof value === 'undefined' || value === null) return;
                    elem.value = value;
                    elem.classList.add('autofilled');
                    // remove the autofilled class on user input
                    const removeAutofill = () => { elem.classList.remove('autofilled'); elem.removeEventListener('input', removeAutofill); };
                    elem.addEventListener('input', removeAutofill);
                };

                addAutofilled(elements.ref.realWidthInput, Number.isFinite(data.reference.realWidth) ? data.reference.realWidth : '');
                addAutofilled(elements.ref.distanceInput, Number.isFinite(data.reference.distance) ? data.reference.distance : '');
                addAutofilled(elements.test.distanceInput, Number.isFinite(data.test.distance) ? data.test.distance : '');

                // display expected test width if present
                const expectedEl = document.getElementById('test-expected-value');
                if (expectedEl) {
                    expectedEl.textContent = Number.isFinite(data.test.expectedWidth) ? Number(data.test.expectedWidth).toFixed(2) : '--';
                }

                // allow the test step to be unlocked after focal-length calculation; show test controls
                updateStatus('reference', 'Example assets loaded. Mark points and calculate focal length.', 'success');
                elements.ref.useSampleBtn.disabled = false;
            } catch (err) {
                elements.ref.useSampleBtn.disabled = false;
                updateStatus('reference', err.message || 'Failed to load example images', 'error');
                console.error(err);
            }
        });
    }
}

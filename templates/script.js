function switchTab(tabId) {
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

        if (isVideo) {
            toggleBtn.innerText = expanded ? 'Watch the video ▾' : 'Watch the video ▴';
        } else {
            toggleBtn.innerText = expanded ? 'View source code ▾' : 'View source code ▴';
        }
    }

    function handleFileSelection(btn) {
        // For now the project requires the actual file contents to be blank —
        // we intentionally set an empty code area. This placeholder keeps
        // the interactive UI consistent for future population.
        const moduleId = btn.dataset.module; // e.g. "a1"
        const codeEl = document.querySelector(`#source-code-${moduleId} code`);
        if (!codeEl) return;

        // Clear content (leave blank as requested) and focus the code area
        codeEl.textContent = '';
        codeEl.parentElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    document.addEventListener('click', (e) => {
        const t = e.target;
        if (t.classList && t.classList.contains('source-toggle')) {
            togglePanel(t);
            return;
        }

        if (t.classList && t.classList.contains('video-toggle')) {
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

        if (target.classList && (target.classList.contains('source-toggle') || target.classList.contains('video-toggle'))) {
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

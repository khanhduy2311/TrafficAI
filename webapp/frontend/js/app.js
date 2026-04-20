/**
 * TrafficAI — Frontend Application
 * WebSocket video streaming, violation tracking, UI management
 */

(() => {
    'use strict';

    // ═══════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════

    const state = {
        ws: null,
        isStreaming: false,
        source: 'webcam',          // 'webcam' | 'upload'
        uploadedFilename: null,
        violations: [],
        activeFilter: 'all',
        stats: {
            vehicles: 0,
            redLight: 0,
            helmet: 0,
            lane: 0,
            speed: 0,
            fps: 0,
        },
    };

    // ═══════════════════════════════════════════════
    // DOM REFS
    // ═══════════════════════════════════════════════

    const $ = (id) => document.getElementById(id);
    const $$ = (sel) => document.querySelectorAll(sel);

    const els = {
        // Connection
        connectionStatus: $('connectionStatus'),

        // Controls
        btnWebcam: $('btnWebcam'),
        btnUpload: $('btnUpload'),
        btnUrl:    $('btnUrl'),
        btnStart:  $('btnStart'),
        btnStop:   $('btnStop'),
        urlInput:  $('urlInput'),

        // Video
        videoCanvas:    $('videoCanvas'),
        videoOverlay:   $('videoOverlay'),
        videoContainer: $('videoContainer'),
        dropZone:       $('dropZone'),
        fileInput:      $('fileInput'),

        // Upload info
        uploadInfo:   $('uploadInfo'),
        uploadName:   $('uploadName'),
        uploadSize:   $('uploadSize'),
        uploadRemove: $('uploadRemove'),

        // HUD
        hudFps:      $('hudFps'),
        hudLight:    $('hudLight'),
        lightDot:    $('lightDot'),
        lightText:   $('lightText'),
        hudProgress: $('hudProgress'),

        // Violations
        violationsList: $('violationsList'),
        violationCount: $('violationCount'),
        emptyState:     $('emptyState'),

        // Stats
        statVehicles: $('statVehicles'),
        statRedLight: $('statRedLight'),
        statHelmet:   $('statHelmet'),
        statLane:     $('statLane'),
        statSpeed:    $('statSpeed'),
        statFps:      $('statFps'),

        // Header buttons
        btnExportCSV:   $('btnExportCSV'),
        btnClearHistory:$('btnClearHistory'),

        // Modals
        evidenceModal: $('evidenceModal'),
        modalClose:    $('modalClose'),
        modalBody:     $('modalBody'),
        confirmDialog: $('confirmDialog'),
        confirmCancel: $('confirmCancel'),
        confirmOk:     $('confirmOk'),

        // Toast
        toastContainer: $('toastContainer'),
    };

    const ctx = els.videoCanvas.getContext('2d');

    // ═══════════════════════════════════════════════
    // VIOLATION LABELS
    // ═══════════════════════════════════════════════

    const VIOLATION_LABELS = {
        red_light:  'Vượt đèn đỏ',
        no_helmet:  'Không đội mũ BH',
        wrong_lane: 'Chạy sai làn',
        speed_limit:'Vượt tốc độ',
    };

    const VIOLATION_ICONS = {
        red_light:  '🚦',
        no_helmet:  '⛑️',
        wrong_lane: '🚧',
        speed_limit:'⚡',
    };

    // ═══════════════════════════════════════════════
    // WEBSOCKET
    // ═══════════════════════════════════════════════

    function connectWebSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/ws/video`;

        setConnectionStatus('connecting');
        state.ws = new WebSocket(wsUrl);
        state.ws.binaryType = 'arraybuffer';

        state.ws.onopen = () => {
            console.log('[WS] Connected');
            setConnectionStatus('online');
            showToast('Đã kết nối server', 'success');
        };

        state.ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                // Binary = JPEG frame
                renderFrame(event.data);
            } else {
                // Text = JSON metadata
                const data = JSON.parse(event.data);
                handleMetadata(data);
            }
        };

        state.ws.onclose = () => {
            console.log('[WS] Disconnected');
            setConnectionStatus('offline');
            if (state.isStreaming) {
                state.isStreaming = false;
                updateControlState();
                showToast('Đã mất kết nối', 'warning');
            }
        };

        state.ws.onerror = (err) => {
            console.error('[WS] Error:', err);
            showToast('Lỗi kết nối WebSocket', 'error');
        };
    }

    function setConnectionStatus(status) {
        const dot = els.connectionStatus.querySelector('.status-dot');
        const text = els.connectionStatus.querySelector('.status-text');
        dot.className = 'status-dot ' + status;
        const labels = {
            offline:    'Chưa kết nối',
            connecting: 'Đang kết nối...',
            online:     'Đã kết nối',
        };
        text.textContent = labels[status] || status;
    }

    // ═══════════════════════════════════════════════
    // RENDER FRAME
    // ═══════════════════════════════════════════════

    const frameImage = new Image();
    let frameUrl = null;

    function renderFrame(arrayBuffer) {
        // Release previous blob URL
        if (frameUrl) URL.revokeObjectURL(frameUrl);

        const blob = new Blob([arrayBuffer], { type: 'image/jpeg' });
        frameUrl = URL.createObjectURL(blob);

        frameImage.onload = () => {
            // Auto-resize canvas to match image aspect ratio
            const cw = els.videoCanvas.parentElement.clientWidth;
            const ch = els.videoCanvas.parentElement.clientHeight;
            const imgRatio = frameImage.width / frameImage.height;
            const canvasRatio = cw / ch;

            let drawW, drawH, offsetX = 0, offsetY = 0;

            if (imgRatio > canvasRatio) {
                drawW = cw;
                drawH = cw / imgRatio;
                offsetY = (ch - drawH) / 2;
            } else {
                drawH = ch;
                drawW = ch * imgRatio;
                offsetX = (cw - drawW) / 2;
            }

            els.videoCanvas.width = cw;
            els.videoCanvas.height = ch;

            ctx.fillStyle = '#06060b';
            ctx.fillRect(0, 0, cw, ch);
            ctx.drawImage(frameImage, offsetX, offsetY, drawW, drawH);
        };

        frameImage.src = frameUrl;
    }

    // ═══════════════════════════════════════════════
    // HANDLE METADATA
    // ═══════════════════════════════════════════════

    function handleMetadata(data) {
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        if (data.status === 'started') {
            state.isStreaming = true;
            els.videoOverlay.classList.add('hidden');
            updateControlState();
            showToast('Bắt đầu stream video', 'success');
            return;
        }

        if (data.status === 'finished') {
            state.isStreaming = false;
            updateControlState();
            showToast('Video đã xử lý xong', 'success');
            return;
        }

        if (data.status === 'stopped') {
            state.isStreaming = false;
            updateControlState();
            return;
        }

        if (data.type === 'frame_data') {
            // Update HUD
            updateStat(els.hudFps, `${data.fps} FPS`);
            updateLightStatus(data.light_status);

            if (data.progress > 0) {
                els.hudProgress.textContent = `${data.progress}%`;
            }

            // Update stats
            animateStat(els.statVehicles, data.vehicle_count);
            animateStat(els.statRedLight, data.total_violations_redlight || 0);
            animateStat(els.statHelmet, data.total_violations_helmet || 0);
            animateStat(els.statFps, data.fps);
            state.stats.vehicles = data.vehicle_count;

            // New violations
            if (data.violations && data.violations.length > 0) {
                for (const v of data.violations) {
                    addViolation(v);
                }
            }
        }
    }

    function updateLightStatus(status) {
        els.lightDot.className = 'light-dot ' + (status || '');
        const labels = {
            red: 'Đèn đỏ',
            yellow: 'Đèn vàng',
            green: 'Đèn xanh',
            off: 'Đèn tắt',
            unknown: '---',
        };
        els.lightText.textContent = labels[status] || '---';
    }

    function updateStat(el, value) {
        if (el.textContent !== String(value)) {
            el.textContent = value;
        }
    }

    function animateStat(el, value) {
        const strVal = String(value);
        if (el.textContent !== strVal) {
            el.textContent = strVal;
            el.classList.add('updated');
            setTimeout(() => el.classList.remove('updated'), 300);
        }
    }

    // ═══════════════════════════════════════════════
    // VIOLATIONS
    // ═══════════════════════════════════════════════

    function addViolation(v) {
        state.violations.unshift(v);

        // Update count badge
        const count = state.violations.length;
        els.violationCount.textContent = count;
        els.violationCount.classList.add('pulse');
        setTimeout(() => els.violationCount.classList.remove('pulse'), 400);

        // Hide empty state
        if (els.emptyState) els.emptyState.style.display = 'none';

        // Create card
        const card = createViolationCard(v);

        // Apply filter
        if (state.activeFilter !== 'all' && v.violation_type !== state.activeFilter) {
            card.style.display = 'none';
        }

        // Insert at top
        els.violationsList.insertBefore(card, els.violationsList.firstChild);

        // Show toast for violation
        const label = VIOLATION_LABELS[v.violation_type] || v.violation_type;
        const icon = VIOLATION_ICONS[v.violation_type] || '🚨';
        showToast(`${icon} ${label} — ID:${v.track_id}`, 'error');
    }

    function createViolationCard(v) {
        const card = document.createElement('div');
        card.className = 'violation-card';
        card.dataset.type = v.violation_type;

        const label = VIOLATION_LABELS[v.violation_type] || v.violation_type;
        const timeStr = v.timestamp ?
            v.timestamp.replace(/_/g, ':').substring(0, 15) :
            new Date().toLocaleTimeString('vi-VN');

        // Thumbnail
        let thumbHtml = '';
        if (v.evidence_path) {
            thumbHtml = `<img class="vio-thumb" src="/evidence/${v.evidence_path}" 
                          alt="evidence" onerror="this.style.display='none'">`;
        } else {
            thumbHtml = `<div class="vio-thumb" style="display:flex;align-items:center;justify-content:center;font-size:1.4rem">
                ${VIOLATION_ICONS[v.violation_type] || '🚨'}
            </div>`;
        }

        card.innerHTML = `
            ${thumbHtml}
            <div class="vio-info">
                <div class="vio-type ${v.violation_type}">${label}</div>
                <div class="vio-detail">
                    ID: <strong>${v.track_id}</strong> · ${v.vehicle_type}
                    <br><span>Conf: ${(v.confidence * 100).toFixed(1)}% · Frame ${v.frame_number}</span>
                </div>
                <div class="vio-time">${timeStr}</div>
            </div>
        `;

        card.addEventListener('click', () => showEvidenceModal(v));
        return card;
    }

    function showEvidenceModal(v) {
        const label = VIOLATION_LABELS[v.violation_type] || v.violation_type;
        const icon = VIOLATION_ICONS[v.violation_type] || '🚨';

        let imgHtml = '';
        if (v.evidence_path) {
            imgHtml = `<img class="evidence-img" src="/evidence/${v.evidence_path}" alt="evidence">`;
        }

        els.modalBody.innerHTML = `
            ${imgHtml}
            <div class="evidence-details">
                <div class="evidence-item">
                    <div class="label">Loại vi phạm</div>
                    <div class="value">${icon} ${label}</div>
                </div>
                <div class="evidence-item">
                    <div class="label">Track ID</div>
                    <div class="value">#${v.track_id}</div>
                </div>
                <div class="evidence-item">
                    <div class="label">Loại xe</div>
                    <div class="value">${v.vehicle_type}</div>
                </div>
                <div class="evidence-item">
                    <div class="label">Confidence</div>
                    <div class="value">${(v.confidence * 100).toFixed(1)}%</div>
                </div>
                <div class="evidence-item">
                    <div class="label">Frame</div>
                    <div class="value">${v.frame_number}</div>
                </div>
                <div class="evidence-item">
                    <div class="label">Thời gian</div>
                    <div class="value">${v.timestamp || '---'}</div>
                </div>
            </div>
        `;

        els.evidenceModal.classList.add('active');
    }

    // ═══════════════════════════════════════════════
    // CONTROLS
    // ═══════════════════════════════════════════════

    function startStream() {
        if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
            showToast('Chưa kết nối server. Đang thử kết nối lại...', 'warning');
            connectWebSocket();
            setTimeout(startStream, 1500);
            return;
        }

        const msg = { action: 'start', source: state.source };

        if (state.source === 'video' && state.uploadedFilename) {
            msg.source = 'video';
            msg.filename = state.uploadedFilename;
        } else if (state.source === 'upload' && !state.uploadedFilename) {
            showToast('Chưa upload video. Hãy chọn file trước.', 'warning');
            return;
        } else if (state.source === 'url') {
            const val = els.urlInput.value.trim();
            if (!val) {
                showToast('Vui lòng nhập đường dẫn URL.', 'warning');
                return;
            }
            msg.url = val;
        }

        state.ws.send(JSON.stringify(msg));
        els.btnStart.disabled = true;
    }

    function stopStream() {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ action: 'stop' }));
        }
        state.isStreaming = false;
        updateControlState();
    }

    function updateControlState() {
        els.btnStart.disabled = state.isStreaming;
        els.btnStop.disabled = !state.isStreaming;

        if (!state.isStreaming) {
            els.videoOverlay.classList.remove('hidden');
        }
    }

    // ═══════════════════════════════════════════════
    // FILE UPLOAD
    // ═══════════════════════════════════════════════

    async function uploadFile(file) {
        if (!file) return;

        showToast(`Đang upload: ${file.name}...`, 'success');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await resp.json();

            if (resp.ok) {
                state.uploadedFilename = data.filename;
                state.source = 'video';

                // Update UI
                els.uploadInfo.style.display = 'flex';
                els.uploadName.textContent = data.original_name;
                els.uploadSize.textContent = `${data.size_mb} MB`;

                // Switch to upload mode
                els.btnWebcam.classList.remove('active');
                els.btnUpload.classList.add('active');

                showToast(`Upload thành công: ${data.original_name}`, 'success');
            } else {
                showToast(data.error || 'Upload thất bại', 'error');
            }
        } catch (err) {
            showToast('Lỗi upload: ' + err.message, 'error');
        }
    }

    function removeUpload() {
        state.uploadedFilename = null;
        els.uploadInfo.style.display = 'none';
        els.fileInput.value = '';
    }

    // ═══════════════════════════════════════════════
    // EXPORT & CLEAR
    // ═══════════════════════════════════════════════

    async function exportCSV() {
        try {
            const resp = await fetch('/api/violations/export');
            const csv = await resp.text();

            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `violations_${new Date().toISOString().slice(0,10)}.csv`;
            a.click();
            URL.revokeObjectURL(url);

            showToast('Đã xuất báo cáo CSV', 'success');
        } catch (err) {
            showToast('Lỗi xuất CSV: ' + err.message, 'error');
        }
    }

    async function clearHistory() {
        try {
            await fetch('/api/violations', { method: 'DELETE' });
            state.violations = [];
            els.violationsList.innerHTML = '';
            els.violationCount.textContent = '0';

            // Reset stats
            animateStat(els.statRedLight, 0);
            animateStat(els.statHelmet, 0);
            animateStat(els.statLane, 0);
            animateStat(els.statSpeed, 0);

            // Show empty state
            els.violationsList.innerHTML = `
                <div class="empty-state" id="emptyState">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.15)" stroke-width="1.5">
                        <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    <p>Chưa phát hiện vi phạm</p>
                    <p class="hint">Vi phạm sẽ hiển thị tại đây khi hệ thống đang chạy</p>
                </div>
            `;

            els.confirmDialog.classList.remove('active');
            showToast('Đã xoá toàn bộ lịch sử', 'success');
        } catch (err) {
            showToast('Lỗi xoá lịch sử: ' + err.message, 'error');
        }
    }

    // ═══════════════════════════════════════════════
    // FILTER
    // ═══════════════════════════════════════════════

    function applyFilter(filter) {
        state.activeFilter = filter;

        // Update button states
        $$('.filter-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });

        // Filter cards
        const cards = els.violationsList.querySelectorAll('.violation-card');
        cards.forEach(card => {
            if (filter === 'all' || card.dataset.type === filter) {
                card.style.display = '';
            } else {
                card.style.display = 'none';
            }
        });
    }

    // ═══════════════════════════════════════════════
    // TOASTS
    // ═══════════════════════════════════════════════

    function showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        els.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('removing');
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    // ═══════════════════════════════════════════════
    // DRAG & DROP
    // ═══════════════════════════════════════════════

    function setupDragDrop() {
        const container = els.videoContainer;

        container.addEventListener('dragover', (e) => {
            e.preventDefault();
            els.dropZone.classList.add('active');
        });

        container.addEventListener('dragleave', (e) => {
            if (!container.contains(e.relatedTarget)) {
                els.dropZone.classList.remove('active');
            }
        });

        container.addEventListener('drop', (e) => {
            e.preventDefault();
            els.dropZone.classList.remove('active');

            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('video/')) {
                uploadFile(files[0]);
            } else {
                showToast('Chỉ hỗ trợ file video', 'warning');
            }
        });
    }

    // ═══════════════════════════════════════════════
    // EVENT LISTENERS
    // ═══════════════════════════════════════════════

    function bindEvents() {
        // Source selector
        els.btnWebcam.addEventListener('click', () => {
            state.source = 'webcam';
            els.btnWebcam.classList.add('active');
            els.btnUpload.classList.remove('active');
            els.btnUrl.classList.remove('active');
            els.urlInput.style.display = 'none';
        });

        els.btnUrl.addEventListener('click', () => {
            state.source = 'url';
            els.btnUrl.classList.add('active');
            els.btnWebcam.classList.remove('active');
            els.btnUpload.classList.remove('active');
            els.urlInput.style.display = 'block';
            els.urlInput.focus();
        });

        els.btnUpload.addEventListener('click', () => {
            state.source = 'upload';
            els.btnUpload.classList.add('active');
            els.btnWebcam.classList.remove('active');
            els.btnUrl.classList.remove('active');
            els.urlInput.style.display = 'none';
            if (!state.uploadedFilename) {
                els.fileInput.click();
            }
        });

        // Start / Stop
        els.btnStart.addEventListener('click', startStream);
        els.btnStop.addEventListener('click', stopStream);

        // File input
        els.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                uploadFile(e.target.files[0]);
            }
        });

        // Remove upload
        els.uploadRemove.addEventListener('click', removeUpload);

        // Filters
        $$('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                applyFilter(btn.dataset.filter);
            });
        });

        // Export CSV
        els.btnExportCSV.addEventListener('click', exportCSV);

        // Clear history
        els.btnClearHistory.addEventListener('click', () => {
            els.confirmDialog.classList.add('active');
        });
        els.confirmCancel.addEventListener('click', () => {
            els.confirmDialog.classList.remove('active');
        });
        els.confirmOk.addEventListener('click', clearHistory);

        // Evidence modal close
        els.modalClose.addEventListener('click', () => {
            els.evidenceModal.classList.remove('active');
        });
        els.evidenceModal.addEventListener('click', (e) => {
            if (e.target === els.evidenceModal) {
                els.evidenceModal.classList.remove('active');
            }
        });

        // Confirm dialog click outside
        els.confirmDialog.addEventListener('click', (e) => {
            if (e.target === els.confirmDialog) {
                els.confirmDialog.classList.remove('active');
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                els.evidenceModal.classList.remove('active');
                els.confirmDialog.classList.remove('active');
            }
        });

        // Drag & Drop
        setupDragDrop();
    }

    // ═══════════════════════════════════════════════
    // LOAD HISTORY
    // ═══════════════════════════════════════════════

    async function loadHistory() {
        try {
            const resp = await fetch('/api/violations?limit=50');
            const data = await resp.json();

            if (data.violations && data.violations.length > 0) {
                if (els.emptyState) els.emptyState.style.display = 'none';

                for (const row of data.violations.reverse()) {
                    const v = {
                        track_id: row.track_id,
                        vehicle_type: row.vehicle_type,
                        violation_type: row.violation_type,
                        confidence: row.confidence,
                        frame_number: row.frame_number,
                        bbox: JSON.parse(row.bbox || '[]'),
                        evidence_path: row.evidence_path,
                        timestamp: row.timestamp,
                    };
                    state.violations.unshift(v);
                    const card = createViolationCard(v);
                    els.violationsList.insertBefore(card, els.violationsList.firstChild);
                }

                els.violationCount.textContent = state.violations.length;
            }

            // Load stats
            const statsResp = await fetch('/api/stats');
            const stats = await statsResp.json();
            if (stats.by_type) {
                animateStat(els.statRedLight, stats.by_type.red_light || 0);
                animateStat(els.statHelmet, stats.by_type.no_helmet || 0);
                animateStat(els.statLane, stats.by_type.wrong_lane || 0);
                animateStat(els.statSpeed, stats.by_type.speed_limit || 0);
            }
        } catch (err) {
            console.log('Could not load history:', err);
        }
    }

    // ═══════════════════════════════════════════════
    // INIT
    // ═══════════════════════════════════════════════

    function init() {
        bindEvents();
        connectWebSocket();
        loadHistory();
        console.log('[TrafficAI] Initialized');
    }

    // Start when DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();

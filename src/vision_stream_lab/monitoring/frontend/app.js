const wall = document.querySelector('#camera-wall');
const cameraTemplate = document.querySelector('#camera-template');
const emptyState = document.querySelector('#empty-state');
const useCaseSelect = document.querySelector('#use-case');
const columnsSelect = document.querySelector('#columns');
const cards = new Map();

let selectedUseCase = '';
let streamFps = 12;
let updateTimer;

function streamUrl(cameraId) {
  const query = new URLSearchParams({ use_case: selectedUseCase, fps: String(streamFps) });
  return `/api/cameras/${encodeURIComponent(cameraId)}/stream.mjpg?${query}`;
}

function connectStream(card, cameraId) {
  const image = card.querySelector('.camera-stream');
  card.classList.remove('streaming');
  image.src = streamUrl(cameraId);
}

function createCard(camera) {
  const card = cameraTemplate.content.firstElementChild.cloneNode(true);
  card.dataset.cameraId = camera.id;
  card.querySelector('.camera-name').textContent = camera.name;
  card.querySelector('.camera-id').textContent = camera.id;
  const image = card.querySelector('.camera-stream');
  image.alt = `Live AI output for ${camera.name}`;
  image.addEventListener('load', () => card.classList.add('streaming'));
  image.addEventListener('error', () => {
    card.classList.remove('streaming');
    window.setTimeout(() => {
      if (!document.hidden && cards.has(camera.id)) connectStream(card, camera.id);
    }, 1500);
  });
  card.querySelector('.card-fullscreen').addEventListener('click', () => {
    if (document.fullscreenElement === card) document.exitFullscreen();
    else card.requestFullscreen();
  });
  wall.appendChild(card);
  cards.set(camera.id, card);
  connectStream(card, camera.id);
  return card;
}

function removeMissingCards(cameraIds) {
  for (const [cameraId, card] of cards) {
    if (!cameraIds.has(cameraId)) {
      card.querySelector('.camera-stream').src = '';
      card.remove();
      cards.delete(cameraId);
    }
  }
}

function updateCard(card, camera) {
  const metrics = camera.use_cases[selectedUseCase];
  card.classList.toggle('online', camera.online);
  card.querySelector('.capture-fps').textContent = camera.capture_fps.toFixed(1);
  card.querySelector('.inference-fps').textContent = metrics ? metrics.inference_fps.toFixed(1) : '—';
  card.querySelector('.output-fps').textContent = metrics ? metrics.output_fps.toFixed(1) : '—';
  card.querySelector('.latency').textContent = metrics ? metrics.latency_ms.toFixed(0) : '—';
  card.querySelector('.frames').textContent = metrics ? `${metrics.inferred}/${camera.captured}` : `0/${camera.captured}`;
  card.querySelector('.event-badge').textContent = metrics
    ? `${metrics.events} events · ${metrics.dropped_signals} dropped`
    : 'Pipeline not assigned';
}

function setUseCases(data) {
  const ids = data.use_cases.map(item => item.id);
  const renderedIds = [...useCaseSelect.options].map(option => option.value);
  if (ids.join('|') === renderedIds.join('|')) return;
  const previous = selectedUseCase;
  useCaseSelect.replaceChildren();
  for (const useCase of data.use_cases) {
    const option = document.createElement('option');
    option.value = useCase.id;
    option.textContent = `${useCase.id} · ${useCase.type}`;
    useCaseSelect.appendChild(option);
  }
  selectedUseCase = ids.includes(previous) ? previous : data.primary_use_case;
  useCaseSelect.value = selectedUseCase;
}

function updateOverview(data) {
  const online = data.cameras.filter(camera => camera.online).length;
  const useCase = data.use_cases.find(item => item.id === selectedUseCase);
  const batchSize = useCase?.runtime?.batch_size;
  const batchElement = document.querySelector('#batch-size');
  document.querySelector('#camera-count').textContent = data.cameras.length;
  document.querySelector('#online-count').textContent = `${online}/${data.cameras.length}`;
  batchElement.textContent = batchSize?.value ?? '—';
  batchElement.title = batchSize?.source ?? '';
  document.querySelector('#shard').textContent = data.shard;
  document.querySelector('#transport').textContent = `${data.stream.transport} · ${data.stream.fps} FPS`;
  document.querySelector('#connection-label').textContent = 'System live';
  document.body.classList.add('connected');
  document.body.classList.remove('disconnected');
}

async function refreshStatus() {
  window.clearTimeout(updateTimer);
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Status ${response.status}`);
    const data = await response.json();
    const oldUseCase = selectedUseCase;
    streamFps = data.stream?.fps || 12;
    setUseCases(data);
    updateOverview(data);
    const cameraIds = new Set(data.cameras.map(camera => camera.id));
    removeMissingCards(cameraIds);
    for (const camera of data.cameras) {
      const card = cards.get(camera.id) || createCard(camera);
      updateCard(card, camera);
    }
    if (oldUseCase && oldUseCase !== selectedUseCase) reconnectAllStreams();
    emptyState.hidden = data.cameras.length > 0;
  } catch (_) {
    document.querySelector('#connection-label').textContent = 'Backend offline';
    document.body.classList.remove('connected');
    document.body.classList.add('disconnected');
  }
  updateTimer = window.setTimeout(refreshStatus, 1000);
}

function reconnectAllStreams() {
  for (const [cameraId, card] of cards) connectStream(card, cameraId);
}

useCaseSelect.addEventListener('change', () => {
  selectedUseCase = useCaseSelect.value;
  reconnectAllStreams();
  refreshStatus();
});

columnsSelect.addEventListener('change', () => {
  wall.dataset.columns = columnsSelect.value;
  localStorage.setItem('camera-wall-columns', columnsSelect.value);
});

document.querySelector('#wall-fullscreen').addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen();
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    for (const card of cards.values()) card.querySelector('.camera-stream').src = '';
  } else {
    reconnectAllStreams();
    refreshStatus();
  }
});

const savedColumns = localStorage.getItem('camera-wall-columns') || 'auto';
columnsSelect.value = savedColumns;
wall.dataset.columns = savedColumns;
window.setInterval(() => {
  document.querySelector('#clock').textContent = new Intl.DateTimeFormat(undefined, {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  }).format(new Date());
}, 1000);
refreshStatus();

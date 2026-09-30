// One visibility observer for all result images, with bounded retries and cleanup.
// Pruning before a grid replacement also cancels outstanding retry timers.
function createResultImageLoader({stallTimeout = 15000, retryDelays = [1000, 3000]} = {}) {
  const pending = new Map();
  const observer = new IntersectionObserver(entries => {
    entries.forEach(({target, isIntersecting}) => {
      if (!isIntersecting) return;
      observer.unobserve(target);
      const state = pending.get(target);
      if (state && !target.complete) state.arm();
      else if (state && !target.naturalWidth) state.retry();
    });
  });
  function cleanup(img) {
    const state = pending.get(img);
    if (!state) return;
    clearTimeout(state.timer);
    observer.unobserve(img);
    img.removeEventListener('load', state.loaded);
    img.removeEventListener('error', state.retry);
    pending.delete(img);
  }
  function observe(img) {
    if (pending.has(img)) return;
    const source = img.src;
    const state = {retries: 0, timer: null};
    state.loaded = () => cleanup(img);
    state.arm = () => {
      clearTimeout(state.timer);
      state.timer = setTimeout(state.retry, stallTimeout);
    };
    state.retry = () => {
      clearTimeout(state.timer);
      if (!img.isConnected || (img.complete && img.naturalWidth > 0)) return cleanup(img);
      if (state.retries >= retryDelays.length) {
        img.classList.add('image-unavailable');
        img.title = 'Preview unavailable. Open the frame to try again.';
        return cleanup(img);
      }
      state.timer = setTimeout(() => {
        if (!img.isConnected) return cleanup(img);
        img.src = source + (source.includes('?') ? '&' : '?') + `retry=${++state.retries}`;
        state.arm();
      }, retryDelays[state.retries]);
    };
    pending.set(img, state);
    img.decoding = 'async';
    img.addEventListener('load', state.loaded);
    img.addEventListener('error', state.retry);
    observer.observe(img);
    if (img.complete && img.naturalWidth > 0) cleanup(img);
  }
  return {
    observe,
    prune(container) {
      for (const img of pending.keys()) {
        if (!img.isConnected || container?.contains(img)) cleanup(img);
      }
    },
    dispose() {
      for (const img of pending.keys()) cleanup(img);
      observer.disconnect();
    },
  };
}
const resultImages = createResultImageLoader();
function setupResultImage(img) { resultImages.observe(img); }
window.addEventListener('pagehide', () => resultImages.dispose());

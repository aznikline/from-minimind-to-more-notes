/* ============================================================
   FMMTM Notes — 阅读交互(broadsheet 版)
   进度条 + 锚点平滑滚动 + 窄屏顶栏抽屉。无侧栏。
   ============================================================ */
(() => {
  'use strict';
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* 顶部进度条 */
  const bar = document.querySelector('.progress-bar');
  const onScroll = () => {
    if (!bar) return;
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0).toFixed(2) + '%';
  };
  document.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* 锚点平滑滚动 */
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href');
      if (!id || id === '#') return;
      const t = document.getElementById(id.slice(1));
      if (!t) return;
      e.preventDefault();
      t.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
      history.replaceState(null, '', id);
    });
  });

  /* 窄屏顶栏抽屉 */
  const toggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('.nav');
  toggle?.addEventListener('click', () => nav?.classList.toggle('is-open'));
  nav?.addEventListener('click', (e) => { if (e.target.closest('a')) nav.classList.remove('is-open'); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') nav?.classList.remove('is-open'); });
})();

/* ============================================================
   FMMTM Notes — 阅读交互
   侧栏当前篇/段高亮 + 阅读进度条 + 锚点平滑滚动 + 窄屏抽屉
   ============================================================ */
(() => {
  'use strict';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- 顶部阅读进度条 ---- */
  const bar = document.querySelector('.progress-bar');
  const onScroll = () => {
    if (!bar) return;
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    const pct = max > 0 ? (h.scrollTop / max) * 100 : 0;
    bar.style.width = pct.toFixed(2) + '%';
  };
  document.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* ---- IntersectionObserver: 当前篇高亮 + 当前段高亮 ---- */
  // 篇级:侧栏对应当前页的 link 标 is-current
  const currentHref = location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.toc__link').forEach(a => {
    const href = a.getAttribute('href') || '';
    if (href === currentHref || href.endsWith('/' + currentHref)) {
      a.classList.add('is-current');
      // 展开该篇的八段子锚点
      const sub = a.parentElement?.querySelector('.toc__sub');
      if (sub) sub.style.display = '';
    }
  });

  // 段级:H2 进入视口时高亮对应子锚点
  // sub-link 的 href 形如 "01-tokenizer.html#sec-1",取 # 后部分作元素 id
  const subLinks = Array.from(document.querySelectorAll('.toc__sub-link'));
  if (subLinks.length && 'IntersectionObserver' in window) {
    const sectionsById = new Map(); // key: 元素 id(如 "sec-1")
    subLinks.forEach(a => {
      const href = a.getAttribute('href') || '';
      const hashIdx = href.indexOf('#');
      if (hashIdx < 0) return;
      const eid = href.slice(hashIdx + 1);
      const el = document.getElementById(eid);
      if (el) sectionsById.set(eid, { el, link: a });
    });
    const visible = new Map();
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        const eid = e.target.id;
        if (sectionsById.has(eid)) {
          if (e.isIntersecting) visible.set(eid, e.intersectionRatio);
          else visible.delete(eid);
        }
      });
      // 选可见度最高的一段
      let best = null, bestRatio = 0;
      for (const [eid, r] of visible) {
        if (r > bestRatio) { bestRatio = r; best = eid; }
      }
      subLinks.forEach(a => {
        const href = a.getAttribute('href') || '';
        const eid = href.slice(href.indexOf('#') + 1);
        a.classList.toggle('is-current', eid === best);
      });
    }, { rootMargin: '-15% 0px -70% 0px', threshold: [0, 0.25, 0.5, 1] });
    for (const { el } of sectionsById.values()) io.observe(el);
  }

  /* ---- 锚点点击:平滑滚动 ---- */
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href');
      if (!id || id === '#') return;
      const target = document.getElementById(id.slice(1));
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
      history.replaceState(null, '', id);
    });
  });

  /* ---- 窄屏抽屉 ---- */
  const toggle = document.querySelector('.sidebar__toggle');
  const sidebar = document.querySelector('.sidebar');
  const backdrop = document.querySelector('.sidebar-backdrop');
  const openDrawer = () => { sidebar?.classList.add('is-open'); backdrop?.classList.add('is-open'); };
  const closeDrawer = () => { sidebar?.classList.remove('is-open'); backdrop?.classList.remove('is-open'); };
  toggle?.addEventListener('click', () => {
    sidebar?.classList.contains('is-open') ? closeDrawer() : openDrawer();
  });
  backdrop?.addEventListener('click', closeDrawer);
  // 抽屉内点链接后自动关
  sidebar?.addEventListener('click', (e) => {
    if (e.target.closest('a')) closeDrawer();
  });
  // ESC 关
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && sidebar?.classList.contains('is-open')) closeDrawer();
  });
})();

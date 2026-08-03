#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FMMTM Notes 静态阅读站构建脚本.

读所有 NN_*.md (17 篇) -> 渲染 HTML -> 套纸墨布局模板 -> 输出 docs/*.html
(输出到 docs/ 以便 GitHub Pages 直接从 main/docs 服)

- 优先用 python-markdown (含 fenced_code/tables/toc/attr_list)
- 无依赖回退:内置极简渲染器覆盖 heading/list/para/code/table/blockquote
- Mermaid:保留 ```mermaid 代码块,前端 mermaid.js 渲染
- LaTeX:保留 $$...$$ 与 $...$,前端 KaTeX auto-render
- 特殊 token (<|im_end|> </answer> 等):python-markdown 自动转义;回退器手动转义

用法: python3 docs/build.py
"""
from __future__ import annotations
import os
import re
import sys
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 仓库根
OUT = Path(__file__).resolve().parent           # docs/ (输出目录)

try:
    import markdown as _md
    _HAS_MD = True
except Exception:
    _HAS_MD = False


def _escape(s: str) -> str:
    return html.escape(s, quote=False)


def _inline(s: str) -> str:
    s = _escape(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<em>\1</em>', s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    # [text](url) -> <a href=url>text</a>(url 仅 http/https/相对,防注入)
    def _link(m):
        text = m.group(1)
        url = m.group(2)
        if re.match(r'^(https?:|/|#|mailto:)', url):
            return f'<a href="{_escape(url)}" target="_blank" rel="noopener">{text}</a>'
        return m.group(0)
    s = re.sub(r'\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)', _link, s)
    return s


def _fallback_render(md_text: str) -> str:
    lines = md_text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    sec_counter = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()

        m = re.match(r'^```(\w*)\s*$', stripped)
        if m:
            lang = m.group(1) or 'text'
            i += 1
            code_lines: list[str] = []
            while i < n and not re.match(r'^```\s*$', lines[i].strip()):
                code_lines.append(lines[i])
                i += 1
            i += 1
            cls = ' class="language-' + lang + '"' if lang != 'text' else ''
            out.append(f'<pre><code{cls}>{_escape(chr(10).join(code_lines))}</code></pre>')
            continue

        if stripped.startswith('|') and i + 1 < n and re.match(r'^\s*\|[\s:|-]+\|\s*$', lines[i + 1]):
            header = [c.strip() for c in stripped.strip('|').split('|')]
            i += 2
            rows: list[list[str]] = []
            while i < n and lines[i].strip().startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            t = ['<table>', '<thead><tr>']
            t.append(''.join(f'<th>{_inline(h)}</th>' for h in header))
            t.append('</tr></thead><tbody>')
            for row in rows:
                t.append('<tr>' + ''.join(f'<td>{_inline(c)}</td>' for c in row) + '</tr>')
            t.append('</tbody></table>')
            out.append(''.join(t))
            continue

        m = re.match(r'^(#{1,6})\s+(.+?)\s*$', stripped)
        if m:
            level = len(m.group(1))
            text = _inline(m.group(2))
            if level == 1:
                i += 1
                continue
            sec_counter += 1
            out.append(f'<h{level} id="sec-{sec_counter}">{text}</h{level}>')
            i += 1
            continue

        if stripped.startswith('>'):
            bq_lines: list[str] = []
            while i < n and lines[i].strip().startswith('>'):
                bq_lines.append(re.sub(r'^>\s?', '', lines[i].strip()))
                i += 1
            out.append(f'<blockquote>{_inline(" ".join(bq_lines))}</blockquote>')
            continue

        if re.match(r'^[-*]\s+', stripped):
            items: list[str] = []
            while i < n and re.match(r'^\s*[-*]\s+', lines[i]):
                items.append(re.sub(r'^\s*[-*]\s+', '', lines[i].rstrip()))
                i += 1
            out.append('<ul>' + ''.join(f'<li>{_inline(it)}</li>' for it in items) + '</ul>')
            continue

        if re.match(r'^\d+\.\s+', stripped):
            items2: list[str] = []
            while i < n and re.match(r'^\s*\d+\.\s+', lines[i]):
                items2.append(re.sub(r'^\s*\d+\.\s+', '', lines[i].rstrip()))
                i += 1
            out.append('<ol>' + ''.join(f'<li>{_inline(it)}</li>' for it in items2) + '</ol>')
            continue

        if not stripped:
            i += 1
            continue

        para: list[str] = []
        while i < n:
            cur = lines[i]
            s = cur.strip()
            if (not s or re.match(r'^#{1,6}\s', s) or re.match(r'^[-*]\s', s)
                    or re.match(r'^\d+\.\s', s) or s.startswith('>') or s.startswith('|')
                    or re.match(r'^```', s)):
                break
            para.append(cur.rstrip())
            i += 1
        if para:
            out.append(f'<p>{_inline(" ".join(para))}</p>')
        else:
            i += 1
    return chr(10).join(out)


_MD_EXTS = ['fenced_code', 'tables', 'toc', 'attr_list', 'sane_lists']


def render_markdown(text: str) -> str:
    if _HAS_MD:
        return _md.markdown(text, extensions=_MD_EXTS, output_format='html5')
    return _fallback_render(text)


GROUPS = [
    ('基石', [1, 2, 3]),
    ('架构', [4, 5, 6, 7, 8]),
    ('算法', [9, 10, 11, 12, 13, 14, 15]),
    ('求职', [16]),
]


def slug_of(path: Path) -> str:
    name = re.sub(r'_', '-', path.stem)
    return name + '.html'


def title_of(path: Path, body_html: str, md_text: str = '') -> str:
    if md_text:
        first = md_text.lstrip().splitlines()[0] if md_text.strip() else ''
        m = re.match(r'^#\s+(.+?)\s*$', first)
        if m:
            return m.group(1).strip()
    m = re.search(r'<h1[^>]*>(.+?)</h1>', body_html, re.S)
    if m:
        return re.sub(r'<[^>]+>', '', m.group(1)).strip()
    stem = path.stem
    m2 = re.match(r'^\d+_(.+)$', stem)
    return m2.group(1) if m2 else stem


def lead_of(body_html: str) -> str:
    m = re.search(r'<h2[^>]*>[^<]*一句话精炼[^<]*</h2>\s*(?:<p>(.+?)</p>)?', body_html, re.S)
    if m and m.group(1):
        return m.group(1).strip()
    m2 = re.search(r'<p>(.+?)</p>', body_html, re.S)
    return m2.group(1).strip() if m2 else ''


def sections_of(body_html: str) -> list[tuple[str, str]]:
    out = []
    for m in re.finditer(r'<h2[^>]*id="([^"]*)"[^>]*>(.+?)</h2>', body_html, re.S):
        out.append((m.group(1), re.sub(r'<[^>]+>', '', m.group(2)).strip()))
    return out


def critique_markup(body_html: str) -> str:
    pat = re.compile(r'(<h2[^>]*>[^<]*批判性批注[^<]*</h2>)(.*?)(?=<h2|\Z)', re.S)

    def wrap(m: re.Match) -> str:
        return f'<div class="section-critique">{m.group(1)}{m.group(2)}</div>'
    return pat.sub(wrap, body_html)


def build_header(notes: list[dict]) -> str:
    """SA 站风顶栏:brand + 横排 nav(总图 / 01-16 篇号 + 标签)。"""
    by_num = {n['num']: n for n in notes if 'num' in n}
    parts = ['<header class="site-header">',
             '<div class="header-inner">',
             '<a class="brand" href="index.html">From Minimind to More<small>精读结构化笔记</small></a>',
             '<nav class="nav">']
    overview = next((n for n in notes if n['slug'] == 'overview.html'), None)
    if overview:
        parts.append(f'<a href="{overview["slug"]}">总图</a>')
        parts.append('<span class="nav-sep">·</span>')
    for gname, nums in GROUPS:
        parts.append(f'<span class="nav-group">{gname}</span>')
        for num in nums:
            n = by_num.get(num)
            if not n:
                continue
            parts.append(f'<a href="{n["slug"]}" title="{html.escape(n["title"])}">{num:02d}</a>')
    parts.append('</nav>')
    parts.append('<button class="nav-toggle" aria-label="切换目录">☰</button>')
    parts.append('</div></header>')
    return chr(10).join(parts)


def mermaid_wrap(body_html: str) -> str:
    return re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        r'<div class="mermaid-wrap"><div class="mermaid">\1</div></div>',
        body_html, flags=re.S
    )


PAGE_TPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title} — From Minimind to More</title>
<meta name="description" content="{desc}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Benne&display=swap">
<link rel="stylesheet" href="assets/style.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css">
</head>
<body>
<div class="progress-bar" role="progressbar" aria-label="阅读进度"></div>
{header}
<main>
<article class="prose">
<div class="reader-head">
{crumb}
<h1>{title}</h1>
{lead}
</div>
{body}
{pager}
</article>
</main>
<footer>
<p>对 <a href="https://github.com/Tongyun1/from-minimind-to-more" target="_blank" rel="noopener">from-minimind-to-more</a>（minimind / jingyaogong）的逐篇精读结构化笔记。非官方，内容版权归原作者。minimind 是极简教学实现，与工业标准有差距，遇关键论断请回原论文核对。</p>
</footer>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{ startOnLoad: true, theme: 'neutral', fontFamily: 'Benne, Georgia, serif' }});
</script>
<script>
  if (window.hljs) hljs.highlightAll();
  if (window.renderMathInElement) {{
    renderMathInElement(document.body, {{
      delimiters: [
        {{left: '$$', right: '$$', display: true}},
        {{left: '$', right: '$', display: false}}
      ],
      ignoredTags: ['script','noscript','style','textarea','pre','code'],
      throwOnError: false
    }});
  }}
</script>
<script src="assets/nav.js"></script>
</body>
</html>
"""


def build_page(notes, current: dict, prev: dict | None, nxt: dict | None) -> str:
    body = render_markdown(current['text'])
    body = mermaid_wrap(body)
    body = critique_markup(body)
    title = current['title'] or current['stem']
    lead = lead_of(body)
    lead_html = f'<p class="lead">{lead}</p>' if lead else ''
    crumb_html = (f'<div class="crumb"><a href="index.html">笔记</a> / '
                  f'{html.escape(current.get("group",""))}</div>') if current.get('group') else ''
    prev_html = (f'<div class="pager__prev"><a href="{prev["slug"]}">'
                 f'<span class="pager__label">← 上一篇</span>{html.escape(prev["title"])}</a></div>') if prev else '<div></div>'
    nxt_html = (f'<div class="pager__next"><a href="{nxt["slug"]}">'
                f'<span class="pager__label">下一篇 →</span>{html.escape(nxt["title"])}</a></div>') if nxt else '<div></div>'
    pager = f'<nav class="pager">{prev_html}{nxt_html}</nav>'
    header = build_header(notes)
    return PAGE_TPL.format(
        page_title=html.escape(title),
        desc=html.escape(lead[:140]),
        header=header,
        crumb=crumb_html,
        title=html.escape(title),
        lead=lead_html,
        body=body,
        pager=pager,
    )


def build_landing(notes) -> str:
    paths = [
        ('新手', '02 → 01 → 03 → 04 → 07 → 10 → 11 → 09 → 16', 'overview.html'),
        ('求职冲刺', '16 → 04/05/06 → 09 → 12/13/14 → 速查表', 'overview.html'),
        ('算法岗', '09 → 12 → 14 → 15 → 13 → 10/11', 'overview.html'),
        ('工程/推理', '05 → 08 → 06 → 07 → 速查表', 'overview.html'),
    ]
    by_num = {n['num']: n for n in notes if 'num' in n}
    path_cards = ''.join(
        f'<a class="card" href="{href}"><h3>{name}</h3><div class="sub">{seq}</div></a>'
        for name, seq, href in paths
    )
    # 各篇卡片
    note_cards_parts = []
    for num in range(1, 17):
        n = by_num.get(num)
        if not n:
            continue
        note_cards_parts.append(
            f'<a class="card" href="{n["slug"]}"><h3>{num:02d} · {html.escape(n["title"])}</h3>'
            f'<div class="sub">{html.escape(n["lead"][:60])}{"…" if len(n["lead"])>60 else ""}</div>'
            f'<div class="meta">{html.escape(n["group"])}</div></a>'
        )
    note_cards = ''.join(note_cards_parts)
    header = build_header(notes)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>From Minimind to More — 精读结构化笔记</title>
<meta name="description" content="对 from-minimind-to-more 全 17 篇正文的逐篇精读合成笔记。minimind 26M 教学级 LLM，Tokenizer→架构→训练算法→对齐→求职一条龙。">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Benne&display=swap">
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="progress-bar"></div>
{header}
<main>
  <section class="hero">
    <h1>From Minimind to More</h1>
    <p>对 from-minimind-to-more 全 17 篇正文的逐篇精读合成笔记。</p>
    <p class="src"><a href="https://github.com/Tongyun1/from-minimind-to-more" target="_blank" rel="noopener">github.com/Tongyun1/from-minimind-to-more</a> · 16 篇正文 + 总图 · 含思维导图 / 公式 / 源码要点 / 批判性批注</p>
  </section>

  <h2 class="section-title">阅读路径</h2>
  <div class="cards">
    {path_cards}
  </div>

  <h2 class="section-title">各篇精读</h2>
  <div class="cards">
    {note_cards}
  </div>
</main>
<footer>
<p>对 <a href="https://github.com/Tongyun1/from-minimind-to-more" target="_blank" rel="noopener">from-minimind-to-more</a>（minimind / jingyaogong）的逐篇精读结构化笔记。非官方，内容版权归原作者。</p>
</footer>
<script src="assets/nav.js"></script>
</body>
</html>"""


def main():
    md_files = sorted([p for p in ROOT.glob('[0-9][0-9]_*.md')])
    if not md_files:
        print('错误:未在仓库根找到 NN_*.md', file=sys.stderr)
        sys.exit(1)

    # 清空 OUT 下旧 .html(保留 assets/)
    for old in OUT.glob('*.html'):
        old.unlink()

    print(f'渲染 {len(md_files)} 篇笔记 -> {OUT}/')
    notes = []
    for p in md_files:
        text = p.read_text(encoding='utf-8')
        body = render_markdown(text)
        body = mermaid_wrap(body)
        body = critique_markup(body)
        sections = sections_of(body)
        m = re.match(r'(\d+)_', p.stem)
        num = int(m.group(1)) if m else 0
        group = ''
        for gname, nums in GROUPS:
            if num in nums:
                group = gname
                break
        if num == 0:
            group = '总图'
        title = title_of(p, body, md_text=text)
        body_no_h1 = re.sub(r'<h1[^>]*>.+?</h1>', '', body, count=1, flags=re.S)
        notes.append({
            'slug': slug_of(p), 'stem': p.stem, 'num': num, 'group': group,
            'title': title, 'text': text, 'sections': sections,
            'body': body_no_h1, 'lead': lead_of(body),
        })

    notes_sorted = sorted(notes, key=lambda n: n['num'])
    for i, n in enumerate(notes_sorted):
        prev = notes_sorted[i - 1] if i > 0 else None
        nxt = notes_sorted[i + 1] if i < len(notes_sorted) - 1 else None
        page = build_page(notes_sorted, n, prev, nxt)
        (OUT / n['slug']).write_text(page, encoding='utf-8')
        print(f'  ✓ {n["slug"]:<28} {n["title"][:30]}')
    (OUT / 'index.html').write_text(build_landing(notes), encoding='utf-8')
    print(f'  ✓ index.html')
    print(f'\n完成: {OUT}/index.html')


if __name__ == '__main__':
    main()

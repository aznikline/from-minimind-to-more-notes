#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FMMTM Notes 静态阅读站构建脚本.

读所有 NN_*.md (17 篇) -> 渲染 HTML -> 套纸墨布局模板 -> 输出 site/*.html
- 优先用 python-markdown (含 fenced_code/tables/toc/attr_list)
- 无依赖回退:内置极简渲染器覆盖 heading/list/para/code/table/blockquote
- Mermaid:保留 ```mermaid 代码块,前端 mermaid.js 渲染
- LaTeX:保留 $$...$$ 与 $...$,前端 KaTeX auto-render
- 特殊 token (<|im_end|> </answer> 等):python-markdown 自动转义;回退器手动转义

用法: python3 site/build.py
"""
from __future__ import annotations
import os
import re
import sys
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 仓库根
SITE = Path(__file__).resolve().parent         # site/

# ---- markdown 依赖探测 ----
try:
    import markdown as _md
    _HAS_MD = True
except Exception:
    _HAS_MD = False


# ============================================================
# 极简回退渲染器(无依赖时兜底)
# 覆盖:heading/paragraph/ul/ol/code-fence/table/blockquote/inline
# ============================================================
def _escape(s: str) -> str:
    return html.escape(s, quote=False)


def _inline(s: str) -> str:
    """inline: 转义 + 粗排 bold/code. 不做链接(笔记内多为文件名)."""
    s = _escape(s)
    # **bold**
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    # *italic* / _italic_ (保守,只处理成对 *)
    s = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<em>\1</em>', s)
    # `code`
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    return s


_FALLBACK_TABLES = {}


def _fallback_render(md_text: str) -> str:
    """极简渲染:按行状态机. 输出 HTML body 字符串.

    覆盖:heading/paragraph/ul/ol/code-fence/table/blockquote/inline bold/code/italic.
    H1 跳过(标题由布局单独渲染). H2/H3 生成稳定 id: sec-1, sec-2 ...
    特殊 token(<|im_end|> </answer> 等)走 _escape 转义.
    """
    lines = md_text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    sec_counter = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 代码块(fenced)
        m = re.match(r'^```(\w*)\s*$', stripped)
        if m:
            lang = m.group(1) or 'text'
            i += 1
            code_lines: list[str] = []
            while i < n and not re.match(r'^```\s*$', lines[i].strip()):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过闭合 ```
            cls = ' class="language-' + lang + '"' if lang != 'text' else ''
            code_html = _escape('\n'.join(code_lines))
            out.append(f'<pre><code{cls}>{code_html}</code></pre>')
            continue

        # 表格(| ... | 表头 + |---| 分隔行)
        if stripped.startswith('|') and i + 1 < n and re.match(r'^\s*\|[\s:|-]+\|\s*$', lines[i + 1]):
            header = [c.strip() for c in stripped.strip('|').split('|')]
            i += 2  # 跳过分隔行
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

        # 标题
        m = re.match(r'^(#{1,6})\s+(.+?)\s*$', stripped)
        if m:
            level = len(m.group(1))
            text = _inline(m.group(2))
            if level == 1:
                # H1 跳过(由 .article__title 单独渲染)
                i += 1
                continue
            sec_counter += 1
            sid = f'sec-{sec_counter}'
            out.append(f'<h{level} id="{sid}">{text}</h{level}>')
            i += 1
            continue

        # 引用
        if stripped.startswith('>'):
            bq_lines: list[str] = []
            while i < n and lines[i].strip().startswith('>'):
                bq_lines.append(re.sub(r'^>\s?', '', lines[i].strip()))
                i += 1
            out.append(f'<blockquote>{_inline(" ".join(bq_lines))}</blockquote>')
            continue

        # 无序列表(- 或 * 开头;支持子项缩进)
        if re.match(r'^[-*]\s+', stripped):
            items: list[str] = []
            while i < n and re.match(r'^\s*[-*]\s+', lines[i]):
                items.append(re.sub(r'^\s*[-*]\s+', '', lines[i].rstrip()))
                i += 1
            out.append('<ul>' + ''.join(f'<li>{_inline(it)}</li>' for it in items) + '</ul>')
            continue

        # 有序列表
        if re.match(r'^\d+\.\s+', stripped):
            items2: list[str] = []
            while i < n and re.match(r'^\s*\d+\.\s+', lines[i]):
                items2.append(re.sub(r'^\s*\d+\.\s+', '', lines[i].rstrip()))
                i += 1
            out.append('<ol>' + ''.join(f'<li>{_inline(it)}</li>' for it in items2) + '</ol>')
            continue

        # 空行
        if not stripped:
            i += 1
            continue

        # 段落(连续非空行合并,直到遇块级语法)
        para: list[str] = []
        while i < n:
            cur = lines[i]
            s = cur.strip()
            if (not s
                or re.match(r'^#{1,6}\s', s)
                or re.match(r'^[-*]\s', s)
                or re.match(r'^\d+\.\s', s)
                or s.startswith('>')
                or s.startswith('|')
                or re.match(r'^```', s)):
                break
            para.append(cur.rstrip())
            i += 1
        if para:
            out.append(f'<p>{_inline(" ".join(para))}</p>')
        else:
            i += 1
    return '\n'.join(out)


# ============================================================
# Markdown 渲染(优先 python-markdown)
# ============================================================
_MD_EXTS = ['fenced_code', 'tables', 'toc', 'attr_list', 'sane_lists']


def render_markdown(text: str) -> str:
    if _HAS_MD:
        return _md.markdown(text, extensions=_MD_EXTS, output_format='html5')
    return _fallback_render(text)


# ============================================================
# 笔记元数据
# ============================================================
GROUPS = [
    ('基石', [1, 2, 3]),
    ('架构', [4, 5, 6, 7, 8]),
    ('算法', [9, 10, 11, 12, 13, 14, 15]),
    ('求职', [16]),
]


def slug_of(path: Path) -> str:
    name = path.stem  # 00_OVERVIEW / 01_tokenizer
    # NN_topic -> NN-topic
    name = re.sub(r'_', '-', name)
    return name + '.html'


def title_of(path: Path, body_html: str, md_text: str = '') -> str:
    """首个 H1. 优先从 markdown 源首行 # 取(回退器会丢掉 H1)."""
    # 1. markdown 源首行 # 标题
    if md_text:
        first = md_text.lstrip().splitlines()[0] if md_text.strip() else ''
        m = re.match(r'^#\s+(.+?)\s*$', first)
        if m:
            return m.group(1).strip()
    # 2. HTML 里的 H1(python-markdown 保留时)
    m = re.search(r'<h1[^>]*>(.+?)</h1>', body_html, re.S)
    if m:
        return re.sub(r'<[^>]+>', '', m.group(1)).strip()
    # 3. 文件名兜底
    stem = path.stem
    m2 = re.match(r'^\d+_(.+)$', stem)
    return m2.group(1) if m2 else stem


def lead_of(body_html: str) -> str:
    """一句话精炼:H2 一句话精炼 下的首段."""
    # 找 <h2 id="...一句话精炼..."> 后的第一个 <p>
    m = re.search(r'<h2[^>]*>[^<]*一句话精炼[^<]*</h2>\s*(?:<p>(.+?)</p>)?', body_html, re.S)
    if m and m.group(1):
        return m.group(1).strip()
    # 回退:第一个 <p>
    m2 = re.search(r'<p>(.+?)</p>', body_html, re.S)
    return m2.group(1).strip() if m2 else ''


def sections_of(body_html: str) -> list[tuple[str, str]]:
    """提取 H2 id+text 作为篇内锚点."""
    out = []
    for m in re.finditer(r'<h2[^>]*id="([^"]*)"[^>]*>(.+?)</h2>', body_html, re.S):
        sid = m.group(1)
        text = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        out.append((sid, text))
    return out


def critique_markup(body_html: str) -> str:
    """把 批判性批注 H2 段包裹成 .section-critique."""
    # 匹配 <h2 ...id=...>...批判性批注...</h2> 到下一个 <h2 或文末
    pat = re.compile(
        r'(<h2[^>]*>[^<]*批判性批注[^<]*</h2>)(.*?)(?=<h2|\Z)',
        re.S
    )

    def wrap(m: re.Match) -> str:
        head = m.group(1)
        rest = m.group(2)
        return f'<div class="section-critique">{head}{rest}</div>'
    return pat.sub(wrap, body_html)


# ============================================================
# 侧栏 + 布局模板
# ============================================================
def build_sidebar(notes: list[dict], current_slug: str) -> str:
    """notes: [{slug,title,group,sections}]"""
    parts = ['<aside class="sidebar" aria-label="目录">',
              '<button class="sidebar__toggle" aria-label="开关目录">☰</button>',
              '<div class="sidebar__brand"><a href="index.html">From Minimind<br/>to More</a></div>',
              '<div class="sidebar__sub">精读结构化笔记 · 16+1 篇</div>']

    # 总图单独置顶
    overview = next((n for n in notes if n['slug'] == 'overview.html'), None)
    if overview:
        is_cur = 'is-current' if current_slug == overview['slug'] else ''
        parts.append(f'<nav class="toc"><div class="toc__group">总图</div>'
                     f'<div class="toc__item"><a class="toc__link {is_cur}" href="{overview["slug"]}">'
                     f'<span class="toc__num">00</span><span>{overview["title"]}</span></a></div></nav>')

    by_num = {n['num']: n for n in notes if 'num' in n}
    for gname, nums in GROUPS:
        parts.append(f'<nav class="toc"><div class="toc__group">{gname}</div>')
        for num in nums:
            n = by_num.get(num)
            if not n:
                continue
            is_cur = 'is-current' if current_slug == n['slug'] else ''
            sub_style = '' if current_slug == n['slug'] else ' style="display:none"'
            sub = ''.join(
                f'<li><a class="toc__sub-link" href="{n["slug"]}#{sid}">{txt}</a></li>'
                for sid, txt in n['sections']
            ) if current_slug == n['slug'] else ''
            parts.append(
                f'<div class="toc__item">'
                f'<a class="toc__link {is_cur}" href="{n["slug"]}">'
                f'<span class="toc__num">{n["num"]:02d}</span><span>{n["title"]}</span></a>'
                + (f'<ul class="toc__sub"{sub_style}>{sub}</ul>' if sub else '')
                + '</div>'
            )
        parts.append('</nav>')
    parts.append('</aside>')
    return '\n'.join(parts)


def mermaid_wrap(body_html: str) -> str:
    """给 mermaid pre>code 包一层 wrap 便于样式."""
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
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<link rel="stylesheet" href="assets/style.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css">
</head>
<body>
<div class="progress-bar" role="progressbar" aria-label="阅读进度"></div>
<div class="sidebar-backdrop"></div>
<div class="layout">
{sidebar}
<main class="content">
<article class="article">
{crumb}
<h1 class="article__title">{title}</h1>
{lead}
{body}
{pager}
</article>
</main>
</div>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{ startOnLoad: true, theme: 'neutral', fontFamily: 'Noto Sans SC, sans-serif' }});
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
    sections = sections_of(body)
    title = current['title'] or current['stem']
    lead = lead_of(body)
    lead_html = f'<p class="article__lead">{lead}</p>' if lead else ''
    crumb_html = (f'<div class="article__crumb"><a href="index.html">总图</a> / '
                  f'{current.get("group","")}</div>') if current.get('group') else ''

    # pager
    prev_html = (f'<div class="pager__prev"><a href="{prev["slug"]}">'
                 f'<span class="pager__label">← 上一篇</span>{prev["title"]}</a></div>') if prev else '<div></div>'
    nxt_html = (f'<div class="pager__next" style="text-align:right"><a href="{nxt["slug"]}">'
                f'<span class="pager__label">下一篇 →</span>{nxt["title"]}</a></div>') if nxt else '<div></div>'
    pager = f'<nav class="pager">{prev_html}{nxt_html}</nav>'

    sidebar = build_sidebar([{**n, 'sections': (sections if n is current else n.get('sections', []))} for n in notes], current['slug'])

    return PAGE_TPL.format(
        page_title=html.escape(title),
        desc=html.escape(lead[:140]),
        sidebar=sidebar,
        crumb=crumb_html,
        title=html.escape(title),
        lead=lead_html,
        body=body,
        pager=pager,
    )


# ============================================================
# 首页
# ============================================================
def build_landing(notes) -> str:
    paths = [
        ('新手', '02 → 01 → 03 → 04 → 07 → 10 → 11 → 09 → 16'),
        ('求职冲刺', '16 → 04/05/06 → 09 → 12/13/14 → 速查表'),
        ('算法岗', '09 → 12 → 14 → 15 → 13 → 10/11'),
        ('工程/推理', '05 → 08 → 06 → 07 → 速查表'),
    ]
    path_html = ''.join(
        f'<a class="landing__path" href="overview.html"><b>{name}</b><span>{seq}</span></a>'
        for name, seq in paths
    )
    landing_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>From Minimind to More — 精读结构化笔记</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="progress-bar"></div>
<main class="landing">
  <p class="article__crumb"><a href="overview.html">进入总图 →</a></p>
  <h1 class="landing__title">From Minimind<br/>to More</h1>
  <p class="landing__sub">对 from-minimind-to-more 全 17 篇正文的逐篇精读合成笔记。
  minimind 是 26M 的「从零训练 LLM」教学项目，以它为骨架，把 Tokenizer → 架构 → 训练算法 → 对齐 → 求职 一条龙讲透。</p>
  <div class="landing__paths">
    <p class="article__crumb" style="margin:0 0 0.4rem">阅读路径</p>
    {path_html}
  </div>
  <a class="landing__cta" href="overview.html">从总图开始 →</a>
</main>
<script src="assets/nav.js"></script>
</body>
</html>"""
    return landing_html


# ============================================================
# 主流程
# ============================================================
def main():
    md_files = sorted([p for p in ROOT.glob('[0-9][0-9]_*.md')])
    if not md_files:
        print('错误:未在仓库根找到 NN_*.md', file=sys.stderr)
        sys.exit(1)

    print(f'渲染 {len(md_files)} 篇笔记 ...')
    notes = []
    for p in md_files:
        text = p.read_text(encoding='utf-8')
        body = render_markdown(text)
        body = mermaid_wrap(body)
        body = critique_markup(body)
        sections = sections_of(body)
        m = re.match(r'(\d+)_', p.stem)
        num = int(m.group(1)) if m else 0
        # 分组
        group = ''
        for gname, nums in GROUPS:
            if num in nums:
                group = gname
                break
        if num == 0:
            group = '总图'
        title = title_of(p, body, md_text=text)
        # 清掉 H1(已单独渲染为 .article__title)
        body_no_h1 = re.sub(r'<h1[^>]*>.+?</h1>', '', body, count=1, flags=re.S)
        notes.append({
            'slug': slug_of(p),
            'stem': p.stem,
            'num': num,  # 0 = 总图, 1-16 = 正文
            'group': group,
            'title': title,
            'text': text,
            'sections': sections,
            'body': body_no_h1,
            'lead': lead_of(body),
        })

    # 排序:总图(00)在前,正文按 num 1..16
    notes_sorted = sorted(notes, key=lambda n: n['num'])
    # prev/next 链按阅读序:overview -> 01..16
    ordered = [n for n in notes_sorted]

    for i, n in enumerate(notes_sorted):
        prev = ordered[i - 1] if i > 0 else None
        nxt = ordered[i + 1] if i < len(ordered) - 1 else None
        page = build_page(notes_sorted, n, prev, nxt)
        out = SITE / n['slug']
        out.write_text(page, encoding='utf-8')
        print(f'  ✓ {n["slug"]:<28} {n["title"][:30]}')

    # 首页
    (SITE / 'index.html').write_text(build_landing(notes), encoding='utf-8')
    print('  ✓ index.html')

    print(f'\n完成: {SITE}/index.html')


if __name__ == '__main__':
    main()

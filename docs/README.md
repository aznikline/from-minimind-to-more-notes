# FMMTM Notes 静态阅读站

`from-minimind-to-more` 精读笔记的纸墨编辑风静态阅读站。零框架、零构建工具链，手写 Python 脚本把 17 篇 Markdown 渲成静态 HTML。

**GitHub Pages**：从 `main` / `/docs` 目录服。即本目录。

## 重建

```bash
# 仓库根目录
bash docs/build.sh          # 或: python3 docs/build.py
```

输出 `docs/index.html` + `docs/*.html`(17 篇)。

## 依赖

- **Python 3.7+**，仅标准库。
- **可选** `pip install markdown`：若装了 python-markdown，构建用它(含 fenced_code/tables/toc/attr_list)；没装则用内置极简渲染器兜底，覆盖标题/列表/段落/代码块/表格/引用，照样可跑。
- **CDN 资源**(页面运行时)：KaTeX、highlight.js、mermaid.js、Google Fonts(Noto Serif SC / Noto Sans SC / JetBrains Mono)。需联网首次加载。

## 目录

```
docs/
├── build.py          构建脚本：.md -> .html 渲染管线 + 侧栏数据 + 模板
├── build.sh          一键构建入口
├── index.html        首页(项目定位 + 阅读路径卡片)
├── 00-OVERVIEW.html  总图
├── overview.html     总图
├── 01-16-*.html      16 篇正文
└── assets/
    ├── style.css     设计系统(token + 布局 + 组件 + 响应式 + 暗色)
    └── nav.js        进度条 + 锚点 + 窄屏抽屉
```

源笔记 `00-16_*.md` 在仓库根，**不修改**。改笔记后重跑 `build.sh` 即可刷新站点。

## 设计

- **风格**：参考 situational-awareness.ai 站设计语言——白底 / Benne + Noto Serif SC / 深灰正文 / 紫链接 / 细边框 / 大留白。
- **字体配对**：Benne(英文展示) + Noto Serif SC(中文正文) / Noto Sans SC(UI 标签) / JetBrains Mono(代码)。
- **暗色**：跟随系统 `prefers-color-scheme`，精修而非反色，强调色提亮。
- **交互**：顶部 sticky 导航(总图 + 01-16 分组) + 阅读进度条 + 窄屏抽屉。
- **内容渲染**：Mermaid(mindmap/flowchart 客户端渲染)、LaTeX(KaTeX auto-render)、代码(highlight.js)、1px 网格表格、批判性批注段(暗红卡片标记)。

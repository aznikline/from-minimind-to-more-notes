# FMMTM Notes 静态阅读站

`from-minimind-to-more` 精读笔记的纸墨编辑风静态阅读站。零框架、零构建工具链，手写 Python 脚本把 17 篇 Markdown 渲成静态 HTML。

## 重建

```bash
# 仓库根目录
bash site/build.sh          # 或: python3 site/build.py
```

输出 `site/index.html` + `site/*.html`(17 篇)。直接 `open site/index.html` 即可阅读。

## 依赖

- **Python 3.7+**，仅标准库。
- **可选** `pip install markdown`：若装了 python-markdown，构建会用它(含 fenced_code/tables/toc/attr_list)；没装则用内置极简渲染器兜底，覆盖标题/列表/段落/代码块/表格/引用，照样可跑。
- **CDN 资源**(页面运行时)：KaTeX、highlight.js、mermaid.js、Google Fonts(Noto Serif SC / Noto Sans SC / JetBrains Mono)。需联网首次加载。

## 部署 GitHub Pages

```bash
# 方式 A：直接推 site/ 目录(零 CI)
git add site && git commit -m "docs: 静态阅读站" && git push
# 仓库 Settings → Pages → Build and deployment:
#   Source: Deploy from a branch → Branch: main / root(或 /site)
```

若选 `/site` 子目录，Pages 会从 `site/index.html` 起服。

## 目录

```
site/
├── build.py          构建脚本：.md → .html 渲染管线 + 侧栏数据 + 模板
├── build.sh          一键构建入口
├── index.html        首页(项目定位 + 阅读路径卡片)
├── 00-OVERVIEW.html  总图
├── 01-16-*.html      16 篇正文
└── assets/
    ├── style.css     纸墨设计系统(token + 布局 + 组件 + 响应式 + 暗色)
    └── nav.js        侧栏高亮 + 进度条 + 锚点 + 窄屏抽屉
```

源笔记 `00-16_*.md` 在仓库根，**不修改**。改笔记后重跑 `build.sh` 即可刷新站点。

## 设计

- **纸墨编辑风**：米白纸底 + 深墨正文 + 暗赭强调 + 墨绿标记批判性批注段。
- **字体配对**：Noto Serif SC(衬线正文) / Noto Sans SC(UI 标签) / JetBrains Mono(代码)。
- **暗色**：跟随系统 `prefers-color-scheme`，非简单反色，强调色提亮、墨绿饱和调整。
- **交互**：左侧固定目录(17 篇 + 总图) + 篇内八段锚点子导航(IntersectionObserver 高亮当前段) + 顶部阅读进度条 + 窄屏抽屉。
- **内容渲染**：Mermaid(mindmap/flowchart 客户端渲染)、LaTeX(KaTeX auto-render)、代码(highlight.js)、表格(横线极简)、批判性批注段(墨绿卡片标记)。

## 设计意图(对照反模板标准)

不做默认 Tailwind/shadcn 模板感。层级靠字号对比(H1/H2/lead 阶差)，间距有节奏(段间/段内不同步长)，强调色语义化(暗赭=当前/进度，墨绿=批注反思)，表格横线极简(无竖线网格)，Mermaid/LaTeX 统一卡片框作设计系统一部分。

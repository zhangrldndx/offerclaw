# static/fonts/

`index.html` 里的 `@font-face` 指向本目录的四个文件。**四个文件已下载到位，页面无外网字体依赖**
（2026-09-01：下载后按下方说明移除了 `<head>` 里的三行 Google Fonts 链接——每次加载都向
`fonts.googleapis.com` 发请求，与本项目"本地优先"的定位相悖）。

文件意外缺失时回退到 `Georgia` / 系统无衬线，页面不会坏，只是失去 Caprasimo 展示字。

当前文件（合计约 55 KB）：

| 文件 | 大小 |
|---|---:|
| `caprasimo-latin-400.woff2` | 20.9 KB |
| `figtree-latin-400.woff2` | 11.4 KB |
| `figtree-latin-600.woff2` | 11.5 KB |
| `figtree-latin-700.woff2` | 11.4 KB |

## 重新下载（在本目录执行）

```bash
cd static/fonts

curl -L -o caprasimo-latin-400.woff2 \
  https://cdn.jsdelivr.net/fontsource/fonts/caprasimo@latest/latin-400-normal.woff2

curl -L -o figtree-latin-400.woff2 \
  https://cdn.jsdelivr.net/fontsource/fonts/figtree@latest/latin-400-normal.woff2

curl -L -o figtree-latin-600.woff2 \
  https://cdn.jsdelivr.net/fontsource/fonts/figtree@latest/latin-600-normal.woff2

curl -L -o figtree-latin-700.woff2 \
  https://cdn.jsdelivr.net/fontsource/fonts/figtree@latest/latin-700-normal.woff2
```

下载完不需要再改 `index.html`——那三行 Google Fonts 链接已经删除，`@font-face` 直接指向本目录：

```html
<!-- 已移除，勿再加回：
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Caprasimo&family=Figtree:wght@400;600;700&display=swap">
-->
```

## 一件事要知道

Caprasimo 和 Figtree 都只有拉丁字符集，没有中文字形。界面里的中文标题实际由系统字体渲染（PingFang SC / 微软雅黑），Caprasimo 只作用在 `OfferClaw`、`JD`、`Plan Agent`、`KB`、数字这些拉丁片段上。

如果希望中文标题也有明确的"展示字"声音，需要额外引一款中文字体（例如思源宋体 Noto Serif SC，700 字重），把 `--font-heading` 改成：

```css
--font-heading: "Caprasimo", "Noto Serif SC", Georgia, serif;
```

代价是中文字体子集体积大（全量 woff2 约 8-10 MB，需要按字符集分片或做子集化）。要做我可以帮你配。

# PC Web 壳前端

基于 Vite + React + TypeScript。设计变量与 `screen-recorder-mvp/design/tokens.json` 对齐（见 `src/index.css`）。

## 开发

```bash
cd screen-recorder-mvp/pc
pip install -r requirements.txt
python main_web.py
```

另开终端（Web 开发热更新，可选）：

```bash
cd screen-recorder-mvp/pc/webui
npm install
npm run dev
```

`vite.config.ts` 已将 `/api` 代理到 `http://127.0.0.1:8776`。

## 生产构建（嵌入 FastAPI）

```bash
cd screen-recorder-mvp/pc/webui
npm install
npm run build
```

生成 `webui/dist/`，由 `app_server.py` 挂载；未构建时仍可访问 `/docs` 调用 API。

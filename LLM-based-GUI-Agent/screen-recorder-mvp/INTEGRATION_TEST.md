# 双端联调与回归

本文件覆盖重构计划中的联调验收项：录屏、上传、分析、记忆。

## 1) PC Web 壳自动化冒烟

在 `screen-recorder-mvp/pc` 目录执行：

```bash
python scripts/smoke_web_bridge.py
```

检查点：

- `/api/health` 返回 `ok: true`
- `/api/status` 返回桥接状态（录屏/接收/分析/记忆相关字段）
- `/api/video/select` 对非法路径返回 400（防止错误输入穿透）

## 2) Android 手工回归（与 PC 联调）

前置：

1. PC 端启动 `python main_web.py`
2. 切到 Recording 页点击“启动上传服务”，记下 `http://<PC_IP>:8765/upload`
3. Android 与 PC 在同一 WiFi

步骤：

1. Android Recording 页：点击“开始录制”并授权，录制几秒后停止
2. Android Data 页：填写 `PC_IP` 和端口 `8765`，选择视频并上传
3. PC Web 壳 Data 页：刷新本地视频，确认收到上传文件
4. PC Web 壳 Settings 页：选择 local/api 后端，启动分析
5. PC Web 壳 Data 页：执行记忆检索，确认可返回片段

通过标准：

- 录屏文件可生成（非 0 字节）
- 上传返回“上传成功”
- 分析进度可见，结果目录可打开
- 记忆统计与检索可读

## 3) 设备分层建议（性能验收基线）

- 8GB 显存：`GUI-Owl-1.5-4B-Instruct`（默认更稳）
- 24GB 显存：`GUI-Owl-1.5-8B-Instruct`（效果更好）


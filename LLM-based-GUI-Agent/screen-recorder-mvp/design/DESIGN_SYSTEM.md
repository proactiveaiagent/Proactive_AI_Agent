# 双端共享设计系统（Design System）

本目录为 **PC Web 壳** 与 **Android** 共用的视觉与组件规范来源。具体色值以 `tokens.json` 为准；本文描述信息架构与组件契约。

## 信息架构（主导航）

| Tab ID      | 中文   | 职责 |
|------------|--------|------|
| `recording`| 录制   | 本机录屏开始/停止、状态、保存路径提示 |
| `earnings` | 收益   | 统计卡片、占位说明（可与后续商业化数据对接） |
| `data`     | 数据   | 选择本地录制文件、填写 PC IP/端口、上传到电脑 |
| `settings` | 设置   | 后端说明、版本、常见问题入口 |

## Design Tokens

- **源文件**：[`tokens.json`](./tokens.json)（JSON，可供 Web 构建脚本或文档生成引用）
- **Android**：[`../android/app/src/main/res/values/design_tokens.xml`](../android/app/src/main/res/values/design_tokens.xml)（颜色、圆角、间距别名，与 JSON 语义对齐）

### 颜色语义

- **canvas**：应用背景
- **surface / surfaceElevated**：卡片与顶栏
- **text.primary / secondary / muted**：正文层级
- **accent.primary**：主按钮、选中 Tab
- **success / warning / danger**：状态反馈

### 字号层级

- **xxl**：页面标题
- **xl / lg**：区块标题
- **md**：正文
- **sm / xs**：辅助说明、Caption

### 间距与圆角

- 卡片内边距优先 **16–24dp/sp**
- 卡片圆角 **12–16**
- 按钮圆角 **pill** 或 **12**

## 组件清单（双端对齐）

| 组件 | 用途 | Web | Android |
|------|------|-----|---------|
| **AppHeader** | 顶栏标题 + 可选副标题 | `App.tsx` header | `Toolbar` / `MaterialToolbar` |
| **StatsCard** | 单列指标：标题、数值、说明 | 收益 Tab | `MaterialCardView` |
| **ActionCard** | 主操作区：标题 + 主按钮 + 辅助文案 | 录制 / 数据 Tab | `MaterialCardView` + `Button` |
| **ToggleRow** | 开关行（未来扩展） | 预留 | `SwitchMaterial` |
| **BottomNav** | 四 Tab 主导航 | 底栏 nav | `BottomNavigationView` |
| **ListItem** | 文件行、设置行 | 数据列表（后续可扩展 `RecyclerView`） | `Spinner` / 列表项 |

## Web 引用方式

- PC 前端在 `pc/webui/src/index.css` 中以 CSS 变量镜像 `tokens.json` 中的 `color`、`radius`、`space`，保持视觉一致。

## Android 引用方式

- 使用 `R.color.ds_*`、`R.dimen.ds_*` 引用 `design_tokens.xml`，布局中避免硬编码与设计稿冲突的色值。

## 变更流程

1. 先改 `tokens.json` 并更新本说明中的表格（如有结构变化）。
2. 同步 `design_tokens.xml` 与 `pc/webui/src/index.css`。
3. 双端截图对比留白与字号层级。

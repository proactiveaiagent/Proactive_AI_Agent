# 人物跨片段身份关联

本模块只整理视频中已观察到的人物身份，不推断关系、年龄或行动建议。

## 证据与关联

- 优先读取人物 person_crop。若裁剪中没有可靠人脸，才在关联 source_asset 中按 crop_bbox 限定搜索区域，并只接受框中心落在该人物区域内的人脸；不会对无关整帧中的其他人做身份比较。
- 本地 InsightFace buffalo_l / ArcFace 在 CPU 上检测人脸并生成归一化向量。人物记录之间余弦相似度达到 MEMORY_FACE_SIMILARITY_THRESHOLD（默认 0.64）才自动建立跨片段链接。
- 两条记录有相同、带语音来源的 probable_visual_match 别名时，可辅助确认 0.52 以上的面部相似候选。口述称呼本身不会单独合并两个人。
- 同一来源片段中的人物不自动合并。自动链接保留相似度、证据图像和区域框；未检测到可靠人脸时保持未关联。

## 纠错

人物记忆卡上的“纠正身份关联”允许选择另一条人物记录，并执行“合并”或“拆分”。纠错写到共享交互累积记忆目录中的 person_identity_corrections.json，带操作历史；记忆正文和源事件不被覆盖。拆分对会作为硬约束参与后续关联，合并/拆分可重复修正。

人脸向量缓存单独写入 person_face_embedding_cache.json；人物裁剪按图像 SHA-256 缓存，原始关联画面的回退项按图像 SHA-256 与人物区域框共同缓存。两份文件均留在对应记忆运行目录，不上传图像或向量。

## 本地权重

运行时使用 MEMORY_FACEID_PYTHON 指定的独立 Python 环境。默认环境为服务器 /data/cxr25/zhx/runtime/faceid_py311。模型权重目录为 ~/.insightface/models/buffalo_l；只有检测与识别 ONNX 文件均完整时才会启用自动匹配。缺权重时 API 会返回 model_missing，不会尝试联网下载或把占位标签自动合并。当前 buffalo_l 权重已在服务器本地安装并通过官方 SHA-256 校验；若关联画面仍检测不到足够清晰的人脸，API 会返回 no_reliable_faces，不做自动合并。

InsightFace 官方说明模型包仅限非商业研究用途；该项目当前按研究用途使用该模型。模型与特征比对均在服务器本地运行。

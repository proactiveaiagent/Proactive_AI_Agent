const COPY = {
  zh: {
    "nav.company": "公司",
    "nav.products": "产品",
    "nav.pipeline": "能力",
    "nav.scenarios": "场景",
    "nav.contact": "联系我们",
    "nav.cta": "了解产品",
    "hero.eyebrow": "主动式人工智能公司",
    "hero.title": "感知当下，主动行动。",
    "hero.lede":
      "PROACTIVE INTELLIGENCE LIMITED 打造上下文感知助手：融合第一人称视频、屏幕活动与可穿戴信号，推断用户需求，并通过数字信息、设备操作和硬件指令主动交付帮助。",
    "hero.cta1": "查看产品",
    "hero.cta2": "工作原理",
    "stats.s1k": "输入模态",
    "stats.s2k": "输出通道",
    "stats.s3k": "记忆层级",
    "stats.s4k": "验证场景",
    "company.eyebrow": "关于我们",
    "company.title": "PROACTIVE INTELLIGENCE LIMITED",
    "company.lede":
      "我们相信助手不该只在被问到时才回答。它应该看见用户正在做什么，记住过往上下文，并在合适的时机主动提供下一步。",
    "company.c1t": "主动，而不是被动",
    "company.c1p":
      "Agent 持续感知场景与屏幕活动，推断当前最需要的事项，而不是等待一句明确指令。",
    "company.c2t": "多模态融合",
    "company.c2p":
      "智能眼镜/手机视频、PC 与 Android 录屏、心率与姿态等可穿戴数据可以单独使用，也可以一起进入同一条推理流水线。",
    "company.c3t": "记得，并跟进",
    "company.c3p":
      "七层记忆保存场景历史、用户偏好与过往建议；下一次会话会检查用户是否已经按上次方案行动。",
    "products.eyebrow": "产品线",
    "products.title": "一套感知系统，两条产品线",
    "products.p1tag": "核心产品",
    "products.p1p":
      "上下文感知助手。它读取视频、录屏与传感器，识别场景与需求，再把方案路由到 AR/通知、手机电脑操作，或智能家居与机器人。",
    "products.p1l1": "场景识别 → 需求分析 → 方案生成 → 多通道交付",
    "products.p1l2": "支持 camera / screen_record / gui_agent / wearable / multimodal",
    "products.p1l3": "视觉理解使用 Qwen3-VL，语音转写使用 Whisper",
    "products.p2tag": "集成能力",
    "products.p2p":
      "PC 与 Android（应用名 XOOGUIAGT）全天录屏系统。GUI-Owl 识别应用、页面、UI 元素和用户操作，再交给 Proactive Agent 做需求推断。",
    "products.p2l1": "PC Web 壳录屏 + Android 上传到电脑",
    "products.p2l2": "本地 GUI-Owl 或云端视觉模型分析",
    "products.p2l3": "分析结果写入记忆，支持检索与用户画像",
    "io.inEyebrow": "输入",
    "io.inTitle": "看见用户正在经历什么",
    "io.in1t": "用户拍摄视频",
    "io.in1p": "智能眼镜、手机摄像头或第一人称 POV 录像。",
    "io.in2t": "手机 / 电脑录屏",
    "io.in2p": "理解应用界面、页面跳转与操作流程。",
    "io.in3t": "GUI Agent",
    "io.in3p": "GUI-Owl 结构化识别应用名、页面、元素与动作。",
    "io.in4t": "可穿戴数据",
    "io.in4p": "心率、步数、位置、姿态；可与视频同时融合。",
    "io.outEyebrow": "输出",
    "io.outTitle": "把帮助送到真正能执行的地方",
    "io.out1t": "数字信息与应用服务",
    "io.out1p": "AR 叠加、通知、提示、应用内引导。",
    "io.out2t": "手机 / 电脑操作",
    "io.out2p": "打开应用、点击界面、输入文字、发送消息、浏览网页。",
    "io.out3t": "智能硬件与机器人",
    "io.out3p": "灯光、温控等家居控制，以及机器人动作指令。",
    "pipe.eyebrow": "工作原理",
    "pipe.title": "四阶段流水线",
    "pipe.lede": "帧提取、音频提取和转写只做一次，在预检查与主分析之间共享，避免重复计算。",
    "pipe.s0t": "预检查",
    "pipe.s0p": "对照上一轮建议，检查用户是否已经按方案行动。",
    "pipe.sAt": "识别与决策",
    "pipe.sAp": "场景分析 → 需求推断 → 方案生成 → 写入记忆。",
    "pipe.sBt": "交付与反馈",
    "pipe.sBp": "按数字 / 设备 / 硬件通道执行，并收集用户确认。",
    "pipe.sCt": "记忆整合",
    "pipe.sCp": "反馈完成后触发，默认后台运行，不阻塞响应。",
    "mem.title": "七层记忆 + HintMemory",
    "mem.p":
      "PersonMemory 从当前时刻、同环境历史、今日记录，压缩到近期任务、长期事件与用户画像，并按时间、活动、人物、地点归档。HintMemory 保存用户自定义的「触发条件 → 需求」规则，在需求分析时注入。",
    "mem.l1": "L1–3 当前 / 同环境 / 今日",
    "mem.l2": "L4–6 任务压缩 / 长期事件 / 画像",
    "mem.l3": "L7 分类归档",
    "mem.l4": "Hint 用户规则",
    "sc.eyebrow": "验证场景",
    "sc.title": "从出行到健康，覆盖日常主动帮助",
    "sc.1": "社交提示",
    "sc.2": "出国旅行",
    "sc.3": "遗忘物品",
    "sc.4": "急救",
    "sc.5": "健康防护",
    "sc.6": "酒店前台",
    "sc.7": "拍照辅助",
    "sc.8": "诈骗识别",
    "sc.9": "安全警报",
    "sc.10": "对话辅助",
    "sc.11": "情感支持",
    "sc.12": "购物选择",
    "stack.eyebrow": "技术栈",
    "stack.title": "可本地运行的感知与执行栈",
    "stack.th1": "服务",
    "stack.th2": "端口",
    "stack.th3": "作用",
    "stack.r1": "Qwen3-VL 视觉 + Whisper 语音",
    "stack.r2": "录屏与 GUI-Owl 屏幕分析",
    "stack.r3": "接收 Android 录屏",
    "footer.p": "主动式上下文感知助手。产品信息来自当前工程实现。",
    "footer.addrLabel": "地址",
    "footer.phoneLabel": "电话",
    "footer.emailLabel": "邮箱",
  },
  en: {
    "nav.company": "Company",
    "nav.products": "Products",
    "nav.pipeline": "How it works",
    "nav.scenarios": "Scenarios",
    "nav.contact": "Contact",
    "nav.cta": "See products",
    "hero.eyebrow": "Proactive AI company",
    "hero.title": "Sense the moment. Act first.",
    "hero.lede":
      "PROACTIVE INTELLIGENCE LIMITED builds a context-aware assistant that fuses first-person video, screen activity, and wearable signals, infers what the user needs, and delivers help through digital info, device control, and hardware commands.",
    "hero.cta1": "View products",
    "hero.cta2": "How it works",
    "stats.s1k": "Input modalities",
    "stats.s2k": "Output channels",
    "stats.s3k": "Memory layers",
    "stats.s4k": "Eval scenarios",
    "company.eyebrow": "About",
    "company.title": "PROACTIVE INTELLIGENCE LIMITED",
    "company.lede":
      "We believe an assistant should not wait to be asked. It should see what the user is doing, remember prior context, and offer the next step at the right time.",
    "company.c1t": "Proactive, not reactive",
    "company.c1p":
      "The agent continuously reads scenes and screen activity to infer the top needs of the moment, instead of waiting for an explicit prompt.",
    "company.c2t": "Multimodal fusion",
    "company.c2p":
      "Smart-glasses or phone video, PC and Android screen recordings, and wearable telemetry such as heart rate or posture can run alone or together in one pipeline.",
    "company.c3t": "Remember, then follow up",
    "company.c3p":
      "A seven-layer memory keeps scene history, preferences, and prior advice. The next session checks whether the user already acted on the last plan.",
    "products.eyebrow": "Products",
    "products.title": "One sensing stack, two product lines",
    "products.p1tag": "Core product",
    "products.p1p":
      "A context-aware assistant. It reads video, screen, and sensors, recognizes scenes and needs, then routes solutions to AR/notifications, phone and PC actions, or smart-home and robot commands.",
    "products.p1l1": "Scene recognition → need analysis → solutions → multi-channel delivery",
    "products.p1l2": "Inputs: camera / screen_record / gui_agent / wearable / multimodal",
    "products.p1l3": "Vision with Qwen3-VL, speech with Whisper",
    "products.p2tag": "Integrated capability",
    "products.p2p":
      "All-day screen recording on PC and Android (app name XOOGUIAGT). GUI-Owl recognizes apps, pages, UI elements, and actions, then Proactive Agent infers needs.",
    "products.p2l1": "PC web shell recording + Android upload",
    "products.p2l2": "Local GUI-Owl or cloud vision analysis",
    "products.p2l3": "Results persist to memory for search and user profiling",
    "io.inEyebrow": "Inputs",
    "io.inTitle": "See what the user is going through",
    "io.in1t": "User-shot video",
    "io.in1p": "Smart glasses, phone cameras, or first-person POV footage.",
    "io.in2t": "Phone / PC screen recordings",
    "io.in2p": "Understand apps, page transitions, and workflows.",
    "io.in3t": "GUI Agent",
    "io.in3p": "GUI-Owl structures app name, page, elements, and actions.",
    "io.in4t": "Wearable data",
    "io.in4p": "Heart rate, steps, location, posture — alone or fused with video.",
    "io.outEyebrow": "Outputs",
    "io.outTitle": "Deliver help where it can actually run",
    "io.out1t": "Digital info & app services",
    "io.out1p": "AR overlays, notifications, tips, and in-app guidance.",
    "io.out2t": "Phone / PC operations",
    "io.out2p": "Open apps, tap UI, type text, send messages, browse.",
    "io.out3t": "Hardware & robots",
    "io.out3p": "Lights, thermostats, and robot motion commands.",
    "pipe.eyebrow": "How it works",
    "pipe.title": "A four-phase pipeline",
    "pipe.lede":
      "Frame extraction, audio extraction, and transcription run once and are shared between the pre-check and the main analysis.",
    "pipe.s0t": "Pre-check",
    "pipe.s0p": "Compare against last session’s advice and see if the user followed it.",
    "pipe.sAt": "Recognize & decide",
    "pipe.sAp": "Scene analysis → need inference → solution generation → memory write.",
    "pipe.sBt": "Deliver & feedback",
    "pipe.sBp": "Execute on digital / device / hardware channels, then collect confirmation.",
    "pipe.sCt": "Consolidate",
    "pipe.sCp": "Triggered after feedback; runs in the background by default.",
    "mem.title": "Seven-layer memory + HintMemory",
    "mem.p":
      "PersonMemory moves from the current moment, same-environment history, and today’s record into compressed recent tasks, long-term events, and a user profile, then archives by time, activity, person, and place. HintMemory stores user-authored trigger→need rules and injects them at analysis time.",
    "mem.l1": "L1–3 now / same place / today",
    "mem.l2": "L4–6 tasks / long-term / profile",
    "mem.l3": "L7 classified archive",
    "mem.l4": "Hint user rules",
    "sc.eyebrow": "Scenarios",
    "sc.title": "Everyday proactive help, from travel to health",
    "sc.1": "Social hints",
    "sc.2": "Travel abroad",
    "sc.3": "Forgotten items",
    "sc.4": "First aid",
    "sc.5": "Health safeguard",
    "sc.6": "Hotel reception",
    "sc.7": "Photo taking",
    "sc.8": "Fraud detection",
    "sc.9": "Security alert",
    "sc.10": "Conversation assistance",
    "sc.11": "Emotional support",
    "sc.12": "Shopping selection",
    "stack.eyebrow": "Stack",
    "stack.title": "A sensing and actuation stack you can run locally",
    "stack.th1": "Service",
    "stack.th2": "Port",
    "stack.th3": "Role",
    "stack.r1": "Qwen3-VL vision + Whisper speech",
    "stack.r2": "Screen recording and GUI-Owl analysis",
    "stack.r3": "Receive Android recordings",
    "footer.p": "A proactive, context-aware assistant. Product details reflect the current implementation.",
    "footer.addrLabel": "Address",
    "footer.phoneLabel": "Phone",
    "footer.emailLabel": "Email",
  },
};

function applyLang(lang) {
  const dict = COPY[lang] || COPY.en;
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (dict[key]) el.textContent = dict[key];
  });
  document.querySelectorAll(".lang-switch button").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.lang === lang);
  });
  const title = "PROACTIVE INTELLIGENCE LIMITED";
  const desc =
    lang === "zh"
      ? "PROACTIVE INTELLIGENCE LIMITED — 构建感知多路输入、执行多通道输出的主动式 AI 助手。"
      : "PROACTIVE INTELLIGENCE LIMITED builds a context-aware assistant that senses multimodal input and acts through multiple output channels.";
  document.title = title;
  const meta = document.querySelector('meta[name="description"]');
  if (meta) meta.setAttribute("content", desc);
  localStorage.setItem("pi-lang", lang);
}

document.querySelectorAll(".lang-switch button").forEach((btn) => {
  btn.addEventListener("click", () => applyLang(btn.dataset.lang));
});

applyLang(localStorage.getItem("pi-lang") === "zh" ? "zh" : "en");

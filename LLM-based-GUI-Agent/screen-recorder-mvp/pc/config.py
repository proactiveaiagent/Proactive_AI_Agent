# -*- coding: utf-8 -*-
"""Configuration for GUI Agent video analyzer."""
import os

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

DASHSCOPE_BASE_URL = os.environ.get(
    "GUI_AGENT_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_MODEL = os.environ.get("GUI_AGENT_MODEL", "qwen-vl-max")
GUI_AGENT_BACKEND = os.environ.get("GUI_AGENT_BACKEND", "local").lower()

# Local GUI-Owl deployment (default backend)
LOCAL_GUI_OWL_MODEL_ID = os.environ.get(
    "LOCAL_GUI_OWL_MODEL_ID",
    "mPLUG/GUI-Owl-1.5-8B-Instruct",
)
LOCAL_MODEL_CACHE_DIRNAME = os.environ.get(
    "LOCAL_MODEL_CACHE_DIRNAME",
    "models",
)
LOCAL_MODEL_GPU_MEMORY = os.environ.get("LOCAL_MODEL_GPU_MEMORY", "16GiB")
LOCAL_MODEL_CPU_MEMORY = os.environ.get("LOCAL_MODEL_CPU_MEMORY", "24GiB")
LOCAL_MODEL_MAX_NEW_TOKENS = int(os.environ.get("LOCAL_MODEL_MAX_NEW_TOKENS", "256"))
LOCAL_IMAGE_MAX_EDGE = int(os.environ.get("LOCAL_IMAGE_MAX_EDGE", "1280"))

FRAME_SAMPLE_INTERVAL_SEC = 2.0
SSIM_THRESHOLD = 0.95

ANALYSIS_OUTPUT_DIR = "analysis_output"

ANALYSIS_PROMPT = """Analyze this GUI screenshot and return a JSON object with the following fields:
{
  "app_name": "the name of the application or website visible on screen",
  "page_name": "the current page, tab, or view name",
  "elements": [
    {"type": "button|icon|text|input|image|link|menu|tab|module", "label": "element text or description", "description": "brief context"}
  ],
  "visible_text": ["list of all readable text blocks on screen"],
  "user_action": "inferred user action or interaction at this moment",
  "description": "one-sentence summary of what is shown on screen"
}

Rules:
- Return ONLY the JSON object, no markdown fences, no extra text.
- "elements" should list the most important UI elements (up to 20).
- "visible_text" should capture all meaningful text content.
- "user_action" should describe what the user is likely doing.
- If you cannot determine a field, use an empty string or empty list.
"""

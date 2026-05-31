import { useCallback, useEffect, useMemo, useState } from "react";

type TabId = "recording" | "earnings" | "data" | "settings" | "search";
type DataDimension = "time" | "topic" | "event" | "value";

type Status = {
  recording: boolean;
  receiver_running: boolean;
  receiver_upload_url: string;
  analysis_running: boolean;
  deploy_running: boolean;
  selected_video: string | null;
  analysis_result_dir: string | null;
  progress: { pct: number; message: string };
  backend_default: string;
  default_model_id: string;
};

const API_BASE = "";

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const j = await r.json();
      detail = j.detail ?? JSON.stringify(j);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

function navIconSrc(base: string, active: boolean): string {
  return `/app-assets/${base}${active ? "_blue" : ""}.svg`;
}

function NavIcon({ name, active }: { name: string; active: boolean }) {
  return (
    <img src={navIconSrc(name, active)} alt="" width={24} height={24} decoding="async" />
  );
}

function onlineLine(h: string, m: string): string {
  return `${h} ${m} · Online Recording`;
}

const DATA_BY_DIM: Record<
  DataDimension,
  { heading: string; rows: { title: string; subtitle: string }[] }[]
> = {
  time: [
    {
      heading: "This Week",
      rows: [
        { title: "19 March, 2026 Thur", subtitle: onlineLine("9h 35", "min") },
        { title: "18 March, 2026 Wed", subtitle: onlineLine("8h 20", "min") },
        { title: "17 March, 2026 Tue", subtitle: onlineLine("7h 55", "min") },
        { title: "16 March, 2026 Mon", subtitle: onlineLine("9h 02", "min") },
      ],
    },
    {
      heading: "This Month",
      rows: [
        { title: "15 March, 2026 Sun", subtitle: onlineLine("6h 40", "min") },
        { title: "14 March, 2026 Sat", subtitle: onlineLine("5h 15", "min") },
      ],
    },
  ],
  topic: [
    {
      heading: "Work",
      rows: [
        { title: "Deep work block · Mon", subtitle: onlineLine("4h 10", "min") },
        { title: "Email & admin · Tue", subtitle: onlineLine("2h 05", "min") },
      ],
    },
    {
      heading: "Social",
      rows: [
        { title: "Calls & messages · Wed", subtitle: onlineLine("1h 30", "min") },
      ],
    },
    {
      heading: "Learning",
      rows: [
        { title: "Course playback · Thu", subtitle: onlineLine("3h 00", "min") },
      ],
    },
  ],
  event: [
    {
      heading: "Meetings",
      rows: [
        { title: "Stand-up · 10:00", subtitle: onlineLine("0h 45", "min") },
        { title: "Project review · 15:00", subtitle: onlineLine("1h 20", "min") },
      ],
    },
    {
      heading: "Travel & commute",
      rows: [
        { title: "Commute · morning", subtitle: onlineLine("0h 35", "min") },
        { title: "Commute · evening", subtitle: onlineLine("0h 42", "min") },
      ],
    },
  ],
  value: [
    {
      heading: "Productivity",
      rows: [
        { title: "Focus score · week", subtitle: "High · Online Recording" },
        { title: "Tasks completed", subtitle: "12 items · Online Recording" },
      ],
    },
    {
      heading: "Wellness",
      rows: [
        { title: "Breaks & movement", subtitle: "On track · Online Recording" },
        { title: "Evening wind-down", subtitle: onlineLine("1h 10", "min") },
      ],
    },
  ],
};

export default function App() {
  const [tab, setTab] = useState<TabId>("recording");
  const [dataDim, setDataDim] = useState<DataDimension>("time");
  const [status, setStatus] = useState<Status | null>(null);
  const [logSince, setLogSince] = useState(0);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [tip, setTip] = useState<string | null>(null);

  const [surveyOpen, setSurveyOpen] = useState(true);
  const [recOpen, setRecOpen] = useState(false);
  const [diaryOpen, setDiaryOpen] = useState(false);

  const [backend, setBackend] = useState<"local" | "api">("local");
  const [modelId, setModelId] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [recordings, setRecordings] = useState<
    { name: string; path: string; size: number }[]
  >([]);
  const [memQuery, setMemQuery] = useState("");
  const [memOut, setMemOut] = useState("");
  const isBusy = Boolean(
    status?.recording || status?.analysis_running || status?.deploy_running
  );

  const refreshStatus = useCallback(async () => {
    try {
      const s = await api<{ ok: boolean } & Status>("/api/status");
      if (s.ok) {
        setStatus({
          recording: s.recording,
          receiver_running: s.receiver_running,
          receiver_upload_url: s.receiver_upload_url,
          analysis_running: s.analysis_running,
          deploy_running: s.deploy_running,
          selected_video: s.selected_video,
          analysis_result_dir: s.analysis_result_dir,
          progress: s.progress,
          backend_default: s.backend_default,
          default_model_id: s.default_model_id,
        });
        setModelId((prev) => prev || s.default_model_id || "");
      }
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  const refreshLogs = useCallback(async () => {
    try {
      const j = await api<{ logs: { i: number; msg: string }[] }>(
        `/api/logs?since=${logSince}`
      );
      if (j.logs.length) {
        setLogSince(j.logs[j.logs.length - 1].i + 1);
        setLogLines((prev) => [...prev, ...j.logs.map((l) => l.msg)].slice(-80));
      }
    } catch {
      /* ignore */
    }
  }, [logSince]);

  const loadRecordings = useCallback(async () => {
    try {
      const j = await api<{ items: typeof recordings }>("/api/recordings");
      setRecordings(j.items || []);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  useEffect(() => {
    if (!isBusy) return;
    const id = setInterval(() => {
      if (document.visibilityState === "visible") {
        refreshStatus();
      }
    }, 1200);
    return () => clearInterval(id);
  }, [isBusy, refreshStatus]);

  useEffect(() => {
    if (!isBusy) return;
    const id = setInterval(() => {
      if (document.visibilityState === "visible") {
        refreshLogs();
      }
    }, 1500);
    return () => clearInterval(id);
  }, [isBusy, refreshLogs]);

  useEffect(() => {
    if (tab === "data") loadRecordings();
  }, [tab, loadRecordings]);

  const pct = status?.progress.pct ?? 0;
  const progressMsg = status?.progress.message ?? "";

  const nav = useMemo(
    () =>
      [
        { id: "recording" as const, label: "Recording", icon: "recording" },
        { id: "earnings" as const, label: "Earnings", icon: "gift" },
        { id: "data" as const, label: "Data", icon: "data" },
        { id: "settings" as const, label: "Settings", icon: "setting" },
        { id: "search" as const, label: "Search", icon: "search" },
      ] as const,
    []
  );

  async function post(path: string, body?: object) {
    setErr(null);
    try {
      await api(path, {
        method: "POST",
        body: body ? JSON.stringify(body) : "{}",
      });
      await refreshStatus();
    } catch (e) {
      setErr(String(e));
    }
  }

  const headerStatus = isBusy ? "Syncing…" : "Idle";

  return (
    <div className="phone-frame">
      <div className="app-shell">
        <header className="app-header">
          <span className="brand">XOOGUIAGT</span>
          <span className="ver">{headerStatus}</span>
        </header>

        <main className="app-main">
          {tip && (
            <div className="card plain-white" style={{ marginBottom: 12 }}>
              <p className="muted" style={{ margin: 0 }}>
                {tip}{" "}
                <button
                  type="button"
                  className="btn secondary"
                  style={{ marginLeft: 8, padding: "4px 10px", fontSize: 12 }}
                  onClick={() => setTip(null)}
                >
                  Dismiss
                </button>
              </p>
            </div>
          )}

          {err && (
            <div className="card plain-white" style={{ borderColor: "var(--ds-danger)" }}>
              <p className="muted" style={{ color: "var(--ds-danger)", margin: 0 }}>
                {err}
              </p>
            </div>
          )}

          {tab === "recording" && (
            <>
              <p className="intro">
                To get best use of your AI agent, please keep recording whole day and every
                day
              </p>

              <button
                type="button"
                className="btn-pill-primary"
                onClick={() => post("/api/record/toggle")}
              >
                {status?.recording ? "Stop PC Recording" : "Start PC Recording"}
              </button>
              <p className="status-small">
                {status?.recording
                  ? "PC: recording — file is being saved to the recordings folder."
                  : "PC: idle — captures this desktop."}
              </p>

              <label
                className="muted"
                style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 14 }}
              >
                <input
                  type="checkbox"
                  onChange={() => setTip("Auto recording toggle is UI-only for now.")}
                />
                Auto recording
              </label>

              <button type="button" className="btn-pill-outline" style={{ marginTop: 12 }} disabled>
                Start Phone Recording
              </button>
              <p className="status-small">Use the mobile app to capture your phone screen.</p>

              <div className="tip-banner">
                <span>
                  Get more benefits and earnings by equipping a portable micro camera to
                  enable your AI agent to get your real-world data
                </span>
              </div>

              <button
                type="button"
                className="pill-link"
                onClick={() => setTip("Offline recording flow is available in the mobile app.")}
              >
                Enable Offline Recording &gt;
              </button>

              <div className="banner-card">
                <p className="banner-title">
                  13 questionnaires has been filled for your review
                </p>
                <div className="banner-actions">
                  <button type="button">✖ Dismiss</button>
                  <button type="button" className="bold">
                    View &gt;
                  </button>
                </div>
              </div>

              <div className="banner-card green">
                <p className="banner-title">
                  Your daily summary and reflection has been generated
                </p>
                <div className="banner-actions">
                  <button type="button">✖ Dismiss</button>
                  <button type="button" className="bold">
                    View &gt;
                  </button>
                </div>
              </div>

              <div className="card" style={{ marginTop: 16 }}>
                <h2 className="section-title">Receive uploads from phone</h2>
                <p className="muted">
                  On the same Wi‑Fi, open the app tab Recording → Send recording to PC, and
                  use this machine&apos;s LAN IP with port <strong>8765</strong>.
                </p>
                <div className="row">
                  <button
                    type="button"
                    className="btn"
                    disabled={status?.receiver_running}
                    onClick={() => post("/api/receiver/start")}
                  >
                    Start upload server
                  </button>
                  <button
                    type="button"
                    className="btn secondary"
                    disabled={!status?.receiver_running}
                    onClick={() => post("/api/receiver/stop")}
                  >
                    Stop
                  </button>
                  <button
                    type="button"
                    className="btn secondary"
                    onClick={() => post("/api/system/open-folder", { kind: "recordings" })}
                  >
                    Open recordings folder
                  </button>
                </div>
                <p className="muted">
                  Server: {status?.receiver_running ? "Running" : "Stopped"}
                </p>
                {status?.receiver_upload_url && (
                  <p className="muted" style={{ wordBreak: "break-all" }}>
                    Upload URL: {status.receiver_upload_url}
                  </p>
                )}
              </div>
            </>
          )}

          {tab === "earnings" && (
            <>
              <h2
                className="section-heading-app"
                style={{ marginTop: 4, marginBottom: 14, fontSize: 28 }}
              >
                Earnings
              </h2>

              <div className="acc-card">
                <button
                  type="button"
                  className="acc-header"
                  onClick={() => setSurveyOpen((v) => !v)}
                >
                  <span className="title">Auto-filled Survey</span>
                  <span className="chev">{surveyOpen ? "▲" : "▼"}</span>
                </button>
                {surveyOpen && (
                  <div className="acc-body">
                    <p
                      className="muted"
                      dangerouslySetInnerHTML={{
                        __html:
                          "Your agent has auto found and filled <b>13 Questionnaires</b> for you to earn",
                      }}
                    />
                    <p className="amount-big">¥ 216</p>
                    <p className="amount-sub">today after you review and submit them</p>
                    <button type="button" className="btn-review">
                      Review and Submit &gt;
                    </button>
                    <div className="stat-row">
                      <div className="stat-col">
                        <div className="lbl">Yesterday earnings</div>
                        <div className="val">¥ 138</div>
                      </div>
                      <div className="stat-col">
                        <div className="lbl">Total earnings</div>
                        <div className="val">¥ 2359</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="acc-card">
                <button
                  type="button"
                  className="acc-header"
                  onClick={() => setRecOpen((v) => !v)}
                >
                  <span className="title">Recommendation</span>
                  <span className="chev">{recOpen ? "▲" : "▼"}</span>
                </button>
                {recOpen && (
                  <div className="acc-body" style={{ paddingTop: 4 }}>
                    {[
                      ["19 March, 2026 Thur", onlineLine("9h 35", "min")],
                      ["18 March, 2026 Wed", onlineLine("8h 20", "min")],
                      ["17 March, 2026 Tue", onlineLine("7h 55", "min")],
                      ["16 March, 2026 Mon", onlineLine("9h 02", "min")],
                    ].map(([t, s]) => (
                      <button
                        type="button"
                        key={t}
                        className="data-row"
                        onClick={() => setTip(t)}
                      >
                        <div className="body">
                          <div className="t">{t}</div>
                          <div className="s">{s}</div>
                        </div>
                        <span className="chev">›</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="acc-card">
                <button
                  type="button"
                  className="acc-header"
                  onClick={() => setDiaryOpen((v) => !v)}
                >
                  <span className="title">AI-powered Diary</span>
                  <span className="chev">{diaryOpen ? "▲" : "▼"}</span>
                </button>
                {diaryOpen && (
                  <div className="acc-body" style={{ paddingTop: 4 }}>
                    {[
                      ["15 March, 2026 Sun", "Daily reflection · 3 highlights"],
                      ["14 March, 2026 Sat", "Voice note · evening summary"],
                      ["13 March, 2026 Fri", "Auto summary · workday"],
                      ["12 March, 2026 Thu", "Mood tracker · check-in"],
                    ].map(([t, s]) => (
                      <button
                        type="button"
                        key={t}
                        className="data-row"
                        onClick={() => setTip(t)}
                      >
                        <div className="body">
                          <div className="t">{t}</div>
                          <div className="s">{s}</div>
                        </div>
                        <span className="chev">›</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div
                className="more-apps-row"
                role="button"
                tabIndex={0}
                onClick={() => setTip("More applications (demo)")}
                onKeyDown={(e) => {
                  if (e.key === "Enter") setTip("More applications (demo)");
                }}
              >
                <span>More Useful Applications</span>
                <span className="chev">›</span>
              </div>
            </>
          )}

          {tab === "data" && (
            <>
              <div className="dim-row">
                {(
                  [
                    ["time", "Time"],
                    ["topic", "Topic"],
                    ["event", "Event"],
                    ["value", "Value"],
                  ] as const
                ).map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    className={`dim-chip ${dataDim === id ? "active" : ""}`}
                    onClick={() => setDataDim(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {DATA_BY_DIM[dataDim].map((sec) => (
                <div key={sec.heading}>
                  <div className="data-section-title">{sec.heading}</div>
                  {sec.rows.map((row) => (
                    <button
                      type="button"
                      key={row.title}
                      className="data-row"
                      onClick={() => setTip(row.title)}
                    >
                      <div className="body">
                        <div className="t">{row.title}</div>
                        <div className="s">{row.subtitle}</div>
                      </div>
                      <span className="chev">›</span>
                    </button>
                  ))}
                </div>
              ))}

              <div className="card" style={{ marginTop: 16 }}>
                <h2 className="section-title">Video for analysis (this PC)</h2>
                <p className="muted">
                  Choose a file from the recordings folder for the analysis pipeline.
                </p>
                <div className="row">
                  <button type="button" className="btn secondary" onClick={loadRecordings}>
                    Refresh list
                  </button>
                </div>
                <label>Current selection</label>
                <select
                  value={status?.selected_video || ""}
                  onChange={async (e) => {
                    const path = e.target.value;
                    await post("/api/video/select", { path });
                    await loadRecordings();
                  }}
                >
                  <option value="">None</option>
                  {recordings.map((r) => (
                    <option key={r.path} value={r.path}>
                      {r.name} ({Math.round(r.size / 1024)} KB)
                    </option>
                  ))}
                </select>
              </div>

              <div className="card">
                <h2 className="section-title">Memory search</h2>
                <p className="muted">After analysis, frames are indexed in local SQLite.</p>
                <MemoryStatsInline />
                <label style={{ marginTop: 12 }}>Keyword / natural language</label>
                <input
                  type="text"
                  value={memQuery}
                  onChange={(e) => setMemQuery(e.target.value)}
                  placeholder="e.g. browser, WeChat"
                />
                <div className="row" style={{ marginTop: 12 }}>
                  <button
                    type="button"
                    className="btn"
                    onClick={async () => {
                      setErr(null);
                      try {
                        const j = await api<{
                          ok: boolean;
                          results: Record<string, unknown>[];
                        }>("/api/memory/search", {
                          method: "POST",
                          body: JSON.stringify({ query: memQuery, limit: 30 }),
                        });
                        if (!j.results?.length) {
                          setMemOut("No matches");
                          return;
                        }
                        const lines = j.results.map((r) => {
                          const ts = Number(r.timestamp_sec ?? 0);
                          const app = String(r.app_name ?? "");
                          const page = String(r.page_name ?? "");
                          const desc = String(r.description ?? "").slice(0, 200);
                          return `[${ts}s] ${app} / ${page}\n  ${desc}\n`;
                        });
                        setMemOut(lines.join("\n"));
                      } catch (e) {
                        setErr(String(e));
                      }
                    }}
                  >
                    Search
                  </button>
                  <button
                    type="button"
                    className="btn secondary"
                    onClick={async () => {
                      setErr(null);
                      try {
                        const j = await api<{
                          ok: boolean;
                          profile: Record<string, unknown>;
                        }>("/api/memory/profile");
                        const p = j.profile || {};
                        const top =
                          (p.top_apps as { app: string; count: number }[]) || [];
                        const lines = [
                          `Total events: ${p.total_events ?? 0}`,
                          "Top apps:",
                          ...top.slice(0, 10).map((x) => `  - ${x.app}: ${x.count}`),
                        ];
                        setMemOut(lines.join("\n"));
                      } catch (e) {
                        setErr(String(e));
                      }
                    }}
                  >
                    Profile
                  </button>
                </div>
                <label style={{ marginTop: 12 }}>Output</label>
                <textarea value={memOut} readOnly />
              </div>
            </>
          )}

          {tab === "settings" && (
            <>
              <div className="settings-section-label">Data Collection</div>
              <div className="settings-card">
                <div className="settings-row">
                  <span className="settings-check" aria-hidden>
                    ✓
                  </span>
                  <div className="grow">
                    <div className="lbl">Phone:</div>
                    <div className="val">Huawei Mate 20</div>
                  </div>
                  <button
                    type="button"
                    className="icon-btn settings-edit"
                    aria-label="Edit phone"
                    onClick={() => setTip("Edit phone label (demo)")}
                  >
                    ✎
                  </button>
                </div>
                <div className="settings-row">
                  <span className="settings-check" aria-hidden>
                    ✓
                  </span>
                  <div className="grow">
                    <div className="lbl">PC:</div>
                    <div className="val">Huawei Matebook</div>
                  </div>
                  <button
                    type="button"
                    className="icon-btn settings-edit"
                    aria-label="Edit PC"
                    onClick={() => setTip("Edit PC label (demo)")}
                  >
                    ✎
                  </button>
                </div>
                <button
                  type="button"
                  className="settings-row clickable"
                  onClick={() =>
                    setTip("Offline recording: use the mobile app Enable Offline flow.")
                  }
                >
                  <div className="status-dot" />
                  <div className="grow">
                    <div className="lbl">Offline Recording</div>
                  </div>
                  <span className="chev">›</span>
                </button>
              </div>

              <div className="settings-section-label">Base LLM</div>
              <div className="settings-card">
                <div className="settings-row">
                  <div className="grow">
                    <div className="lbl">GUI Agent:</div>
                    <div className="val">Mobile-Agent-v3.5</div>
                  </div>
                  <button
                    type="button"
                    className="icon-btn settings-edit"
                    onClick={() => setTip("GUI Agent: Mobile-Agent-v3.5 (demo)")}
                  >
                    ✎
                  </button>
                </div>
                <div className="settings-row">
                  <div className="grow">
                    <div className="lbl">Multi-modal LLM:</div>
                    <div className="val">Qwen VL 8B</div>
                  </div>
                  <button
                    type="button"
                    className="icon-btn settings-edit"
                    onClick={() => setTip("Multi-modal LLM (demo)")}
                  >
                    ✎
                  </button>
                </div>
              </div>

              <div className="settings-section-label">Storage</div>
              <div className="settings-card">
                <div className="settings-row">
                  <div className="grow">
                    <div className="lbl">Data:</div>
                    <div className="val">
                      .\recordings and analysis output (this PC) · 6.7 GB (demo)
                    </div>
                  </div>
                  <button
                    type="button"
                    className="icon-btn settings-edit"
                    onClick={() => setTip("Storage path (demo)")}
                  >
                    ✎
                  </button>
                </div>
              </div>

              <div className="settings-section-label">Application</div>
              <div className="settings-card">
                <button
                  type="button"
                  className="settings-row clickable"
                  onClick={() => setTip("Questionnaire Auto-filling")}
                >
                  <span className="settings-check" aria-hidden>
                    ✓
                  </span>
                  <div className="grow">
                    <div className="lbl">Questionnaire Auto-filling</div>
                  </div>
                  <span className="chev">›</span>
                </button>
                <button
                  type="button"
                  className="settings-row clickable"
                  onClick={() => setTip("Personal recommendation")}
                >
                  <span className="settings-check" aria-hidden>
                    ✓
                  </span>
                  <div className="grow">
                    <div className="lbl">Personal recommendation</div>
                  </div>
                  <span className="chev">›</span>
                </button>
                <button
                  type="button"
                  className="settings-row clickable"
                  onClick={() => setTip("AI-powered diary")}
                >
                  <span className="settings-check" aria-hidden>
                    ✓
                  </span>
                  <div className="grow">
                    <div className="lbl">AI-powered diary</div>
                  </div>
                  <span className="chev">›</span>
                </button>
              </div>

              <div className="settings-section-label">Analysis backend (PC)</div>
              <div className="card">
                <p className="muted">
                  Local GUI-Owl / API analysis — same pipeline as the desktop controller.
                </p>
                <div className="row">
                  <label style={{ flex: "1 1 120px" }}>
                    Backend
                    <select
                      value={backend}
                      onChange={(e) => setBackend(e.target.value as "local" | "api")}
                    >
                      <option value="local">local (GUI-Owl)</option>
                      <option value="api">api (cloud-compatible)</option>
                    </select>
                  </label>
                </div>
                <label>Model ID</label>
                <input
                  type="text"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  placeholder="HuggingFace model id"
                />
                {backend === "api" && (
                  <>
                    <label style={{ marginTop: 12 }}>API key</label>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="DASHSCOPE / compatible"
                    />
                  </>
                )}
                <div className="row" style={{ marginTop: 12 }}>
                  <button
                    type="button"
                    className="btn secondary"
                    disabled={status?.deploy_running || status?.analysis_running}
                    onClick={() => post("/api/model/deploy", { model_id: modelId })}
                  >
                    Deploy local model
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={status?.analysis_running}
                    onClick={() =>
                      post("/api/analyze/start", {
                        backend,
                        api_key: apiKey,
                        model_id: modelId,
                      })
                    }
                  >
                    Start analysis
                  </button>
                  <button
                    type="button"
                    className="btn secondary"
                    onClick={() => post("/api/system/open-folder", { kind: "analysis" })}
                  >
                    Open analysis output
                  </button>
                </div>
                <div className="progress">
                  <div style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
                </div>
                <p className="muted">{progressMsg}</p>
                {status?.analysis_result_dir && (
                  <p className="muted" style={{ wordBreak: "break-all" }}>
                    Last result: {status.analysis_result_dir}
                  </p>
                )}
              </div>

              <details className="diagnostics">
                <summary>Diagnostics &amp; environment</summary>
                <p className="muted">
                  Web UI listens on{" "}
                  <code>SCREEN_RECORDER_WEB_HOST</code> /{" "}
                  <code>SCREEN_RECORDER_WEB_PORT</code>
                  ; phone upload port remains <strong>8765</strong>. Set{" "}
                  <code>SCREEN_RECORDER_NO_BROWSER=1</code> to skip opening a browser.
                </p>
                <p className="muted">Design tokens: aligned with Android design_tokens.xml</p>
                <div className="log-box">
                  {logLines.map((l, i) => (
                    <div key={i}>{l}</div>
                  ))}
                </div>
              </details>
            </>
          )}

          {tab === "search" && (
            <div className="card">
              <h2 className="section-title">Search</h2>
              <p className="muted">Coming soon — same entry as in the mobile app.</p>
            </div>
          )}
        </main>

        <nav className="bottom-nav">
          {nav.map((n) => (
            <button
              key={n.id}
              className={tab === n.id ? "active" : ""}
              type="button"
              onClick={() => setTab(n.id)}
            >
              <NavIcon name={n.icon} active={tab === n.id} />
              {n.label}
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
}

function MemoryStatsInline() {
  const [text, setText] = useState("Loading…");
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const j = await api<{
          ok: boolean;
          stats?: Record<string, number>;
          error?: string;
        }>("/api/memory/stats");
        if (cancelled) return;
        if (j.ok && j.stats) {
          setText(
            `raw: ${j.stats.raw_records ?? 0} | compressed: ${j.stats.compressed_records ?? 0}`
          );
        } else {
          setText(j.error || "Unavailable");
        }
      } catch (e) {
        if (!cancelled) setText(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);
  return <p className="muted">Memory stats: {text}</p>;
}

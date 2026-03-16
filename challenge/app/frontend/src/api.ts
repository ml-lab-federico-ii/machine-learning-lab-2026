// Typed wrappers around fetch for all backend API calls.

const BASE = "/api";

async function _extractError(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const json = JSON.parse(text);
    if (typeof json.detail === "string") return json.detail;
    if (Array.isArray(json.detail)) return json.detail.map((d: any) => d.msg).join(", ");
  } catch {}
  return text || `HTTP ${res.status}`;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { credentials: "include" });
  if (!res.ok) throw new Error(await _extractError(res));
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await _extractError(res));
  return res.json();
}

async function postForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    credentials: "include",
    body: form,
  });
  if (!res.ok) throw new Error(await _extractError(res));
  return res.json();
}

// ── EDA ────────────────────────────────────────────────────────────────────

export const api = {
  schema: () => get<Schema>("/data/schema"),
  preview: () => get<PreviewData>("/data/preview"),
  eda: (column: string) => get<EdaData>(`/data/eda/${column}`),
  correlations: () => get<CorrelationData>("/data/correlations"),

  train: (pipeline_config: PipelineConfig) =>
    post<TrainResult>("/train", { pipeline_config }),

  evaluateThreshold: (payload: ThresholdPayload) =>
    post<ThresholdResult>("/evaluate-threshold", payload),

  deadline: () => get<DeadlineInfo>("/deadline"),
  authMe: () => get<AuthMe>("/auth/me"),
  authLogout: () => get<{ ok: boolean }>("/auth/logout"),

  submit: (payload: SubmitPayload) => post<SubmitResult>("/submit", payload),

  // Instructor
  instructorLogin: (password: string) =>
    post<{ ok: boolean }>("/instructor/login", { password }),
  instructorUploadTest: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return postForm<{ ok: boolean; rows: number; columns: string[] }>(
      "/instructor/upload-test",
      form
    );
  },
  instructorSubmissions: () => get<Submission[]>("/instructor/submissions"),
  instructorScore: () => post<LeaderboardEntry[]>("/instructor/score", {}),
  instructorExport: () =>
    fetch(`${BASE}/instructor/export`, { credentials: "include" }),
};

// ── Types ──────────────────────────────────────────────────────────────────

export interface Schema {
  id_column: string;
  target_column: string;
  target_positive_class: number | string;
}

export interface ColumnMeta {
  name: string;
  dtype: string;
  missing: number;
  missing_pct: number;
}

export interface PreviewData {
  shape: { rows: number; cols: number };
  columns: ColumnMeta[];
  class_balance: Record<string, number>;
  head: Record<string, unknown>[];
}

export interface EdaData {
  type: "categorical" | "numeric";
  column: string;
  data: { label?: string; count?: number; bin_start?: number; bin_end?: number }[];
}

export interface CorrelationData {
  columns: string[];
  matrix: (number | null)[][];
}

export interface PipelineConfig {
  split: { train: number; val: number; test: number };
  selected_features: string[];
  preprocessing: {
    imputation_numeric: string;
    imputation_categorical: string;
    scaling: string;
    encoding: string;
    imbalance_handling: string;
  };
  model: {
    name: string;
    params: Record<string, number | string>;
  };
}

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface RocCurve {
  fpr: number[];
  tpr: number[];
  thresholds: number[];
}

export interface SplitMetrics {
  roc_auc: number;
  roc_curve: RocCurve;
  confusion_matrix: ConfusionMatrix;
  precision: number;
  recall: number;
  f1: number;
  threshold: number;
}

export interface TrainResult {
  validation: SplitMetrics;
  test: SplitMetrics;
  split_sizes: { train: number; val: number; test: number };
  val_y_true: number[];
  val_y_proba: number[];
  model_b64: string;
}

export interface ThresholdPayload {
  threshold: number;
  y_true: number[];
  y_proba: number[];
}

export interface ThresholdResult {
  confusion_matrix: ConfusionMatrix;
  precision: number;
  recall: number;
  f1: number;
  threshold: number;
}

export interface DeadlineInfo {
  deadline: string | null;
  has_passed: boolean;
  seconds_remaining: number | null;
}

export interface AuthMe {
  authenticated: boolean;
  user?: { login: string; avatar_url: string; name: string };
}

export interface SubmitPayload {
  pipeline_config: PipelineConfig;
  model_b64: string;
}

export interface SubmitResult {
  ok: boolean;
  pr_url: string;
  submitted_at: string;
}

export interface Submission {
  number: number;
  title: string;
  user: string;
  updated_at: string;
  pr_url: string;
}

export interface LeaderboardEntry {
  username: string;
  pr_url: string;
  updated_at: string;
  roc_auc: number | null;
  precision?: number;
  recall?: number;
  f1?: number;
  roc_curve?: RocCurve;
  confusion_matrix?: ConfusionMatrix;
  error: string | null;
}

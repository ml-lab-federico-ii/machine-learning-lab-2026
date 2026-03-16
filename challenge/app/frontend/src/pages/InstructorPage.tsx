import { useState } from "react";
import { api, LeaderboardEntry } from "../api";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ReferenceLine, ResponsiveContainer } from "recharts";

export default function InstructorPage() {
  const [authenticated, setAuthenticated] = useState(false);
  const [password, setPassword] = useState("");
  const [loginError, setLoginError] = useState<string | null>(null);

  const [testUploaded, setTestUploaded] = useState(false);
  const [testStats, setTestStats] = useState<{ rows: number; columns: string[] } | null>(null);

  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState<string | null>(null);

  const [compareA, setCompareA] = useState<string | null>(null);
  const [compareB, setCompareB] = useState<string | null>(null);

  const handleLogin = async () => {
    try {
      await api.instructorLogin(password);
      setAuthenticated(true);
      setLoginError(null);
    } catch {
      setLoginError("Invalid password");
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const res = await api.instructorUploadTest(file);
      setTestUploaded(true);
      setTestStats({ rows: res.rows, columns: res.columns });
    } catch (err: any) {
      alert("Upload failed: " + err.message);
    }
  };

  const handleScore = async () => {
    setScoring(true);
    setScoreError(null);
    try {
      const results = await api.instructorScore();
      setLeaderboard(results);
    } catch (e: any) {
      setScoreError(e.message);
    } finally {
      setScoring(false);
    }
  };

  const handleExport = async () => {
    const res = await api.instructorExport();
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "leaderboard.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!authenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="bg-white rounded-2xl shadow p-8 w-full max-w-sm space-y-4">
          <h1 className="text-xl font-bold text-center text-gray-800">Instructor Dashboard</h1>
          <input
            type="password"
            placeholder="Instructor password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleLogin()}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          {loginError && <p className="text-red-500 text-sm">{loginError}</p>}
          <button
            onClick={handleLogin}
            className="w-full py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700"
          >
            Login
          </button>
        </div>
      </div>
    );
  }

  const entryA = leaderboard.find((e) => e.username === compareA);
  const entryB = leaderboard.find((e) => e.username === compareB);

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 py-4 px-6">
        <h1 className="text-lg font-bold text-gray-900">Instructor Dashboard</h1>
        <p className="text-xs text-gray-400">ML Challenge — Federico II</p>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8 space-y-8">
        {/* Upload test set */}
        <section className="bg-white rounded-xl shadow p-6 space-y-4">
          <h2 className="text-base font-semibold">1. Upload Hidden Test Set</h2>
          <p className="text-sm text-gray-500">
            Upload <code>test_hidden.csv</code>. It will be held in memory and never written to disk.
          </p>
          <input type="file" accept=".csv" onChange={handleUpload} className="text-sm" />
          {testStats && (
            <div className="text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg p-3">
              ✅ Loaded {testStats.rows} rows · {testStats.columns.length} columns
            </div>
          )}
        </section>

        {/* Score all */}
        <section className="bg-white rounded-xl shadow p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold">2. Score All Submissions</h2>
            <div className="flex gap-3">
              <button
                onClick={handleScore}
                disabled={!testUploaded || scoring}
                className="px-5 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
              >
                {scoring ? "Scoring…" : "Score All"}
              </button>
              {leaderboard.length > 0 && (
                <button
                  onClick={handleExport}
                  className="px-5 py-2 border border-gray-300 text-gray-700 rounded-lg text-sm hover:bg-gray-50"
                >
                  Export CSV
                </button>
              )}
            </div>
          </div>
          {scoreError && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">
              {scoreError}
            </div>
          )}

          {leaderboard.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-xs text-gray-500 uppercase tracking-wide">
                    <th className="pb-2 pr-4">#</th>
                    <th className="pb-2 pr-4">User</th>
                    <th className="pb-2 pr-4">ROC-AUC</th>
                    <th className="pb-2 pr-4">Precision</th>
                    <th className="pb-2 pr-4">Recall</th>
                    <th className="pb-2 pr-4">F1</th>
                    <th className="pb-2 pr-4">Last updated</th>
                    <th className="pb-2">Compare</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.map((entry, i) => (
                    <tr key={entry.username} className="border-b last:border-0 hover:bg-gray-50">
                      <td className="py-2 pr-4 text-gray-400">{i + 1}</td>
                      <td className="py-2 pr-4">
                        <a href={entry.pr_url} target="_blank" rel="noreferrer" className="text-blue-600 underline">
                          {entry.username}
                        </a>
                      </td>
                      <td className="py-2 pr-4 font-bold text-blue-700">
                        {entry.roc_auc != null ? entry.roc_auc.toFixed(4) : <span className="text-red-400 text-xs">{entry.error}</span>}
                      </td>
                      <td className="py-2 pr-4">{entry.precision?.toFixed(4) ?? "—"}</td>
                      <td className="py-2 pr-4">{entry.recall?.toFixed(4) ?? "—"}</td>
                      <td className="py-2 pr-4">{entry.f1?.toFixed(4) ?? "—"}</td>
                      <td className="py-2 pr-4 text-xs text-gray-400">
                        {new Date(entry.updated_at).toLocaleString()}
                      </td>
                      <td className="py-2">
                        <div className="flex gap-1">
                          <button
                            onClick={() => setCompareA(entry.username === compareA ? null : entry.username)}
                            className={`px-2 py-0.5 rounded text-xs border ${compareA === entry.username ? "bg-blue-100 border-blue-400 text-blue-700" : "border-gray-200"}`}
                          >
                            A
                          </button>
                          <button
                            onClick={() => setCompareB(entry.username === compareB ? null : entry.username)}
                            className={`px-2 py-0.5 rounded text-xs border ${compareB === entry.username ? "bg-amber-100 border-amber-400 text-amber-700" : "border-gray-200"}`}
                          >
                            B
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* ROC Overlay */}
        {leaderboard.some((e) => e.roc_curve) && (
          <section className="bg-white rounded-xl shadow p-6">
            <h2 className="text-base font-semibold mb-4">ROC Curve Overlay (all submissions)</h2>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart>
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: "FPR", position: "insideBottomRight", offset: -5 }} />
                <YAxis domain={[0, 1]} label={{ value: "TPR", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Legend />
                <ReferenceLine stroke="#d1d5db" strokeDasharray="4 4" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
                {leaderboard
                  .filter((e) => e.roc_curve)
                  .map((entry, i) => {
                    const data = entry.roc_curve!.fpr.map((fpr, j) => ({
                      fpr,
                      tpr: entry.roc_curve!.tpr[j],
                    }));
                    return (
                      <Line
                        key={entry.username}
                        data={data}
                        type="monotone"
                        dataKey="tpr"
                        name={`${entry.username} (${entry.roc_auc?.toFixed(3)})`}
                        stroke={`hsl(${(i * 60) % 360}, 70%, 45%)`}
                        dot={false}
                        strokeWidth={2}
                      />
                    );
                  })}
              </LineChart>
            </ResponsiveContainer>
          </section>
        )}

        {/* Side-by-side compare */}
        {entryA && entryB && (
          <section className="bg-white rounded-xl shadow p-6">
            <h2 className="text-base font-semibold mb-4">
              Compare: <span className="text-blue-600">{compareA}</span> vs{" "}
              <span className="text-amber-600">{compareB}</span>
            </h2>
            <div className="grid grid-cols-2 gap-6">
              {([entryA, entryB] as LeaderboardEntry[]).map((entry, i) => (
                <div key={entry.username} className="space-y-3">
                  <h3 className={`font-semibold ${i === 0 ? "text-blue-600" : "text-amber-600"}`}>
                    {entry.username}
                  </h3>
                  {entry.confusion_matrix && (
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Confusion Matrix</p>
                      <table className="text-sm text-center border-collapse w-full">
                        <thead>
                          <tr>
                            <th />
                            <th className="text-gray-400 font-normal p-1">Pred 0</th>
                            <th className="text-gray-400 font-normal p-1">Pred 1</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td className="text-gray-400 p-1">Actual 0</td>
                            <td className="bg-green-100 text-green-800 font-bold p-2 rounded">{entry.confusion_matrix.tn}</td>
                            <td className="bg-red-100 text-red-800 font-bold p-2 rounded">{entry.confusion_matrix.fp}</td>
                          </tr>
                          <tr>
                            <td className="text-gray-400 p-1">Actual 1</td>
                            <td className="bg-red-100 text-red-800 font-bold p-2 rounded">{entry.confusion_matrix.fn}</td>
                            <td className="bg-green-100 text-green-800 font-bold p-2 rounded">{entry.confusion_matrix.tp}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  )}
                  <table className="w-full text-sm">
                    <tbody>
                      {[
                        ["ROC-AUC", entry.roc_auc?.toFixed(4)],
                        ["Precision", entry.precision?.toFixed(4)],
                        ["Recall", entry.recall?.toFixed(4)],
                        ["F1", entry.f1?.toFixed(4)],
                      ].map(([k, v]) => (
                        <tr key={k} className="border-b last:border-0">
                          <td className="py-1 text-gray-500">{k}</td>
                          <td className="py-1 font-semibold text-right">{v ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

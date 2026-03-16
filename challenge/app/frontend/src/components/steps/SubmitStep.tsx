import { useEffect, useState } from "react";
import { api, DeadlineInfo } from "../../api";
import { useWizard } from "../../WizardContext";

function useCountdown(secondsRemaining: number | null) {
  const [secs, setSecs] = useState(secondsRemaining ?? 0);
  useEffect(() => {
    if (!secondsRemaining || secondsRemaining <= 0) return;
    setSecs(secondsRemaining);
    const id = setInterval(() => setSecs((s) => Math.max(0, s - 1)), 1000);
    return () => clearInterval(id);
  }, [secondsRemaining]);
  return secs;
}

function formatCountdown(secs: number): string {
  const d = Math.floor(secs / 86400);
  const h = Math.floor((secs % 86400) / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (d > 0) return `${d}d ${h}h ${m}m ${s}s`;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export default function SubmitStep() {
  const { config, trainResult, setStep, lastPrUrl, setLastPrUrl, currentUser } = useWizard();
  const [deadline, setDeadline] = useState<DeadlineInfo | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submittedAt, setSubmittedAt] = useState<string | null>(null);

  const secs = useCountdown(deadline?.seconds_remaining ?? null);
  const deadlinePassed = deadline?.has_passed || secs === 0;

  useEffect(() => {
    api.deadline().then(setDeadline);
  }, []);

  const handleSubmit = async () => {
    if (!trainResult) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await api.submit({
        pipeline_config: config,
        model_b64: trainResult.model_b64,
      });
      setLastPrUrl(res.pr_url);
      setSubmittedAt(res.submitted_at);
    } catch (e: any) {
      setError(e.message ?? "Submission failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Deadline banner */}
      <div
        className={`rounded-xl p-4 text-center text-sm font-medium ${
          deadlinePassed
            ? "bg-red-50 border border-red-200 text-red-700"
            : "bg-blue-50 border border-blue-200 text-blue-700"
        }`}
      >
        {deadline?.deadline ? (
          deadlinePassed ? (
            "⛔ Submission deadline has passed."
          ) : (
            <>
              ⏱ Time remaining:{" "}
              <span className="font-mono text-base">{formatCountdown(secs)}</span>
            </>
          )
        ) : (
          "No deadline configured."
        )}
      </div>

      {/* Submitting as */}
      {currentUser && (
        <div className="bg-white rounded-xl shadow p-4 flex items-center gap-3 text-sm text-gray-600">
          <img src={currentUser.avatar_url} alt={currentUser.login} className="w-8 h-8 rounded-full" />
          <span>Submitting as <strong className="text-gray-900">{currentUser.login}</strong></span>
        </div>
      )}

      {/* Pipeline config preview */}
      <div className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold mb-2">Pipeline Configuration Preview</h2>
        <pre className="bg-gray-50 rounded-lg p-4 text-xs overflow-auto max-h-64 text-gray-700">
          {JSON.stringify(config, null, 2)}
        </pre>
      </div>

      {/* Submission result */}
      {lastPrUrl && (
        <div className="bg-green-50 border border-green-200 rounded-xl p-4">
          <div className="font-semibold text-green-800 mb-1">
            ✅ Submitted successfully {submittedAt && `— ${submittedAt}`}
          </div>
          <a
            href={lastPrUrl}
            target="_blank"
            rel="noreferrer"
            className="text-sm text-green-700 underline break-all"
          >
            {lastPrUrl}
          </a>
          <p className="mt-2 text-xs text-green-600">
            You can resubmit before the deadline — each submission updates this
            Pull Request.
          </p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 text-sm space-y-3">
          <p>{error}</p>
          {error.includes("Grant") && (
            <a
              href="/api/auth/github"
              className="inline-flex items-center gap-2 px-4 py-2 bg-red-700 text-white rounded-lg hover:bg-red-800 text-sm font-medium"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.39.6.11.82-.26.82-.58v-2.17c-3.34.73-4.04-1.61-4.04-1.61-.54-1.38-1.32-1.75-1.32-1.75-1.08-.74.08-.73.08-.73 1.2.08 1.83 1.23 1.83 1.23 1.06 1.82 2.79 1.29 3.47.99.1-.77.41-1.29.75-1.59-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02.004 2.05.14 3 .4 2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.24 2.87.12 3.17.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.7.83.58C20.57 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z" />
              </svg>
              Re-authorize with GitHub
            </a>
          )}
        </div>
      )}

      <div className="flex justify-between">
        <button
          onClick={() => setStep(3)}
          className="px-6 py-2 border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-50"
        >
          ← Back: Train & Score
        </button>
        <button
          onClick={handleSubmit}
          disabled={!trainResult || deadlinePassed || submitting}
          className="px-8 py-2 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 disabled:opacity-50"
        >
          {submitting
            ? "Submitting…"
            : deadlinePassed
            ? "Deadline passed"
            : lastPrUrl
            ? "Resubmit"
            : "Submit Final Solution"}
        </button>
      </div>
    </div>
  );
}

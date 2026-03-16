import { useState, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { api, SplitMetrics, ConfusionMatrix } from "../../api";
import { useWizard } from "../../WizardContext";

export default function ScoringStep() {
  const { config, trainResult, setTrainResult, setStep } = useWizard();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [liveMetrics, setLiveMetrics] = useState<{
    confusion_matrix: ConfusionMatrix;
    precision: number;
    recall: number;
    f1: number;
  } | null>(null);

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.train(config);
      setTrainResult(result);
      setThreshold(0.5);
      setLiveMetrics(null);
    } catch (e: any) {
      setError(e.message ?? "Training failed");
    } finally {
      setLoading(false);
    }
  };

  const handleThreshold = useCallback(
    async (t: number) => {
      setThreshold(t);
      if (!trainResult) return;
      try {
        const res = await api.evaluateThreshold({
          threshold: t,
          y_true: trainResult.val_y_true,
          y_proba: trainResult.val_y_proba,
        });
        setLiveMetrics(res);
      } catch {
        // best-effort
      }
    },
    [trainResult]
  );

  const shownMetrics = liveMetrics ?? (trainResult?.validation ?? null);
  const cm = shownMetrics?.confusion_matrix ?? null;

  return (
    <div className="space-y-6">
      {/* Train button */}
      <div className="bg-white rounded-xl shadow p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">Train Model</h2>
          <p className="text-sm text-gray-500">
            Trains a new model with your current configuration and evaluates it
            on your validation and test splits.
          </p>
        </div>
        <button
          onClick={handleTrain}
          disabled={loading}
          className="px-8 py-3 bg-blue-600 text-white rounded-xl font-semibold text-base hover:bg-blue-700 disabled:opacity-50 whitespace-nowrap"
        >
          {loading ? "Training…" : "▶ Train"}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 text-sm">
          {error}
        </div>
      )}

      {trainResult && (
        <>
          {/* ROC-AUC badges */}
          <div className="grid grid-cols-2 gap-4">
            <AucBadge label="Validation ROC-AUC" value={trainResult.validation.roc_auc} />
            <AucBadge label="Test ROC-AUC" value={trainResult.test.roc_auc} />
          </div>

          {/* ROC Curve */}
          <div className="bg-white rounded-xl shadow p-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-4">
              ROC Curve (Validation)
            </h3>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart
                data={trainResult.validation.roc_curve.fpr.map((fpr, i) => ({
                  fpr,
                  tpr: trainResult.validation.roc_curve.tpr[i],
                }))}
              >
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: "FPR", position: "insideBottomRight", offset: -5 }} tickCount={6} />
                <YAxis domain={[0, 1]} label={{ value: "TPR", angle: -90, position: "insideLeft" }} tickCount={6} />
                <Tooltip formatter={(v: number) => v.toFixed(3)} />
                <ReferenceLine stroke="#d1d5db" strokeDasharray="4 4" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
                <Line type="monotone" dataKey="tpr" stroke="#3b82f6" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Confusion Matrix + Threshold */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="bg-white rounded-xl shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-700">
                  Confusion Matrix (Validation, threshold = {threshold.toFixed(2)})
                </h3>
              </div>
              <div className="mb-4">
                <input
                  type="range"
                  min={0.01}
                  max={0.99}
                  step={0.01}
                  value={threshold}
                  onChange={(e) => handleThreshold(Number(e.target.value))}
                  className="w-full accent-blue-600"
                />
                <div className="flex justify-between text-xs text-gray-400">
                  <span>0.01</span>
                  <span>0.99</span>
                </div>
              </div>
              {cm && <CMTable cm={cm} />}
            </div>

            <div className="bg-white rounded-xl shadow p-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-4">
                Classification Metrics (Validation)
              </h3>
              {shownMetrics && (
                <table className="w-full text-sm">
                  <tbody>
                    {[
                      ["Precision", shownMetrics.precision],
                      ["Recall", shownMetrics.recall],
                      ["F1 Score", shownMetrics.f1],
                    ].map(([label, val]) => (
                      <tr key={label as string} className="border-b last:border-0">
                        <td className="py-2 text-gray-600">{label}</td>
                        <td className="py-2 font-semibold text-right text-blue-700">
                          {(val as number).toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          {/* Split sizes */}
          <div className="bg-gray-50 rounded-xl p-4 text-sm text-gray-600 flex gap-6">
            <span>Train: <b>{trainResult.split_sizes.train}</b></span>
            <span>Val: <b>{trainResult.split_sizes.val}</b></span>
            <span>Test: <b>{trainResult.split_sizes.test}</b></span>
          </div>
        </>
      )}

      <div className="flex justify-between">
        <button
          onClick={() => setStep(2)}
          className="px-6 py-2 border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-50"
        >
          ← Back: Model
        </button>
        <button
          onClick={() => setStep(4)}
          disabled={!trainResult}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
        >
          Next: Submit →
        </button>
      </div>
    </div>
  );
}

function AucBadge({ label, value }: { label: string; value: number }) {
  const color =
    value >= 0.9
      ? "text-green-600 bg-green-50 border-green-300"
      : value >= 0.7
      ? "text-blue-600 bg-blue-50 border-blue-300"
      : "text-amber-600 bg-amber-50 border-amber-300";
  return (
    <div className={`border rounded-xl p-4 text-center ${color}`}>
      <div className="text-3xl font-extrabold">{value.toFixed(4)}</div>
      <div className="text-xs mt-1">{label}</div>
    </div>
  );
}

function CMTable({ cm }: { cm: ConfusionMatrix }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse text-center">
        <thead>
          <tr>
            <th />
            <th className="p-2 text-gray-500 font-medium">Pred 0</th>
            <th className="p-2 text-gray-500 font-medium">Pred 1</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="p-2 text-gray-500 font-medium">Actual 0</td>
            <td className="p-2 bg-green-100 text-green-800 font-bold rounded">{cm.tn}</td>
            <td className="p-2 bg-red-100 text-red-800 font-bold rounded">{cm.fp}</td>
          </tr>
          <tr>
            <td className="p-2 text-gray-500 font-medium">Actual 1</td>
            <td className="p-2 bg-red-100 text-red-800 font-bold rounded">{cm.fn}</td>
            <td className="p-2 bg-green-100 text-green-800 font-bold rounded">{cm.tp}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

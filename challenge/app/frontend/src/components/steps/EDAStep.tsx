import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell,
} from "recharts";
import { api, PreviewData, CorrelationData, Schema } from "../../api";
import { useWizard } from "../../WizardContext";

const COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"];

// Map correlation -1â†’red, 0â†’white, 1â†’blue
function corrColor(v: number | null): string {
  if (v === null) return "#e5e7eb";
  const clamped = Math.max(-1, Math.min(1, v));
  if (clamped >= 0) {
    const t = clamped;
    const r = Math.round(255 * (1 - t));
    const g = Math.round(255 * (1 - t));
    return `rgb(${r},${g},255)`;
  } else {
    const t = -clamped;
    const b = Math.round(255 * (1 - t));
    const g = Math.round(255 * (1 - t));
    return `rgb(255,${g},${b})`;
  }
}

export default function EDAStep() {
  const { config, setConfig, setStep } = useWizard();
  const [preview, setPreview] = useState<PreviewData | null>(null);
  const [schema, setSchema] = useState<Schema | null>(null);
  const [corr, setCorr] = useState<CorrelationData | null>(null);
  const [selectedCol, setSelectedCol] = useState<string | null>(null);
  const [colDist, setColDist] = useState<{ label: string; count: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCorr, setShowCorr] = useState(false);

  // Split state (local, saved to config on Next)
  const [trainPct, setTrainPct] = useState(Math.round(config.split.train * 100));
  const [valPct, setValPct] = useState(Math.round(config.split.val * 100));
  const testPct = 100 - trainPct - valPct;

  useEffect(() => {
    Promise.all([api.preview(), api.correlations(), api.schema()])
      .then(([p, c, s]) => {
        setPreview(p);
        setCorr(c);
        setSchema(s);
        // Default: select all feature columns (excludes id and target)
        if (config.selected_features.length === 0) {
          const featureCols = p.columns
            .map((col) => col.name)
            .filter((name) => name !== s.id_column && name !== s.target_column);
          setConfig({ ...config, selected_features: featureCols });
        }
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedCol) return;
    api.eda(selectedCol).then((d) => {
      if (d.type === "categorical") {
        setColDist((d.data as any[]).map((x) => ({ label: x.label, count: x.count })));
      } else {
        setColDist(
          (d.data as any[]).map((x) => ({
            label: `${x.bin_start}`,
            count: x.count,
          }))
        );
      }
    });
  }, [selectedCol]);

  const handleNext = () => {
    setConfig({
      ...config,
      split: {
        train: trainPct / 100,
        val: valPct / 100,
        test: testPct / 100,
      },
    });
    setStep(1);
  };

  if (loading) return <div className="text-center py-20 text-gray-400">Loading datasetâ€¦</div>;
  if (!preview || !schema) return null;

  const balanceData = Object.entries(preview.class_balance).map(([k, v]) => ({
    label: k,
    count: v,
  }));

  // Only show feature columns (not id / target) in selection and explorer
  const featureColumns = preview.columns.filter(
    (col) => col.name !== schema.id_column && col.name !== schema.target_column
  );

  return (
    <div className="space-y-8">
      {/* Dataset summary */}
      <section className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Dataset Overview</h2>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 mb-6">
          <Stat label="Rows" value={preview.shape.rows} />
          <Stat label="Columns" value={preview.shape.cols} />
          <Stat
            label="Missing values"
            value={preview.columns.reduce((s, c) => s + c.missing, 0)}
          />
          <Stat
            label="Class ratio"
            value={`${Object.values(preview.class_balance).join(" / ")}`}
          />
        </div>

        {/* Class balance chart */}
        <h3 className="text-sm font-medium text-gray-600 mb-2">Class Balance</h3>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={balanceData}>
            <XAxis dataKey="label" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count">
              {balanceData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </section>

      {/* Column explorer */}
      <section className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Column Explorer</h2>
        <div className="flex flex-wrap gap-2 mb-4">
          {featureColumns.map((col) => (
            <button
              key={col.name}
              onClick={() => setSelectedCol(col.name === selectedCol ? null : col.name)}
              className={`px-3 py-1 rounded-full text-xs border transition-colors ${
                selectedCol === col.name
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-gray-600 border-gray-300 hover:border-blue-400"
              }`}
            >
              {col.name}
              {col.missing_pct > 0 && (
                <span className="ml-1 text-amber-500">({col.missing_pct}%)</span>
              )}
            </button>
          ))}
        </div>
        {selectedCol && colDist.length > 0 && (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={colDist}>
              <XAxis dataKey="label" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        )}
      </section>

      {/* Feature correlation matrix */}
      {corr && corr.columns.length > 0 && (
        <section className="bg-white rounded-xl shadow p-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">Feature Correlations</h2>
            <button
              onClick={() => setShowCorr((v) => !v)}
              className="text-sm text-blue-600 hover:underline"
            >
              {showCorr ? "Hide" : "Show"} matrix
            </button>
          </div>
          {showCorr && (
            <div className="overflow-x-auto">
              <table className="text-xs border-collapse">
                <thead>
                  <tr>
                    <th className="p-1" />
                    {corr.columns.map((col) => (
                      <th
                        key={col}
                        className="p-1 text-gray-500 font-normal max-w-[60px] truncate"
                        title={col}
                      >
                        <span className="block truncate max-w-[56px]">{col}</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {corr.matrix.map((row, i) => (
                    <tr key={corr.columns[i]}>
                      <td
                        className="p-1 text-gray-500 font-normal pr-2 max-w-[80px] truncate"
                        title={corr.columns[i]}
                      >
                        <span className="block truncate max-w-[76px]">{corr.columns[i]}</span>
                      </td>
                      {row.map((val, j) => (
                        <td
                          key={j}
                          title={val !== null ? val.toFixed(3) : "N/A"}
                          className="w-8 h-8 text-center cursor-default select-none"
                          style={{ backgroundColor: corrColor(val) }}
                        >
                          {i === j ? "" : val !== null ? (
                            <span className="text-[9px] text-gray-700">
                              {val.toFixed(2)}
                            </span>
                          ) : ""}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="flex items-center gap-4 mt-3 text-xs text-gray-500">
                <span>Correlation scale:</span>
                <span className="flex items-center gap-1">
                  <span className="inline-block w-4 h-4 rounded" style={{ backgroundColor: corrColor(-1) }} />
                  âˆ’1
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block w-4 h-4 rounded" style={{ backgroundColor: corrColor(0) }} />
                  0
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block w-4 h-4 rounded" style={{ backgroundColor: corrColor(1) }} />
                  +1
                </span>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Feature selection */}
      <section className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold mb-2">Feature Selection</h2>
        <p className="text-sm text-gray-500 mb-4">
          Deselect columns you want to exclude from training.
          <span className="ml-2 text-gray-400 text-xs">
            (id and target columns are excluded automatically)
          </span>
        </p>
        <div className="flex flex-wrap gap-2">
          {featureColumns.map((col) => {
            const selected = config.selected_features.includes(col.name);
            return (
              <button
                key={col.name}
                onClick={() =>
                  setConfig({
                    ...config,
                    selected_features: selected
                      ? config.selected_features.filter((c) => c !== col.name)
                      : [...config.selected_features, col.name],
                  })
                }
                className={`px-3 py-1 rounded-full text-xs border transition-colors ${
                  selected
                    ? "bg-green-100 text-green-700 border-green-400"
                    : "bg-gray-100 text-gray-400 border-gray-200 line-through"
                }`}
              >
                {col.name}
              </button>
            );
          })}
        </div>
        {config.selected_features.length === 0 && (
          <p className="text-xs text-red-500 mt-2">
            âš  No features selected â€” select at least one to train.
          </p>
        )}
      </section>

      {/* Train / Val / Test split */}
      <section className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold mb-2">Train / Val / Test Split</h2>
        <p className="text-sm text-gray-500 mb-4">
          Adjust ratios for your local evaluation. The instructors will evaluate
          your submitted model on a separate hidden test set.
        </p>
        <SplitSlider
          label="Train"
          value={trainPct}
          onChange={(v) => setTrainPct(Math.min(v, 98 - valPct))}
          color="bg-blue-500"
        />
        <SplitSlider
          label="Validation"
          value={valPct}
          onChange={(v) => setValPct(Math.min(v, 98 - trainPct))}
          color="bg-amber-400"
        />
        <div className="text-sm text-gray-600 mt-2">
          Test:{" "}
          <span className="font-semibold text-green-600">{testPct}%</span>
          {testPct < 1 && (
            <span className="ml-2 text-red-500 text-xs">
              âš  Test split too small â€” adjust train or val
            </span>
          )}
        </div>
      </section>

      <div className="flex justify-end">
        <button
          onClick={handleNext}
          disabled={testPct < 1 || config.selected_features.length === 0}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
        >
          Next: Preprocessing â†’
        </button>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-50 rounded-lg p-3 text-center">
      <div className="text-2xl font-bold text-blue-600">{value}</div>
      <div className="text-xs text-gray-500 mt-1">{label}</div>
    </div>
  );
}

function SplitSlider({
  label,
  value,
  onChange,
  color,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  color: string;
}) {
  return (
    <div className="flex items-center gap-4 mb-3">
      <span className="w-24 text-sm text-gray-600">{label}</span>
      <input
        type="range"
        min={1}
        max={97}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 accent-blue-600"
      />
      <span className={`w-12 text-right font-semibold text-sm`}>{value}%</span>
    </div>
  );
}

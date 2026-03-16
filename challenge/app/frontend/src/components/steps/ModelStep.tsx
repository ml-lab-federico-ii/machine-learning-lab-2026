import { useWizard } from "../../WizardContext";

const MODELS = [
  {
    value: "logistic_regression",
    label: "Logistic Regression",
    params: [
      { key: "C", label: "Regularisation (C)", min: 0.01, max: 10, step: 0.01, default: 1.0 },
    ],
  },
  {
    value: "random_forest",
    label: "Random Forest",
    params: [
      { key: "n_estimators", label: "Number of trees", min: 10, max: 300, step: 10, default: 100 },
      { key: "max_depth", label: "Max depth", min: 1, max: 6, step: 1, default: 4 },
    ],
  },
  {
    value: "gradient_boosting",
    label: "Gradient Boosting",
    params: [
      { key: "n_estimators", label: "Number of boosting rounds", min: 10, max: 300, step: 10, default: 100 },
      { key: "max_depth", label: "Max depth", min: 1, max: 6, step: 1, default: 3 },
      { key: "learning_rate", label: "Learning rate", min: 0.01, max: 0.5, step: 0.01, default: 0.1 },
    ],
  },
  {
    value: "xgboost",
    label: "XGBoost (CPU)",
    params: [
      { key: "n_estimators", label: "Number of boosting rounds", min: 10, max: 300, step: 10, default: 100 },
      { key: "max_depth", label: "Max depth", min: 1, max: 6, step: 1, default: 4 },
      { key: "learning_rate", label: "Learning rate", min: 0.01, max: 0.5, step: 0.01, default: 0.1 },
    ],
  },
  {
    value: "lightgbm",
    label: "LightGBM (CPU)",
    params: [
      { key: "n_estimators", label: "Number of trees", min: 10, max: 300, step: 10, default: 100 },
      { key: "max_depth", label: "Max depth", min: 1, max: 6, step: 1, default: 4 },
      { key: "learning_rate", label: "Learning rate", min: 0.01, max: 0.5, step: 0.01, default: 0.1 },
    ],
  },
];

export default function ModelStep() {
  const { config, setConfig, setStep } = useWizard();
  const modelName = config.model.name;
  const modelDef = MODELS.find((m) => m.value === modelName) ?? MODELS[0];

  const setModel = (name: string) => {
    const def = MODELS.find((m) => m.value === name) ?? MODELS[0];
    const params = Object.fromEntries(def.params.map((p) => [p.key, p.default]));
    setConfig({ ...config, model: { name, params } });
  };

  const setParam = (key: string, value: number) => {
    setConfig({
      ...config,
      model: { ...config.model, params: { ...config.model.params, [key]: value } },
    });
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow p-6 space-y-6">
        <h2 className="text-lg font-semibold">Model Configuration</h2>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Algorithm
          </label>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {MODELS.map((m) => (
              <button
                key={m.value}
                onClick={() => setModel(m.value)}
                className={`border rounded-lg px-4 py-3 text-left transition-colors ${
                  modelName === m.value
                    ? "border-blue-500 bg-blue-50 text-blue-700"
                    : "border-gray-200 hover:border-blue-300"
                }`}
              >
                <span className="font-medium text-sm">{m.label}</span>
              </button>
            ))}
          </div>
        </div>

        {modelDef.params.length > 0 && (
          <div className="space-y-5">
            <h3 className="text-sm font-semibold text-gray-700">Hyperparameters</h3>
            {modelDef.params.map((p) => {
              const val = Number(config.model.params[p.key] ?? p.default);
              return (
                <div key={p.key}>
                  <div className="flex justify-between mb-1">
                    <label className="text-sm text-gray-600">{p.label}</label>
                    <span className="text-sm font-semibold text-blue-700">{val}</span>
                  </div>
                  <input
                    type="range"
                    min={p.min}
                    max={p.max}
                    step={p.step}
                    value={val}
                    onChange={(e) => setParam(p.key, Number(e.target.value))}
                    className="w-full accent-blue-600"
                  />
                  <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                    <span>{p.min}</span>
                    <span>{p.max}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="flex justify-between">
        <button
          onClick={() => setStep(1)}
          className="px-6 py-2 border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-50"
        >
          ← Back: Preprocessing
        </button>
        <button
          onClick={() => setStep(3)}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700"
        >
          Next: Train & Score →
        </button>
      </div>
    </div>
  );
}

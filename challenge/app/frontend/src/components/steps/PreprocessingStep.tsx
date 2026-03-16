import { useWizard } from "../../WizardContext";

type PreprocConfig = typeof import("../../api").api extends never
  ? never
  : {
      imputation_numeric: string;
      imputation_categorical: string;
      scaling: string;
      encoding: string;
      imbalance_handling: string;
    };

const NUM_IMPUTE_OPTIONS = [
  { value: "mean", label: "Mean" },
  { value: "median", label: "Median" },
  { value: "most_frequent", label: "Most Frequent" },
];

const CAT_IMPUTE_OPTIONS = [
  { value: "most_frequent", label: "Most Frequent" },
  { value: "constant", label: "Constant (\"missing\")" },
];

const SCALING_OPTIONS = [
  { value: "standard", label: "Standard Scaler (z-score)" },
  { value: "minmax", label: "Min-Max Scaler [0, 1]" },
  { value: "none", label: "No scaling" },
];

const ENCODING_OPTIONS = [
  { value: "onehot", label: "One-Hot Encoding" },
  { value: "ordinal", label: "Ordinal Encoding" },
  { value: "label", label: "Label Encoding" },
];

const IMBALANCE_OPTIONS = [
  { value: "none", label: "None (ignore imbalance)" },
  { value: "class_weight", label: "Balanced class weights" },
  { value: "smote", label: "SMOTE oversampling" },
];

export default function PreprocessingStep() {
  const { config, setConfig, setStep } = useWizard();
  const prep = config.preprocessing;

  const update = (key: keyof typeof prep, value: string) =>
    setConfig({
      ...config,
      preprocessing: { ...prep, [key]: value },
    });

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow p-6 space-y-6">
        <h2 className="text-lg font-semibold">Preprocessing Configuration</h2>

        <FormRow label="Numeric imputation" hint="Strategy for filling missing numeric values">
          <Select
            value={prep.imputation_numeric}
            onChange={(v) => update("imputation_numeric", v)}
            options={NUM_IMPUTE_OPTIONS}
          />
        </FormRow>

        <FormRow label="Categorical imputation" hint="Strategy for filling missing categorical values">
          <Select
            value={prep.imputation_categorical}
            onChange={(v) => update("imputation_categorical", v)}
            options={CAT_IMPUTE_OPTIONS}
          />
        </FormRow>

        <FormRow label="Feature scaling" hint="Applied to numeric features after imputation">
          <Select
            value={prep.scaling}
            onChange={(v) => update("scaling", v)}
            options={SCALING_OPTIONS}
          />
        </FormRow>

        <FormRow label="Categorical encoding" hint="How to represent categorical columns as numbers">
          <Select
            value={prep.encoding}
            onChange={(v) => update("encoding", v)}
            options={ENCODING_OPTIONS}
          />
        </FormRow>

        <FormRow
          label="Class imbalance handling"
          hint="Technique for dealing with unequal class distribution"
        >
          <Select
            value={prep.imbalance_handling}
            onChange={(v) => update("imbalance_handling", v)}
            options={IMBALANCE_OPTIONS}
          />
        </FormRow>
      </div>

      <div className="flex justify-between">
        <button
          onClick={() => setStep(0)}
          className="px-6 py-2 border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-50"
        >
          ← Back: EDA
        </button>
        <button
          onClick={() => setStep(2)}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700"
        >
          Next: Model →
        </button>
      </div>
    </div>
  );
}

function FormRow({
  label,
  hint,
  children,
}: {
  label: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-1 gap-1 sm:grid-cols-3 sm:items-center">
      <div>
        <div className="font-medium text-sm text-gray-800">{label}</div>
        <div className="text-xs text-gray-400">{hint}</div>
      </div>
      <div className="sm:col-span-2">{children}</div>
    </div>
  );
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

import { createContext, useContext, useState, ReactNode } from "react";
import type { PipelineConfig, TrainResult, AuthMe } from "./api";

interface WizardState {
  step: number;
  setStep: (s: number) => void;
  config: PipelineConfig;
  setConfig: (c: PipelineConfig) => void;
  trainResult: TrainResult | null;
  setTrainResult: (r: TrainResult | null) => void;
  lastPrUrl: string | null;
  setLastPrUrl: (url: string | null) => void;
  currentUser: AuthMe["user"] | null;
}

const defaultConfig: PipelineConfig = {
  split: { train: 0.7, val: 0.15, test: 0.15 },
  selected_features: [],
  preprocessing: {
    imputation_numeric: "mean",
    imputation_categorical: "most_frequent",
    scaling: "standard",
    encoding: "onehot",
    imbalance_handling: "none",
  },
  model: {
    name: "logistic_regression",
    params: { C: 1.0 },
  },
};

const WizardContext = createContext<WizardState>({} as WizardState);

export function WizardProvider({ children, initialUser }: { children: ReactNode; initialUser?: AuthMe["user"] }) {
  const [step, setStep] = useState(0);
  const [config, setConfig] = useState<PipelineConfig>(defaultConfig);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [lastPrUrl, setLastPrUrl] = useState<string | null>(null);

  return (
    <WizardContext.Provider
      value={{ step, setStep, config, setConfig, trainResult, setTrainResult, lastPrUrl, setLastPrUrl, currentUser: initialUser ?? null }}
    >
      {children}
    </WizardContext.Provider>
  );
}

export const useWizard = () => useContext(WizardContext);

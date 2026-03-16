import { useWizard } from "../WizardContext";
import { clsx } from "clsx";

const STEPS = ["EDA", "Preprocessing", "Model", "Train & Score", "Submit"];

export default function Stepper() {
  const { step, setStep, trainResult } = useWizard();

  return (
    <nav className="flex items-center justify-center gap-0 mb-8">
      {STEPS.map((label, i) => {
        const isActive = i === step;
        const isCompleted = i < step;
        const isLocked = i === 4 && trainResult === null;

        return (
          <div key={i} className="flex items-center">
            <button
              onClick={() => !isLocked && setStep(i)}
              disabled={isLocked}
              className={clsx(
                "flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-colors",
                isActive && "bg-blue-600 text-white shadow",
                isCompleted && !isActive && "bg-blue-100 text-blue-700 hover:bg-blue-200",
                !isActive && !isCompleted && !isLocked && "bg-white text-gray-600 hover:bg-gray-100 border border-gray-200",
                isLocked && "bg-gray-100 text-gray-400 cursor-not-allowed"
              )}
            >
              <span
                className={clsx(
                  "w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold",
                  isActive && "bg-white text-blue-600",
                  isCompleted && !isActive && "bg-blue-600 text-white",
                  !isActive && !isCompleted && "bg-gray-300 text-gray-600"
                )}
              >
                {isCompleted ? "✓" : i + 1}
              </span>
              {label}
            </button>
            {i < STEPS.length - 1 && (
              <div className={clsx("w-8 h-px", i < step ? "bg-blue-400" : "bg-gray-200")} />
            )}
          </div>
        );
      })}
    </nav>
  );
}

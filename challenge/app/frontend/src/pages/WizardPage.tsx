import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { WizardProvider, useWizard } from "../WizardContext";
import Stepper from "../components/Stepper";
import EDAStep from "../components/steps/EDAStep";
import PreprocessingStep from "../components/steps/PreprocessingStep";
import ModelStep from "../components/steps/ModelStep";
import ScoringStep from "../components/steps/ScoringStep";
import SubmitStep from "../components/steps/SubmitStep";
import { api, AuthMe } from "../api";

function LoginGate() {
  const [loading, setLoading] = useState(false);
  const [params] = useSearchParams();

  const authError = params.get("auth_error");
  const errorMessage =
    authError === "cancelled"
      ? "Login cancelled \u2014 please try again."
      : authError === "token_failed"
      ? "Login failed \u2014 GitHub did not return a token. Please try again."
      : authError
      ? "Something went wrong \u2014 please try again."
      : null;

  useEffect(() => {
    if (authError) {
      window.history.replaceState({}, "", "/");
    }
  }, [authError]);

  const handleLogin = () => {
    setLoading(true);
    window.location.href = "/api/auth/github";
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-md p-10 w-full max-w-md text-center space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">ML Challenge</h1>
          <p className="text-sm text-gray-400 mt-1">
            Università degli Studi di Napoli Federico II
          </p>
        </div>
        <p className="text-sm text-gray-600">
          Build and submit an end-to-end machine learning pipeline — guided step by step.
          Log in with GitHub to get started.
        </p>
        {errorMessage && (
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-lg px-4 py-2.5 text-sm">
            {errorMessage}
          </div>
        )}
        <button
          onClick={handleLogin}
          disabled={loading}
          className="w-full flex items-center justify-center gap-3 px-6 py-3 bg-gray-900 text-white rounded-xl hover:bg-gray-700 font-medium text-sm transition-colors disabled:opacity-60"
        >
          {loading ? (
            <>
              <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Redirecting to GitHub…
            </>
          ) : (
            <>
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.39.6.11.82-.26.82-.58v-2.17c-3.34.73-4.04-1.61-4.04-1.61-.54-1.38-1.32-1.75-1.32-1.75-1.08-.74.08-.73.08-.73 1.2.08 1.83 1.23 1.83 1.23 1.06 1.82 2.79 1.29 3.47.99.1-.77.41-1.29.75-1.59-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02.004 2.05.14 3 .4 2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.24 2.87.12 3.17.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.7.83.58C20.57 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z" />
              </svg>
              Login with GitHub
            </>
          )}
        </button>
      </div>
    </div>
  );
}

function WizardContent({ auth, onLogout }: { auth: AuthMe; onLogout: () => void }) {
  const { step } = useWizard();
  const steps = [
    <EDAStep />,
    <PreprocessingStep />,
    <ModelStep />,
    <ScoringStep />,
    <SubmitStep />,
  ];
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 py-4 px-6 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-gray-900">ML Challenge</h1>
          <p className="text-xs text-gray-400">
            Università degli Studi di Napoli Federico II
          </p>
        </div>
        {auth.user && (
          <div className="flex items-center gap-3 text-sm text-gray-600">
            <img src={auth.user.avatar_url} alt={auth.user.login} className="w-7 h-7 rounded-full" />
            <span>{auth.user.login}</span>
            <button
              onClick={onLogout}
              className="text-xs text-gray-400 hover:text-gray-600 underline"
            >
              Log out
            </button>
          </div>
        )}
      </header>
      <main className="max-w-4xl mx-auto px-4 py-8">
        <Stepper />
        {steps[step]}
      </main>
    </div>
  );
}

export default function WizardPage() {
  const [auth, setAuth] = useState<AuthMe | null>(null);

  useEffect(() => {
    api.authMe().then(setAuth);
  }, []);

  if (auth === null) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center text-gray-400 text-sm">
        Loading…
      </div>
    );
  }

  if (!auth.authenticated) {
    return <LoginGate />;
  }

  return (
    <WizardProvider initialUser={auth.user}>
      <WizardContent auth={auth} onLogout={async () => { await api.authLogout(); setAuth({ authenticated: false }); }} />
    </WizardProvider>
  );
}

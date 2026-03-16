import { Routes, Route } from "react-router-dom";
import WizardPage from "./pages/WizardPage";
import InstructorPage from "./pages/InstructorPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<WizardPage />} />
      <Route path="/instructor" element={<InstructorPage />} />
    </Routes>
  );
}

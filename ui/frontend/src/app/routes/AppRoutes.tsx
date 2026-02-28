import { Navigate, Route, Routes } from "react-router-dom";
import { ConfigurationPage } from "../../components/ConfigurationPage";
import { DashboardPage } from "../../components/DashboardPage";
import { EvaluationPage } from "../../components/EvaluationPage";
import { GovernancePage } from "../../components/GovernancePage";
import { MeetingsPage } from "../../components/MeetingsPage";
import { ReproducibilityPage } from "../../components/ReproducibilityPage";
import { RunDetailsPage } from "../../components/RunDetailsPage";
import { RunHistoryPage } from "../../components/RunHistoryPage";

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<DashboardPage />} />
      <Route path="/meetings" element={<MeetingsPage />} />
      <Route path="/meetings/:meetingId" element={<MeetingsPage />} />
      <Route path="/evaluation" element={<EvaluationPage />} />
      <Route path="/reproducibility" element={<ReproducibilityPage />} />
      <Route path="/configuration" element={<ConfigurationPage />} />
      <Route path="/governance" element={<GovernancePage />} />
      <Route path="/runs" element={<RunHistoryPage />} />
      <Route path="/runs/:runId" element={<RunDetailsPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

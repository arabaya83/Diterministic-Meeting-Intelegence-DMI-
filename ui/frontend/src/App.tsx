import { AppLayout } from "./app/layout/AppLayout";
import { AppRoutes } from "./app/routes/AppRoutes";

export default function App() {
  return (
    <AppLayout>
      <AppRoutes />
    </AppLayout>
  );
}

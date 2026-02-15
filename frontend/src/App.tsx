import { useState } from "react";
import { Layout } from "./components/Layout.tsx";
import type { View } from "./components/Layout.tsx";
import { GenerateForm } from "./components/GenerateForm.tsx";
import { ProgressView } from "./components/ProgressView.tsx";
import { ProjectList } from "./components/ProjectList.tsx";
import { ProjectDetail } from "./components/ProjectDetail.tsx";

function App() {
  const [currentView, setCurrentView] = useState<View>("generate");
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);

  function navigateTo(view: View, projectId?: string) {
    setCurrentView(view);
    if (projectId !== undefined) {
      setActiveProjectId(projectId);
    }
  }

  function handleGenerated(projectId: string) {
    navigateTo("progress", projectId);
  }

  function handleSelectProject(projectId: string) {
    navigateTo("detail", projectId);
  }

  function handleViewProgress(projectId: string) {
    navigateTo("progress", projectId);
  }

  function handleViewDetail(projectId: string) {
    navigateTo("detail", projectId);
  }

  return (
    <Layout currentView={currentView} onNavigate={(v) => navigateTo(v)}>
      {currentView === "generate" && (
        <GenerateForm onGenerated={handleGenerated} />
      )}
      {currentView === "progress" && activeProjectId && (
        <ProgressView
          projectId={activeProjectId}
          onViewDetail={handleViewDetail}
        />
      )}
      {currentView === "list" && (
        <ProjectList onSelectProject={handleSelectProject} />
      )}
      {currentView === "detail" && activeProjectId && (
        <ProjectDetail
          projectId={activeProjectId}
          onViewProgress={handleViewProgress}
        />
      )}
    </Layout>
  );
}

export default App;

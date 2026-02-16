import { useState } from "react";
import { Layout } from "./components/Layout.tsx";
import type { View } from "./components/Layout.tsx";
import { GenerateForm } from "./components/GenerateForm.tsx";
import { ProgressView } from "./components/ProgressView.tsx";
import { ProjectList } from "./components/ProjectList.tsx";
import { ProjectDetail } from "./components/ProjectDetail.tsx";
import { Dashboard } from "./components/Dashboard.tsx";
import { ManifestLibrary } from "./components/ManifestLibrary.tsx";
import { ManifestCreator } from "./components/ManifestCreator.tsx";

function App() {
  const [currentView, setCurrentView] = useState<View>("generate");
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [activeManifestId, setActiveManifestId] = useState<string | null>(null);

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

  function handleCreateManifest() {
    setActiveManifestId(null);
    navigateTo("manifest-creator");
  }

  function handleEditManifest(manifestId: string) {
    setActiveManifestId(manifestId);
    navigateTo("manifest-creator");
  }

  function handleViewManifest(manifestId: string) {
    setActiveManifestId(manifestId);
    navigateTo("manifest-creator"); // View opens creator in read mode for now
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
          onForked={(newId) => navigateTo("progress", newId)}
          onViewProject={(id) => navigateTo("detail", id)}
        />
      )}
      {currentView === "dashboard" && <Dashboard />}
      {currentView === "manifests" && (
        <ManifestLibrary
          onCreateNew={handleCreateManifest}
          onEditManifest={handleEditManifest}
          onViewManifest={handleViewManifest}
        />
      )}
      {currentView === "manifest-creator" && (
        <ManifestCreator
          manifestId={activeManifestId}
          onSaved={() => {
            setActiveManifestId(null);
            navigateTo("manifests");
          }}
          onCancel={() => {
            setActiveManifestId(null);
            navigateTo("manifests");
          }}
        />
      )}
    </Layout>
  );
}

export default App;

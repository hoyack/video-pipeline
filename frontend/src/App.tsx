import { Route, Switch, Redirect, useLocation } from "wouter";
import { Layout } from "./components/Layout.tsx";
import { GenerateForm } from "./components/GenerateForm.tsx";
import { ProgressView } from "./components/ProgressView.tsx";
import { ProjectList } from "./components/ProjectList.tsx";
import { ProjectDetail } from "./components/ProjectDetail.tsx";
import { Dashboard } from "./components/Dashboard.tsx";
import { ManifestLibrary } from "./components/ManifestLibrary.tsx";
import { ManifestCreator } from "./components/ManifestCreator.tsx";
import { SettingsPage } from "./components/SettingsPage.tsx";
import { createDraftProject } from "./api/client.ts";
import { ShowCostProvider } from "./hooks/useShowCost.tsx";

function App() {
  const [, navigate] = useLocation();

  async function handleNewDraft() {
    try {
      const res = await createDraftProject();
      navigate(`/projects/${res.project_id}`);
    } catch {
      navigate("/generate");
    }
  }

  return (
    <ShowCostProvider>
      <Layout>
        <Switch>
          <Route path="/">
            <ProjectList
              onSelectProject={(id) => navigate(`/projects/${id}`)}
              onNewProject={() => navigate("/generate")}
              onNewDraft={handleNewDraft}
            />
          </Route>
          <Route path="/generate">
            <GenerateForm
              onGenerated={(id) => navigate(`/projects/${id}/progress`)}
            />
          </Route>
          <Route path="/projects/:id/progress">
            {(params) => (
              <ProgressView
                projectId={params.id}
                onViewDetail={(id) => navigate(`/projects/${id}`)}
              />
            )}
          </Route>
          <Route path="/projects/:id">
            {(params) => (
              <ProjectDetail
                projectId={params.id}
                onViewProgress={(id) => navigate(`/projects/${id}/progress`)}
                onForked={(newId) => navigate(`/projects/${newId}/progress`)}
                onViewProject={(id) => navigate(`/projects/${id}`)}
                onViewManifest={(manifestId) =>
                  navigate(`/manifests/${manifestId}/edit`)
                }
              />
            )}
          </Route>
          <Route path="/dashboard">
            <Dashboard />
          </Route>
          <Route path="/manifests/new">
            <ManifestCreator
              manifestId={null}
              onSaved={() => navigate("/manifests")}
              onCancel={() => navigate("/manifests")}
            />
          </Route>
          <Route path="/manifests/:id/edit">
            {(params) => (
              <ManifestCreator
                manifestId={params.id}
                onSaved={() => navigate("/manifests")}
                onCancel={() => navigate("/manifests")}
              />
            )}
          </Route>
          <Route path="/manifests">
            <ManifestLibrary
              onCreateNew={() => navigate("/manifests/new")}
              onEditManifest={(id) => navigate(`/manifests/${id}/edit`)}
              onViewManifest={(id) => navigate(`/manifests/${id}/edit`)}
            />
          </Route>
          <Route path="/settings">
            <SettingsPage />
          </Route>
          <Route>
            <Redirect to="/" />
          </Route>
        </Switch>
      </Layout>
    </ShowCostProvider>
  );
}

export default App;

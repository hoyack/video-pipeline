import { Route, Switch, Redirect, useLocation } from "wouter";
import { Layout } from "./components/Layout.tsx";
import { GenerateForm } from "./components/GenerateForm.tsx";
import { ProgressView } from "./components/ProgressView.tsx";
import { SceneList } from "./components/SceneList.tsx";
import { SceneDetail } from "./components/SceneDetail.tsx";
import { Dashboard } from "./components/Dashboard.tsx";
import { ProductionBibleLibrary } from "./components/ProductionBibleLibrary.tsx";
import { ProductionBibleCreator } from "./components/ProductionBibleCreator.tsx";
import { SettingsPage } from "./components/SettingsPage.tsx";
import { ProductionList } from "./components/ProductionList.tsx";
import { ProductionDetail } from "./components/ProductionDetail.tsx";
import { AssetLibrary } from "./components/AssetLibrary.tsx";
import { ActorLibraryDetail } from "./components/ActorLibraryDetail.tsx";
import { createDraftScene } from "./api/client.ts";
import { ShowCostProvider } from "./hooks/useShowCost.tsx";

function App() {
  const [, navigate] = useLocation();

  async function handleNewDraft() {
    try {
      const res = await createDraftScene();
      navigate(`/scenes/${res.scene_id}`);
    } catch {
      navigate("/generate");
    }
  }

  return (
    <ShowCostProvider>
      <Layout>
        <Switch>
          <Route path="/">
            <SceneList
              onSelectScene={(id) => navigate(`/scenes/${id}`)}
              onNewScene={() => navigate("/generate")}
              onNewDraft={handleNewDraft}
            />
          </Route>
          <Route path="/generate">
            <GenerateForm
              onGenerated={(id) => navigate(`/scenes/${id}/progress`)}
            />
          </Route>
          <Route path="/scenes/:id/progress">
            {(params) => (
              <ProgressView
                sceneId={params.id}
                onViewDetail={(id) => navigate(`/scenes/${id}`)}
              />
            )}
          </Route>
          <Route path="/scenes/:id">
            {(params) => (
              <SceneDetail
                sceneId={params.id}
                onViewProgress={(id) => navigate(`/scenes/${id}/progress`)}
                onForked={(newId) => navigate(`/scenes/${newId}/progress`)}
                onViewScene={(id) => navigate(`/scenes/${id}`)}
                onViewManifest={(manifestId) =>
                  navigate(`/production-bibles/${manifestId}/edit`)
                }
              />
            )}
          </Route>
          <Route path="/productions">
            <ProductionList
              onSelectProduction={(id) => navigate(`/productions/${id}`)}
            />
          </Route>
          <Route path="/productions/:id">
            {(params) => (
              <ProductionDetail
                productionId={params.id}
                onViewScene={(id) => navigate(`/scenes/${id}`)}
              />
            )}
          </Route>
          <Route path="/dashboard">
            <Dashboard />
          </Route>
          <Route path="/production-bibles/new">
            <ProductionBibleCreator
              productionBibleId={null}
              onSaved={() => navigate("/production-bibles")}
              onCancel={() => navigate("/production-bibles")}
            />
          </Route>
          <Route path="/production-bibles/:id/edit">
            {(params) => (
              <ProductionBibleCreator
                productionBibleId={params.id}
                onSaved={() => navigate("/production-bibles")}
                onCancel={() => navigate("/production-bibles")}
              />
            )}
          </Route>
          <Route path="/production-bibles">
            <ProductionBibleLibrary
              onCreateNew={() => navigate("/production-bibles/new")}
              onEditManifest={(id) => navigate(`/production-bibles/${id}/edit`)}
              onViewManifest={(id) => navigate(`/production-bibles/${id}/edit`)}
            />
          </Route>
          <Route path="/asset-library/actors/:id">
            {(params) => <ActorLibraryDetail actorId={params.id} />}
          </Route>
          <Route path="/asset-library">
            <AssetLibrary />
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

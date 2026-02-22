import { createContext, useContext, useState, useEffect } from "react";
import { getEnabledModels } from "../api/client";

const ShowCostContext = createContext<{
  showCost: boolean;
  setShowCost: (v: boolean) => void;
}>({ showCost: true, setShowCost: () => {} });

export function ShowCostProvider({ children }: { children: React.ReactNode }) {
  const [showCost, setShowCost] = useState(true);
  useEffect(() => {
    getEnabledModels().then((r) => setShowCost(r.show_cost));
  }, []);
  return (
    <ShowCostContext.Provider value={{ showCost, setShowCost }}>
      {children}
    </ShowCostContext.Provider>
  );
}

export function useShowCost() {
  return useContext(ShowCostContext);
}

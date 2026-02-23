import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { SceneEditorCard } from "./SceneEditorCard.tsx";
import type { ComponentProps } from "react";

type SceneEditorCardProps = ComponentProps<typeof SceneEditorCard>;

interface SortableSceneCardProps extends SceneEditorCardProps {
  /** Used as the sortable item id — typically scene.scene_index */
  id: number;
}

export function SortableSceneCard({ id, ...rest }: SortableSceneCardProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id });

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : undefined,
    position: "relative",
    zIndex: isDragging ? 10 : undefined,
  };

  return (
    <div ref={setNodeRef} style={style}>
      <SceneEditorCard
        {...rest}
        dragHandleListeners={listeners}
        dragHandleAttributes={attributes}
      />
    </div>
  );
}

import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { ShotEditorCard } from "./ShotEditorCard.tsx";
import type { ComponentProps } from "react";

type ShotEditorCardProps = ComponentProps<typeof ShotEditorCard>;

interface SortableShotCardProps extends ShotEditorCardProps {
  /** Used as the sortable item id — typically shot.shot_index */
  id: number;
}

export function SortableShotCard({ id, ...rest }: SortableShotCardProps) {
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
      <ShotEditorCard
        {...rest}
        dragHandleListeners={listeners}
        dragHandleAttributes={attributes as unknown as Record<string, unknown>}
      />
    </div>
  );
}

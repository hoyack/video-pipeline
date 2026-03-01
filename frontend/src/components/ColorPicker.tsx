const PRESET_COLORS = [
  "#3B82F6", // blue
  "#10B981", // emerald
  "#F59E0B", // amber
  "#EF4444", // red
  "#8B5CF6", // violet
  "#EC4899", // pink
  "#06B6D4", // cyan
  "#6B7280", // gray
];

interface ColorPickerProps {
  selectedColor: string | null;
  onColorSelect: (color: string) => void;
}

export function ColorPicker({ selectedColor, onColorSelect }: ColorPickerProps) {
  return (
    <div className="flex flex-wrap gap-1.5 p-1">
      {PRESET_COLORS.map((color) => (
        <button
          key={color}
          onClick={() => onColorSelect(color)}
          title={color}
          style={{ backgroundColor: color }}
          className={`w-6 h-6 rounded-full flex-shrink-0 transition-transform hover:scale-110 ${
            selectedColor === color
              ? "ring-2 ring-offset-2 ring-offset-gray-800 ring-white scale-110"
              : ""
          }`}
        />
      ))}
    </div>
  );
}

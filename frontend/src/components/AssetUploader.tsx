import { useState, useRef } from "react";

interface AssetUploaderProps {
  onFilesSelected: (files: File[]) => void;
  onVideoSelected?: (file: File) => void;
  disabled?: boolean;
}

const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB
const MAX_VIDEO_SIZE = 200 * 1024 * 1024; // 200MB
const ACCEPTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];
const ACCEPTED_VIDEO_TYPES = ["video/mp4", "video/quicktime", "video/webm"];

export function AssetUploader({ onFilesSelected, onVideoSelected, disabled = false }: AssetUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const filterFiles = (fileList: FileList | null): File[] => {
    if (!fileList) return [];

    const validFiles: File[] = [];
    const errors: string[] = [];

    Array.from(fileList).forEach((file) => {
      // Check for video files first
      if (ACCEPTED_VIDEO_TYPES.includes(file.type)) {
        if (file.size > MAX_VIDEO_SIZE) {
          errors.push(`${file.name} exceeds 200MB video limit`);
          return;
        }
        if (onVideoSelected) {
          onVideoSelected(file);
        }
        return;
      }

      if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
        // Silently filter out unsupported files
        return;
      }
      if (file.size > MAX_IMAGE_SIZE) {
        errors.push(`${file.name} exceeds 10MB limit`);
        return;
      }
      validFiles.push(file);
    });

    if (errors.length > 0) {
      setError(errors.join(", "));
      setTimeout(() => setError(null), 5000);
    }

    return validFiles;
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    if (disabled) return;
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    if (disabled) return;
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    if (disabled) return;
    e.preventDefault();
    setIsDragging(false);

    const files = filterFiles(e.dataTransfer.files);
    if (files.length > 0) {
      onFilesSelected(files);
    }
  };

  const handleClick = () => {
    if (disabled) return;
    inputRef.current?.click();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = filterFiles(e.target.files);
    if (files.length > 0) {
      onFilesSelected(files);
    }
    // Reset input value to allow re-selecting the same file
    e.target.value = "";
  };

  return (
    <div>
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className={`
          rounded-lg border-2 border-dashed p-8 text-center transition-colors cursor-pointer
          ${isDragging ? "border-blue-500 bg-blue-500/5" : "border-gray-700"}
          ${disabled ? "opacity-50 cursor-not-allowed" : "hover:border-gray-600"}
        `}
      >
        <div className="flex flex-col items-center gap-2">
          <div className="text-4xl text-gray-500">↑</div>
          <p className="text-gray-300">Drag and drop images or video here</p>
          <p className="text-sm text-gray-500">or click to browse</p>
          <p className="text-xs text-gray-600 mt-2">
            PNG, JPEG, WebP images (up to 10MB) or MP4, MOV, WebM video (up to 200MB)
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="image/png,image/jpeg,image/webp,video/mp4,video/quicktime,video/webm"
          onChange={handleInputChange}
          className="hidden"
          disabled={disabled}
        />
      </div>
      {error && (
        <div className="mt-2 rounded-md bg-red-900/30 border border-red-700 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}
    </div>
  );
}

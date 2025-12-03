'use client';

/**
 * DMPlaceholder Component
 * Invisible placeholder for the DM talking head animation.
 * This reserves space in the layout for future implementation.
 */

interface DMPlaceholderProps {
  isVisible?: boolean;
}

export default function DMPlaceholder({ isVisible = false }: DMPlaceholderProps) {
  // Currently invisible - placeholder for future DM talking head
  if (!isVisible) {
    return (
      <div
        className="w-64 h-24 pointer-events-none"
        aria-hidden="true"
      />
    );
  }

  // Future: visible state with DM animation
  return (
    <div className="w-64 h-24 bg-gray-900/80 backdrop-blur-sm border border-gray-700 rounded-lg flex items-center justify-center">
      <div className="text-center">
        <div className="text-2xl mb-1">DM</div>
        <div className="text-xs text-gray-400">Speaking...</div>
      </div>
    </div>
  );
}

'use client';

/**
 * TimeIndicator Component
 * Circular day/night indicator showing current game time.
 */

import { useGameStore, selectIsNight } from '@/stores/gameStore';

export default function TimeIndicator() {
  const gameTime = useGameStore((state) => state.gameTime);
  const isNight = useGameStore(selectIsNight);

  const { hour, minute } = gameTime;

  // Format time as HH:MM
  const timeString = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;

  // Calculate sun/moon position (0 = bottom, 0.5 = top)
  // Sunrise at 6, sunset at 20
  let position: number;
  if (isNight) {
    // Night: moon rises at 20, peaks at 1, sets at 6
    if (hour >= 20) {
      position = (hour - 20) / 10; // 20-24 -> 0-0.4
    } else {
      position = (hour + 4) / 10; // 0-6 -> 0.4-1.0
    }
  } else {
    // Day: sun rises at 6, peaks at 13, sets at 20
    position = (hour - 6) / 14; // 6-20 -> 0-1
  }

  // Convert to arc position (in degrees, 180 = top)
  const arcDegrees = position * 180;

  // Background gradient based on time of day
  const getBackgroundGradient = () => {
    if (hour >= 5 && hour < 7) {
      // Dawn
      return 'from-orange-900 via-pink-800 to-purple-900';
    } else if (hour >= 7 && hour < 18) {
      // Day
      return 'from-sky-400 via-sky-500 to-blue-600';
    } else if (hour >= 18 && hour < 20) {
      // Dusk
      return 'from-orange-600 via-red-700 to-purple-900';
    } else {
      // Night
      return 'from-slate-900 via-indigo-950 to-slate-900';
    }
  };

  return (
    <div
      className="relative w-14 h-14 rounded-full overflow-hidden shadow-lg border-2 border-gray-700 cursor-default group"
      title={`Day ${gameTime.day}, ${timeString}`}
    >
      {/* Sky gradient background */}
      <div className={`absolute inset-0 bg-gradient-to-b ${getBackgroundGradient()} transition-colors duration-1000`} />

      {/* Stars (visible at night) */}
      {isNight && (
        <div className="absolute inset-0">
          <div className="absolute top-2 left-3 w-0.5 h-0.5 bg-white rounded-full opacity-80" />
          <div className="absolute top-4 right-4 w-1 h-1 bg-white rounded-full opacity-60" />
          <div className="absolute top-6 left-5 w-0.5 h-0.5 bg-white rounded-full opacity-70" />
          <div className="absolute bottom-4 right-3 w-0.5 h-0.5 bg-white rounded-full opacity-50" />
        </div>
      )}

      {/* Sun/Moon arc path */}
      <div
        className="absolute w-5 h-5 transition-all duration-300"
        style={{
          left: '50%',
          bottom: '10%',
          transform: `translateX(-50%) rotate(${arcDegrees}deg) translateY(-18px) rotate(-${arcDegrees}deg)`,
        }}
      >
        {isNight ? (
          // Moon
          <div className="w-5 h-5 rounded-full bg-gray-200 shadow-lg relative">
            <div className="absolute top-0.5 left-1 w-1 h-1 rounded-full bg-gray-400 opacity-50" />
            <div className="absolute top-2 right-1 w-0.5 h-0.5 rounded-full bg-gray-400 opacity-40" />
          </div>
        ) : (
          // Sun
          <div className="w-5 h-5 rounded-full bg-yellow-300 shadow-lg shadow-yellow-400/50" />
        )}
      </div>

      {/* Time overlay on hover */}
      <div className="absolute inset-0 flex items-center justify-center bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity">
        <span className="text-white text-xs font-mono">{timeString}</span>
      </div>
    </div>
  );
}

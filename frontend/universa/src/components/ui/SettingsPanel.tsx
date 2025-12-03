'use client';

/**
 * SettingsPanel Component
 * Modal panel for game settings.
 */

import { useState } from 'react';

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

interface SettingsState {
  masterVolume: number;
  musicVolume: number;
  sfxVolume: number;
  showTooltips: boolean;
  autoSave: boolean;
  showDamageNumbers: boolean;
}

export default function SettingsPanel({ isOpen, onClose }: SettingsPanelProps) {
  const [settings, setSettings] = useState<SettingsState>({
    masterVolume: 80,
    musicVolume: 60,
    sfxVolume: 70,
    showTooltips: true,
    autoSave: true,
    showDamageNumbers: true,
  });

  if (!isOpen) return null;

  const updateSetting = <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-lg max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span></span> Settings
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            x
          </button>
        </div>

        <div className="p-4 overflow-y-auto max-h-[calc(80vh-120px)]">
          {/* Audio Settings */}
          <section className="mb-6">
            <h3 className="text-lg font-semibold text-amber-400 mb-4">Audio</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-sm text-gray-300">Master Volume</label>
                  <span className="text-sm text-gray-500">{settings.masterVolume}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={settings.masterVolume}
                  onChange={(e) => updateSetting('masterVolume', Number(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-sm text-gray-300">Music Volume</label>
                  <span className="text-sm text-gray-500">{settings.musicVolume}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={settings.musicVolume}
                  onChange={(e) => updateSetting('musicVolume', Number(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-sm text-gray-300">Sound Effects</label>
                  <span className="text-sm text-gray-500">{settings.sfxVolume}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={settings.sfxVolume}
                  onChange={(e) => updateSetting('sfxVolume', Number(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
              </div>
            </div>
          </section>

          {/* Gameplay Settings */}
          <section className="mb-6">
            <h3 className="text-lg font-semibold text-amber-400 mb-4">Gameplay</h3>
            <div className="space-y-3">
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-sm text-gray-300">Show Tooltips</span>
                <input
                  type="checkbox"
                  checked={settings.showTooltips}
                  onChange={(e) => updateSetting('showTooltips', e.target.checked)}
                  className="w-5 h-5 rounded bg-gray-700 border-gray-600 text-amber-500 focus:ring-amber-500 focus:ring-offset-gray-900"
                />
              </label>
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-sm text-gray-300">Auto-Save</span>
                <input
                  type="checkbox"
                  checked={settings.autoSave}
                  onChange={(e) => updateSetting('autoSave', e.target.checked)}
                  className="w-5 h-5 rounded bg-gray-700 border-gray-600 text-amber-500 focus:ring-amber-500 focus:ring-offset-gray-900"
                />
              </label>
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-sm text-gray-300">Show Damage Numbers</span>
                <input
                  type="checkbox"
                  checked={settings.showDamageNumbers}
                  onChange={(e) => updateSetting('showDamageNumbers', e.target.checked)}
                  className="w-5 h-5 rounded bg-gray-700 border-gray-600 text-amber-500 focus:ring-amber-500 focus:ring-offset-gray-900"
                />
              </label>
            </div>
          </section>

          {/* Controls Reference */}
          <section>
            <h3 className="text-lg font-semibold text-amber-400 mb-4">Controls</h3>
            <div className="bg-gray-800 rounded-lg p-3 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <div className="text-gray-400">Movement</div>
                <div className="text-gray-200">WASD / Arrow Keys</div>
                <div className="text-gray-400">Inventory</div>
                <div className="text-gray-200">I</div>
                <div className="text-gray-400">Journal</div>
                <div className="text-gray-200">J</div>
                <div className="text-gray-400">Quests</div>
                <div className="text-gray-200">Q</div>
                <div className="text-gray-400">Character</div>
                <div className="text-gray-200">C</div>
                <div className="text-gray-400">Close Panel</div>
                <div className="text-gray-200">Escape</div>
              </div>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

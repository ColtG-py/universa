'use client';

/**
 * ChatPanel Component
 * Handles chat messages between players, agents, and the DM.
 */

import { useState, useRef, useEffect } from 'react';
import { useGameStore } from '@/stores/gameStore';
import type { ChatMessage } from '@/types/game';

interface ChatPanelProps {
  onSendMessage: (content: string, channel: ChatMessage['channel']) => void;
  isDialogueMode?: boolean;
}

export default function ChatPanel({ onSendMessage, isDialogueMode = false }: ChatPanelProps) {
  const messages = useGameStore((state) => state.messages);
  const [inputValue, setInputValue] = useState('');
  const [activeChannel, setActiveChannel] = useState<ChatMessage['channel']>('party');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onSendMessage(inputValue.trim(), activeChannel);
      setInputValue('');
    }
  };

  const getMessageColor = (message: ChatMessage): string => {
    switch (message.senderType) {
      case 'player':
        return 'text-blue-400';
      case 'agent':
        return 'text-green-400';
      case 'dm':
        return 'text-purple-400';
      case 'system':
        return 'text-gray-400';
      default:
        return 'text-white';
    }
  };

  const getChannelBadgeColor = (channel: ChatMessage['channel']): string => {
    switch (channel) {
      case 'global':
        return 'bg-gray-600';
      case 'party':
        return 'bg-blue-600';
      case 'whisper':
        return 'bg-purple-600';
      case 'narration':
        return 'bg-amber-600';
      default:
        return 'bg-gray-600';
    }
  };

  const formatTimestamp = (timestamp: number): string => {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="flex flex-col h-full bg-gray-900 border border-gray-700 rounded-lg">
      {/* Channel tabs */}
      <div className="flex border-b border-gray-700">
        {(['party', 'global', 'whisper'] as const).map((channel) => (
          <button
            key={channel}
            onClick={() => setActiveChannel(channel)}
            className={`px-4 py-2 text-sm font-medium capitalize transition-colors ${
              activeChannel === channel
                ? 'text-white bg-gray-800 border-b-2 border-blue-500'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            {channel}
          </button>
        ))}
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {messages
          .filter(
            (msg) =>
              msg.channel === activeChannel ||
              msg.channel === 'narration' ||
              msg.senderType === 'system'
          )
          .map((message) => (
            <div
              key={message.id}
              className={`${
                message.channel === 'narration'
                  ? 'bg-amber-900/30 p-2 rounded italic'
                  : ''
              }`}
            >
              <div className="flex items-start gap-2">
                {/* Channel badge for mixed view */}
                {message.channel !== activeChannel && (
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${getChannelBadgeColor(
                      message.channel
                    )}`}
                  >
                    {message.channel}
                  </span>
                )}

                {/* Timestamp */}
                <span className="text-xs text-gray-500">
                  {formatTimestamp(message.timestamp)}
                </span>

                {/* Sender */}
                <span className={`font-semibold ${getMessageColor(message)}`}>
                  {message.senderName}:
                </span>

                {/* Content */}
                <span className="text-gray-200 break-words">{message.content}</span>
              </div>
            </div>
          ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <form onSubmit={handleSubmit} className="p-3 border-t border-gray-700">
        {isDialogueMode && (
          <div className="mb-2 px-2 py-1 bg-purple-900/30 border border-purple-700 rounded text-sm text-purple-300">
            In conversation - type to speak to the agent
          </div>
        )}
        <div className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={isDialogueMode ? 'Say something...' : `Message ${activeChannel}...`}
            className={`flex-1 px-3 py-2 bg-gray-800 border rounded text-white placeholder-gray-500 focus:outline-none transition-colors ${
              isDialogueMode
                ? 'border-purple-600 focus:border-purple-400'
                : 'border-gray-600 focus:border-blue-500'
            }`}
          />
          <button
            type="submit"
            className={`px-4 py-2 text-white rounded transition-colors ${
              isDialogueMode
                ? 'bg-purple-600 hover:bg-purple-700'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isDialogueMode ? 'Speak' : 'Send'}
          </button>
        </div>
      </form>
    </div>
  );
}

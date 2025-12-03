/**
 * Dialogue Hooks
 * React hooks for managing conversations with agents.
 */

'use client';

import { useState, useCallback } from 'react';
import { api, Conversation, DialogueTurn } from '@/lib/api';
import { useGameStore } from '@/stores/gameStore';

// =============================================================================
// useDialogue - Manage a conversation with an agent
// =============================================================================

export function useDialogue() {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isWaitingForResponse, setIsWaitingForResponse] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setInputMode = useGameStore(state => state.setInputMode);
  const addMessage = useGameStore(state => state.addMessage);

  const startConversation = useCallback(async (params: {
    sessionId: string;
    playerId: string;
    playerName: string;
    agentId: string;
    agentName: string;
    playerX: number;
    playerY: number;
    openingMessage?: string;
  }) => {
    setIsLoading(true);
    setError(null);
    try {
      // Start interaction to promote agent tier
      await api.startInteraction(params.sessionId, params.agentId);

      const conv = await api.startConversation(params);
      setConversation(conv);
      setInputMode('dialogue');

      // Add initial turns to chat
      for (const turn of conv.turns) {
        addMessage({
          id: turn.id,
          senderId: turn.speaker_id,
          senderName: turn.speaker_name,
          senderType: turn.speaker_type,
          content: turn.content,
          timestamp: new Date(turn.timestamp).getTime(),
          channel: 'whisper',
        });
      }

      return conv;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start conversation';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [setInputMode, addMessage]);

  const sendMessage = useCallback(async (
    message: string,
    speakerId: string,
    speakerName: string
  ) => {
    if (!conversation) {
      setError('No active conversation');
      return null;
    }

    setIsLoading(true);
    setError(null);
    try {
      // Add player message
      const playerTurn = await api.continueConversation({
        conversationId: conversation.conversation_id,
        message,
        speakerId,
        speakerName,
      });

      // Update conversation with player message
      setConversation(prev => prev ? {
        ...prev,
        turns: [...prev.turns, playerTurn],
      } : null);

      // Add to chat
      addMessage({
        id: playerTurn.id,
        senderId: playerTurn.speaker_id,
        senderName: playerTurn.speaker_name,
        senderType: 'player',
        content: playerTurn.content,
        timestamp: new Date(playerTurn.timestamp).getTime(),
        channel: 'whisper',
      });

      return playerTurn;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send message';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [conversation, addMessage]);

  const getAgentResponse = useCallback(async (
    agentId: string,
    agentName: string
  ) => {
    if (!conversation) {
      setError('No active conversation');
      return null;
    }

    setIsWaitingForResponse(true);
    setError(null);
    try {
      const response = await api.getAgentResponse({
        conversationId: conversation.conversation_id,
        agentId,
        agentName,
      });

      // Update conversation with agent response
      setConversation(prev => prev ? {
        ...prev,
        turns: [...prev.turns, response.turn],
      } : null);

      // Add to chat
      addMessage({
        id: response.turn.id,
        senderId: response.turn.speaker_id,
        senderName: response.turn.speaker_name,
        senderType: 'agent',
        content: response.turn.content,
        timestamp: new Date(response.turn.timestamp).getTime(),
        channel: 'whisper',
      });

      // Auto-end if agent wants to end
      if (response.should_end) {
        await endConversation();
      }

      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to get response';
      setError(message);
      throw err;
    } finally {
      setIsWaitingForResponse(false);
    }
  }, [conversation, addMessage]);

  const endConversation = useCallback(async (reason = 'player_ended') => {
    if (!conversation) return null;

    setIsLoading(true);
    try {
      const ended = await api.endConversation(conversation.conversation_id, reason);

      // End interaction to return agent to normal tier
      const agentParticipant = conversation.participants.find(p => p.type === 'agent');
      if (agentParticipant) {
        // Get session ID from conversation
        await api.endInteraction(conversation.session_id, agentParticipant.id);
      }

      setConversation(null);
      setInputMode('explore');

      // Add system message
      addMessage({
        id: `end-${conversation.conversation_id}`,
        senderId: 'system',
        senderName: 'System',
        senderType: 'system',
        content: 'Conversation ended.',
        timestamp: Date.now(),
        channel: 'narration',
      });

      return ended;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to end conversation';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [conversation, setInputMode, addMessage]);

  return {
    conversation,
    isLoading,
    isWaitingForResponse,
    error,
    startConversation,
    sendMessage,
    getAgentResponse,
    endConversation,
    isActive: conversation !== null && conversation.state === 'active',
  };
}

// =============================================================================
// useQuickTalk - For single-exchange conversations
// =============================================================================

export function useQuickTalk() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResponse, setLastResponse] = useState<DialogueTurn | null>(null);

  const addMessage = useGameStore(state => state.addMessage);

  const quickTalk = useCallback(async (params: {
    sessionId: string;
    playerId: string;
    playerName: string;
    agentId: string;
    agentName: string;
    playerX: number;
    playerY: number;
    message: string;
  }) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await api.quickTalk({
        ...params,
        openingMessage: params.message,
      });

      // Add player message
      addMessage({
        id: `qt-player-${Date.now()}`,
        senderId: params.playerId,
        senderName: params.playerName,
        senderType: 'player',
        content: params.message,
        timestamp: Date.now(),
        channel: 'whisper',
      });

      // Add agent response
      if (result.agent_response) {
        setLastResponse(result.agent_response);
        addMessage({
          id: result.agent_response.id,
          senderId: result.agent_response.speaker_id,
          senderName: result.agent_response.speaker_name,
          senderType: 'agent',
          content: result.agent_response.content,
          timestamp: new Date(result.agent_response.timestamp).getTime(),
          channel: 'whisper',
        });
      }

      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to talk';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [addMessage]);

  return {
    quickTalk,
    isLoading,
    error,
    lastResponse,
  };
}

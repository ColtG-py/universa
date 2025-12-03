/**
 * Database Types
 * Placeholder for Supabase-generated types.
 * Run `npx supabase gen types typescript` to generate actual types.
 */

export interface Database {
  public: {
    Tables: {
      worlds: {
        Row: {
          id: string;
          name: string;
          seed: number;
          width: number;
          height: number;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          name: string;
          seed: number;
          width: number;
          height: number;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          name?: string;
          seed?: number;
          width?: number;
          height?: number;
          updated_at?: string;
        };
      };
      agents: {
        Row: {
          id: string;
          world_id: string;
          name: string;
          x: number;
          y: number;
          current_action: string | null;
          status: 'idle' | 'active' | 'sleeping' | 'dead';
          traits: Record<string, unknown>;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          world_id: string;
          name: string;
          x: number;
          y: number;
          current_action?: string | null;
          status?: 'idle' | 'active' | 'sleeping' | 'dead';
          traits?: Record<string, unknown>;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          world_id?: string;
          name?: string;
          x?: number;
          y?: number;
          current_action?: string | null;
          status?: 'idle' | 'active' | 'sleeping' | 'dead';
          traits?: Record<string, unknown>;
          updated_at?: string;
        };
      };
      world_events: {
        Row: {
          id: string;
          world_id: string;
          event_type: string;
          description: string;
          x: number;
          y: number;
          metadata: Record<string, unknown>;
          timestamp: string;
        };
        Insert: {
          id?: string;
          world_id: string;
          event_type: string;
          description: string;
          x: number;
          y: number;
          metadata?: Record<string, unknown>;
          timestamp?: string;
        };
        Update: {
          id?: string;
          world_id?: string;
          event_type?: string;
          description?: string;
          x?: number;
          y?: number;
          metadata?: Record<string, unknown>;
          timestamp?: string;
        };
      };
      game_sessions: {
        Row: {
          id: string;
          world_id: string;
          name: string;
          dm_agent_id: string | null;
          status: 'lobby' | 'active' | 'paused' | 'ended';
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          world_id: string;
          name: string;
          dm_agent_id?: string | null;
          status?: 'lobby' | 'active' | 'paused' | 'ended';
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          world_id?: string;
          name?: string;
          dm_agent_id?: string | null;
          status?: 'lobby' | 'active' | 'paused' | 'ended';
          updated_at?: string;
        };
      };
      players: {
        Row: {
          id: string;
          session_id: string;
          user_id: string;
          character_name: string;
          x: number;
          y: number;
          is_agent: boolean;
          agent_id: string | null;
          created_at: string;
        };
        Insert: {
          id?: string;
          session_id: string;
          user_id: string;
          character_name: string;
          x: number;
          y: number;
          is_agent?: boolean;
          agent_id?: string | null;
          created_at?: string;
        };
        Update: {
          id?: string;
          session_id?: string;
          user_id?: string;
          character_name?: string;
          x?: number;
          y?: number;
          is_agent?: boolean;
          agent_id?: string | null;
        };
      };
    };
    Views: Record<string, never>;
    Functions: Record<string, never>;
    Enums: Record<string, never>;
  };
}

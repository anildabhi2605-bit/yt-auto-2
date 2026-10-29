import { GoogleGenAI } from '@google/genai';

// Initialize Gemini client with server-side API key
const apiKey = process.env.GEMINI_API_KEY || '';
const ai = new GoogleGenAI({ apiKey });

export interface GenerationOptions {
  model?: string;
  temperature?: number;
  systemInstruction?: string;
}

export interface GenerationResult {
  text: string;
  model: string;
  latency_ms: number;
  usage?: {
    promptTokens?: number;
    responseTokens?: number;
    totalTokens?: number;
  };
}

/**
 * Execute AI generation via Google Gemini SDK (@google/genai)
 * Strictly enforces server-side vaulting and error classification per 07_AI_PROVIDER_MODEL_MANAGER.md
 */
export async function executeGeminiGeneration(
  prompt: string,
  options: GenerationOptions = {}
): Promise<GenerationResult> {
  const model = options.model || 'gemini-3.8-flash';
  const startTime = Date.now();

  if (!apiKey) {
    // If API key is not set, provide informative error without crashing
    throw new Error('GEMINI_API_KEY is not configured on the server. Please set GEMINI_API_KEY in the environment or server configuration.');
  }

  let attempts = 0;
  const maxAttempts = 2;

  while (attempts < maxAttempts) {
    attempts++;
    try {
      const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          temperature: options.temperature ?? 0.7,
          systemInstruction: options.systemInstruction
        }
      });

      const text = response.text || '';
      const usageMetadata = response.usageMetadata;
      const latency_ms = Date.now() - startTime;

      return {
        text,
        model,
        latency_ms,
        usage: {
          promptTokens: usageMetadata?.promptTokenCount,
          responseTokens: usageMetadata?.candidatesTokenCount,
          totalTokens: usageMetadata?.totalTokenCount
        }
      };
    } catch (error: any) {
      console.error(`[Gemini Service] Attempt ${attempts} failed:`, error.message);
      
      // Classify error without leaking secret keys
      if (error.status === 401 || error.message?.includes('API_KEY_INVALID') || error.message?.includes('401')) {
        throw new Error('Google Gemini API Key is invalid or expired. Please check server vault.');
      }
      if (error.status === 403 || error.message?.includes('PERMISSION_DENIED') || error.message?.includes('403')) {
        throw new Error('Google Gemini API permission error: Free tier is unavailable for this project or billing upgrade required (403 PERMISSION_DENIED).');
      }
      if (error.status === 429 || error.message?.includes('RESOURCE_EXHAUSTED')) {
        throw new Error('Google Gemini rate limit reached (429). Bounded retry or user permission required for alternative model.');
      }
      
      // If 503 (high demand) or 5xx, retry if attempts remain
      if ((error.status === 503 || error.message?.includes('503') || error.message?.includes('UNAVAILABLE')) && attempts < maxAttempts) {
        await new Promise(res => setTimeout(res, 2000));
        continue;
      }

      throw new Error(`Google Gemini generation failed: ${error.message || 'Unknown error'}`);
    }
  }

  throw new Error('Google Gemini service unavailable after retries. Please retry in a few moments.');
}

export interface ModelCatalogEntry {
  id: string;
  name: string;
  provider: 'Google Gemini';
  tier: 'Fast' | 'Reasoning';
  status: 'Supported' | 'Candidate';
  capabilities: {
    multimodal: boolean;
    structured_json: boolean;
    max_output_tokens: number;
    recommended_use: string;
  };
}

export const MODEL_CATALOG: ModelCatalogEntry[] = [
  {
    id: 'gemini-3.8-flash',
    name: 'Gemini 3.8 Flash',
    provider: 'Google Gemini',
    tier: 'Fast',
    status: 'Supported',
    capabilities: {
      multimodal: true,
      structured_json: true,
      max_output_tokens: 8192,
      recommended_use: 'Default production model for Story-to-Script, Video Prompts, Thumbnails & SEO'
    }
  },
  {
    id: 'gemini-4-argon',
    name: 'Gemini 4 Argon',
    provider: 'Google Gemini',
    tier: 'Reasoning',
    status: 'Supported',
    capabilities: {
      multimodal: true,
      structured_json: true,
      max_output_tokens: 16384,
      recommended_use: 'High-reasoning alternative model for intricate multi-narrative analysis'
    }
  }
];

/**
 * Live connection diagnostic test for Google Gemini
 */
export async function testGeminiConnection(): Promise<{ success: boolean; latency_ms: number; model: string; error?: string }> {
  const startTime = Date.now();
  if (!apiKey) {
    return {
      success: false,
      latency_ms: 0,
      model: 'gemini-3.8-flash',
      error: 'GEMINI_API_KEY is not configured on the server environment. Please set GEMINI_API_KEY in your server configuration.'
    };
  }

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: 'Ping',
      config: {
        maxOutputTokens: 5
      }
    });

    const latency_ms = Date.now() - startTime;
    return {
      success: Boolean(response && response.text),
      latency_ms,
      model: 'gemini-3.8-flash'
    };
  } catch (err: any) {
    return {
      success: false,
      latency_ms: Date.now() - startTime,
      model: 'gemini-3.8-flash',
      error: err.message || 'Connection test failed'
    };
  }
}

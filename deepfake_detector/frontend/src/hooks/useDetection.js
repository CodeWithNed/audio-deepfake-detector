import { useState } from 'react';
import { apiClient } from '../api/client';

export function useDetection() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const detectAudio = async (audioFile) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiClient.detectAudio(audioFile);
      setResult(response);
      return response;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message || 'Detection failed';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
    setIsLoading(false);
  };

  return {
    isLoading,
    result,
    error,
    detectAudio,
    reset,
  };
}

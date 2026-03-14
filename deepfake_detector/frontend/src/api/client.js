import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 120000, // 2 minutes for audio processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// API functions
export const apiClient = {
  // Health check
  async health() {
    const response = await api.get('/health');
    return response.data;
  },

  // Detect deepfake audio
  async detectAudio(audioFile) {
    const formData = new FormData();
    formData.append('file', audioFile);

    const response = await api.post('/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  // Get detection history
  async getHistory(limit = 50) {
    const response = await api.get(`/history?limit=${limit}`);
    return response.data;
  },

  // Clear history
  async clearHistory() {
    const response = await api.delete('/history');
    return response.data;
  },

  // Delete specific history item
  async deleteHistoryItem(id) {
    const response = await api.delete(`/history/${id}`);
    return response.data;
  },
};

export default apiClient;

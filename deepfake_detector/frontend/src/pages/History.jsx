import { useState, useEffect } from 'react';
import { HistoryTable } from '../components/HistoryTable';
import { apiClient } from '../api/client';

export function History() {
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchHistory = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.getHistory();
      setHistory(response.items || []);
    } catch (err) {
      setError(err.message || 'Failed to load history');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = async () => {
    if (!confirm('Are you sure you want to clear all history?')) {
      return;
    }

    try {
      await apiClient.clearHistory();
      setHistory([]);
    } catch (err) {
      alert('Failed to clear history: ' + err.message);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Detection History</h1>
        <p style={styles.subtitle}>View past audio analysis results</p>
      </div>

      {isLoading && <div style={styles.loading}>Loading history...</div>}

      {error && (
        <div style={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {!isLoading && !error && (
        <HistoryTable history={history} onClear={handleClear} />
      )}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '40px 20px',
  },
  header: {
    textAlign: 'center',
    marginBottom: '40px',
  },
  title: {
    fontSize: '36px',
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: '12px',
  },
  subtitle: {
    fontSize: '16px',
    color: '#6b7280',
  },
  loading: {
    textAlign: 'center',
    padding: '40px',
    color: '#6b7280',
  },
  error: {
    backgroundColor: '#fee2e2',
    border: '1px solid #f87171',
    color: '#991b1b',
    padding: '16px',
    borderRadius: '8px',
    marginTop: '20px',
    textAlign: 'center',
  },
};

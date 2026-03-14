import { useState } from 'react';
import { AudioUploader } from '../components/AudioUploader';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ResultCard } from '../components/ResultCard';
import { useDetection } from '../hooks/useDetection';

export function Home() {
  const { isLoading, result, error, detectAudio, reset } = useDetection();
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileSelect = async (file) => {
    setSelectedFile(file);
    reset();

    try {
      await detectAudio(file);
    } catch (err) {
      console.error('Detection failed:', err);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>🎵 Deepfake Audio Detector</h1>
        <p style={styles.subtitle}>
          Multimodal detection using acoustic (WavLM) and linguistic (RoBERTa) analysis
        </p>
      </div>

      <AudioUploader onFileSelect={handleFileSelect} isLoading={isLoading} />

      {isLoading && <LoadingSpinner />}

      {error && (
        <div style={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && !isLoading && <ResultCard result={result} />}

      {result && (
        <button style={styles.resetButton} onClick={reset}>
          Analyze Another File
        </button>
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
  error: {
    backgroundColor: '#fee2e2',
    border: '1px solid #f87171',
    color: '#991b1b',
    padding: '16px',
    borderRadius: '8px',
    marginTop: '20px',
    textAlign: 'center',
  },
  resetButton: {
    display: 'block',
    margin: '30px auto',
    padding: '12px 24px',
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '16px',
    fontWeight: '500',
    cursor: 'pointer',
  },
};

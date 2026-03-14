export function TranscriptPanel({ transcript, language, confidence }) {
  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Transcript</h3>
      <div style={styles.metadata}>
        <span style={styles.badge}>Language: {language.toUpperCase()}</span>
        <span style={styles.badge}>
          ASR Confidence: {Math.round(confidence * 100)}%
        </span>
      </div>
      <div style={styles.transcript}>
        {transcript || 'No transcript available'}
      </div>
    </div>
  );
}

const styles = {
  container: {
    backgroundColor: '#f9fafb',
    borderRadius: '8px',
    padding: '20px',
    marginTop: '20px',
  },
  title: {
    fontSize: '18px',
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: '12px',
  },
  metadata: {
    display: 'flex',
    gap: '12px',
    marginBottom: '16px',
  },
  badge: {
    padding: '4px 12px',
    backgroundColor: '#e5e7eb',
    borderRadius: '12px',
    fontSize: '12px',
    color: '#4b5563',
  },
  transcript: {
    backgroundColor: '#ffffff',
    padding: '16px',
    borderRadius: '6px',
    lineHeight: '1.6',
    color: '#374151',
    fontSize: '14px',
    border: '1px solid #e5e7eb',
  },
};

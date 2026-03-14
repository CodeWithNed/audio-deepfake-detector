export function ScoreBreakdown({ acousticScore, linguisticScore }) {
  const ScoreBar = ({ label, score, color }) => {
    const percentage = Math.round(score * 100);

    return (
      <div style={styles.scoreBar}>
        <div style={styles.scoreHeader}>
          <span style={styles.scoreLabel}>{label}</span>
          <span style={styles.scoreValue}>{percentage}%</span>
        </div>
        <div style={styles.barBackground}>
          <div
            style={{
              ...styles.barFill,
              width: `${percentage}%`,
              backgroundColor: color,
            }}
          />
        </div>
      </div>
    );
  };

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Score Breakdown</h3>
      <ScoreBar
        label="Acoustic Analysis (WavLM)"
        score={acousticScore}
        color="#3b82f6"
      />
      <ScoreBar
        label="Linguistic Analysis (RoBERTa)"
        score={linguisticScore}
        color="#8b5cf6"
      />
    </div>
  );
}

const styles = {
  container: {
    backgroundColor: '#ffffff',
    borderRadius: '8px',
    padding: '20px',
    marginTop: '20px',
    border: '1px solid #e5e7eb',
  },
  title: {
    fontSize: '18px',
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: '20px',
  },
  scoreBar: {
    marginBottom: '20px',
  },
  scoreHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '8px',
  },
  scoreLabel: {
    fontSize: '14px',
    color: '#4b5563',
  },
  scoreValue: {
    fontSize: '14px',
    fontWeight: '600',
    color: '#1f2937',
  },
  barBackground: {
    height: '12px',
    backgroundColor: '#e5e7eb',
    borderRadius: '6px',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    transition: 'width 0.5s ease',
    borderRadius: '6px',
  },
};

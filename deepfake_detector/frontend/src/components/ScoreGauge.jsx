export function ScoreGauge({ score, label }) {
  const percentage = Math.round(score * 100);
  const color = score < 0.3 ? '#10b981' : score < 0.7 ? '#f59e0b' : '#ef4444';

  return (
    <div style={styles.container}>
      <div style={styles.gaugeContainer}>
        <svg style={styles.svg} viewBox="0 0 100 100">
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="#e5e7eb"
            strokeWidth="10"
          />
          {/* Progress circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeDasharray={`${percentage * 2.827} 282.7`}
            strokeLinecap="round"
            transform="rotate(-90 50 50)"
          />
        </svg>
        <div style={styles.scoreText}>
          <div style={{ ...styles.percentage, color }}>{percentage}%</div>
          <div style={styles.label}>{label}</div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '20px',
  },
  gaugeContainer: {
    position: 'relative',
    width: '150px',
    height: '150px',
  },
  svg: {
    width: '100%',
    height: '100%',
    transform: 'rotate(0deg)',
  },
  scoreText: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    textAlign: 'center',
  },
  percentage: {
    fontSize: '32px',
    fontWeight: 'bold',
  },
  label: {
    fontSize: '12px',
    color: '#6b7280',
    marginTop: '4px',
  },
};

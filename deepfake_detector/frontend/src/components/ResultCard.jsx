import { ScoreGauge } from './ScoreGauge';
import { ScoreBreakdown } from './ScoreBreakdown';
import { TranscriptPanel } from './TranscriptPanel';

export function ResultCard({ result }) {
  const isFake = result.decision === 'FAKE';

  return (
    <div style={styles.container}>
      {/* Verdict Banner */}
      <div
        style={{
          ...styles.banner,
          backgroundColor: isFake ? '#fee2e2' : '#d1fae5',
          borderColor: isFake ? '#ef4444' : '#10b981',
        }}
      >
        <div style={styles.bannerContent}>
          <span style={styles.bannerIcon}>{isFake ? '⚠️' : '✓'}</span>
          <div>
            <h2
              style={{
                ...styles.verdict,
                color: isFake ? '#991b1b' : '#065f46',
              }}
            >
              {result.decision}
            </h2>
            <p style={styles.verdictSubtext}>
              Confidence: {Math.round(result.confidence * 100)}%
            </p>
          </div>
        </div>
      </div>

      {/* Score Gauge */}
      <ScoreGauge score={result.final_score} label="Fake Probability" />

      {/* Score Breakdown */}
      <ScoreBreakdown
        acousticScore={result.acoustic_score}
        linguisticScore={result.linguistic_score}
      />

      {/* Transcript */}
      {result.transcript && (
        <TranscriptPanel
          transcript={result.transcript}
          language={result.language}
          confidence={result.asr_confidence}
        />
      )}

      {/* Timestamp */}
      <div style={styles.timestamp}>
        Analyzed at: {new Date(result.timestamp).toLocaleString()}
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '800px',
    margin: '0 auto',
    marginTop: '30px',
  },
  banner: {
    padding: '24px',
    borderRadius: '8px',
    border: '2px solid',
    marginBottom: '24px',
  },
  bannerContent: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
  },
  bannerIcon: {
    fontSize: '48px',
  },
  verdict: {
    fontSize: '32px',
    fontWeight: 'bold',
    margin: '0',
  },
  verdictSubtext: {
    fontSize: '14px',
    margin: '4px 0 0 0',
    opacity: 0.8,
  },
  timestamp: {
    textAlign: 'center',
    marginTop: '20px',
    fontSize: '12px',
    color: '#9ca3af',
  },
};

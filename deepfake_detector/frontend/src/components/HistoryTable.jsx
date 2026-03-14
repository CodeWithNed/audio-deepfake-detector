export function HistoryTable({ history, onClear }) {
  if (!history || history.length === 0) {
    return (
      <div style={styles.emptyState}>
        <p>No detection history yet</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Detection History ({history.length})</h3>
        <button style={styles.clearButton} onClick={onClear}>
          Clear History
        </button>
      </div>

      <div style={styles.tableContainer}>
        <table style={styles.table}>
          <thead>
            <tr style={styles.headerRow}>
              <th style={styles.th}>Filename</th>
              <th style={styles.th}>Decision</th>
              <th style={styles.th}>Score</th>
              <th style={styles.th}>Confidence</th>
              <th style={styles.th}>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {history.map((item, index) => (
              <tr key={item.id || index} style={styles.row}>
                <td style={styles.td}>
                  <div style={styles.filename}>{item.filename}</div>
                  {item.transcript && (
                    <div style={styles.transcriptPreview}>
                      {item.transcript.substring(0, 50)}...
                    </div>
                  )}
                </td>
                <td style={styles.td}>
                  <span
                    style={{
                      ...styles.badge,
                      backgroundColor:
                        item.decision === 'FAKE' ? '#fee2e2' : '#d1fae5',
                      color: item.decision === 'FAKE' ? '#991b1b' : '#065f46',
                    }}
                  >
                    {item.decision}
                  </span>
                </td>
                <td style={styles.td}>
                  {Math.round(item.final_score * 100)}%
                </td>
                <td style={styles.td}>
                  {Math.round(item.confidence * 100)}%
                </td>
                <td style={styles.td}>
                  {new Date(item.timestamp).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const styles = {
  container: {
    marginTop: '40px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
  },
  title: {
    fontSize: '20px',
    fontWeight: '600',
    color: '#1f2937',
  },
  clearButton: {
    padding: '8px 16px',
    backgroundColor: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
  },
  tableContainer: {
    overflowX: 'auto',
    backgroundColor: '#ffffff',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
  },
  headerRow: {
    backgroundColor: '#f9fafb',
  },
  th: {
    padding: '12px',
    textAlign: 'left',
    fontSize: '12px',
    fontWeight: '600',
    color: '#6b7280',
    textTransform: 'uppercase',
    borderBottom: '1px solid #e5e7eb',
  },
  row: {
    borderBottom: '1px solid #f3f4f6',
  },
  td: {
    padding: '12px',
    fontSize: '14px',
    color: '#374151',
  },
  filename: {
    fontWeight: '500',
    marginBottom: '4px',
  },
  transcriptPreview: {
    fontSize: '12px',
    color: '#9ca3af',
  },
  badge: {
    padding: '4px 12px',
    borderRadius: '12px',
    fontSize: '12px',
    fontWeight: '600',
    display: 'inline-block',
  },
  emptyState: {
    textAlign: 'center',
    padding: '40px',
    color: '#9ca3af',
  },
};

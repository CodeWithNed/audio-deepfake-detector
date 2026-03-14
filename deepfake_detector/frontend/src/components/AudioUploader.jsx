import { useState } from 'react';

export function AudioUploader({ onFileSelect, isLoading }) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
      setSelectedFile(file);
      onFileSelect(file);
    } else {
      alert('Please upload an audio file');
    }
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      onFileSelect(file);
    }
  };

  return (
    <div style={styles.container}>
      <div
        style={{
          ...styles.dropZone,
          ...(isDragging ? styles.dropZoneDragging : {}),
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="audio/*"
          onChange={handleFileInput}
          style={styles.fileInput}
          id="audio-upload"
          disabled={isLoading}
        />
        <label htmlFor="audio-upload" style={styles.label}>
          <div style={styles.iconContainer}>
            <svg
              style={styles.icon}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
              />
            </svg>
          </div>
          <p style={styles.primaryText}>
            {selectedFile ? selectedFile.name : 'Drop audio file here or click to browse'}
          </p>
          <p style={styles.secondaryText}>WAV, MP3, FLAC (max 30 seconds)</p>
        </label>
      </div>
    </div>
  );
}

const styles = {
  container: {
    width: '100%',
    maxWidth: '600px',
    margin: '0 auto',
  },
  dropZone: {
    border: '2px dashed #cbd5e0',
    borderRadius: '8px',
    padding: '40px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    backgroundColor: '#f7fafc',
  },
  dropZoneDragging: {
    borderColor: '#3498db',
    backgroundColor: '#ebf8ff',
  },
  fileInput: {
    display: 'none',
  },
  label: {
    cursor: 'pointer',
    display: 'block',
  },
  iconContainer: {
    marginBottom: '16px',
  },
  icon: {
    width: '64px',
    height: '64px',
    margin: '0 auto',
    color: '#4a5568',
  },
  primaryText: {
    fontSize: '16px',
    fontWeight: '600',
    color: '#2d3748',
    marginBottom: '8px',
  },
  secondaryText: {
    fontSize: '14px',
    color: '#718096',
  },
};

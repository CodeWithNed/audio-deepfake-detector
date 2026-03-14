import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Home } from './pages/Home';
import { History } from './pages/History';

function App() {
  return (
    <BrowserRouter>
      <div style={styles.app}>
        <nav style={styles.nav}>
          <div style={styles.navContainer}>
            <Link to="/" style={styles.navBrand}>
              🎵 Deepfake Detector
            </Link>
            <div style={styles.navLinks}>
              <Link to="/" style={styles.navLink}>
                Home
              </Link>
              <Link to="/history" style={styles.navLink}>
                History
              </Link>
            </div>
          </div>
        </nav>

        <main style={styles.main}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </main>

        <footer style={styles.footer}>
          <p>
            Deepfake Audio Detector v1.0 | Powered by WavLM, Whisper & RoBERTa
          </p>
        </footer>
      </div>
    </BrowserRouter>
  );
}

const styles = {
  app: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: '#f9fafb',
  },
  nav: {
    backgroundColor: '#ffffff',
    borderBottom: '1px solid #e5e7eb',
    padding: '16px 0',
  },
  navContainer: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 20px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  navBrand: {
    fontSize: '20px',
    fontWeight: 'bold',
    color: '#1f2937',
    textDecoration: 'none',
  },
  navLinks: {
    display: 'flex',
    gap: '24px',
  },
  navLink: {
    fontSize: '16px',
    color: '#6b7280',
    textDecoration: 'none',
    fontWeight: '500',
  },
  main: {
    flex: 1,
  },
  footer: {
    backgroundColor: '#ffffff',
    borderTop: '1px solid #e5e7eb',
    padding: '20px',
    textAlign: 'center',
    color: '#6b7280',
    fontSize: '14px',
  },
};

export default App;

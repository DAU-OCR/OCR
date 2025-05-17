import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';

// default export 로 내보냈으니 아래처럼 중괄호 없이 import
import UploadPage  from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import FailedPage  from './pages/FailedPage';

import './App.css';

function App() {
  return (
    <Router>
      <nav className="nav-links">
        <Link to="/">메인</Link>
        <span className="divider">|</span>
        <Link to="/results">미리보기</Link>
        <span className="divider">|</span>
        <Link to="/failed">인식실패</Link>
      </nav>
      <div className="page-container">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/failed" element={<FailedPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

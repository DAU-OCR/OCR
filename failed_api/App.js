import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';

import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import FailedPage from './pages/FailedPage';

import './App.css';

export default function App() {
  const baseLink = "px-2 py-1 transition-colors duration-200";

  return (
    <Router>
      {/* 절대 위치 로고 */}
      <NavLink to="/" className="absolute-logo">
        <img src="/icons/logo.png" alt="로고" className="logo-img" />
      </NavLink>

      {/* 네비게이션 바 (오른쪽 메뉴들) */}
      <nav className="nav-links flex justify-end px-6 py-2 border-b-2 bg-white shadow-sm">
        <div className="flex space-x-6">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `${baseLink} ${
                isActive
                  ? "text-blue-600 border-b-2 border-blue-600"
                  : "text-black hover:text-blue-600"
              }`
            }
          >
            메인
          </NavLink>
          <NavLink
            to="/results"
            className={({ isActive }) =>
              `${baseLink} ${
                isActive
                  ? "text-blue-600 border-b-2 border-blue-600"
                  : "text-black hover:text-blue-600"
              }`
            }
          >
            미리보기
          </NavLink>
          <NavLink
            to="/failed"
            className={({ isActive }) =>
              `${baseLink} ${
                isActive
                  ? "text-blue-600 border-b-2 border-blue-600"
                  : "text-black hover:text-blue-600"
              }`
            }
          >
            인식실패
          </NavLink>
        </div>
      </nav>

      {/* 페이지 본문 */}
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

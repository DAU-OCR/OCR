// src/App.js

import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';

import UploadPage  from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import FailedPage  from './pages/FailedPage';

import './App.css';

export default function App() {
  // 공통 클래스
  const baseLink = "px-2 py-1 transition-colors duration-200";
  
  return (
    <Router>
      <nav className="nav-links flex justify-center items-center space-x-6 border-b-2 bg-white">
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

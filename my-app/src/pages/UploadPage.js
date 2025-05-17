import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './UploadPage.css';

export default function UploadPage() {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const onFileChange = e => {
    const fileList = Array.from(e.target.files);
    setFiles(fileList);
    setPreviews(fileList.map(f => URL.createObjectURL(f)));
  };

  const onUpload = async () => {
    if (files.length === 0) return alert('이미지를 선택하세요');
    setLoading(true);
    const fd = new FormData();
    files.forEach(f => fd.append('images', f));
    try {
      await axios.post('/upload', fd, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      navigate('/results');
    } catch (e) {
      console.error(e);
      alert('서버 오류: ' + (e.response?.statusText || e.message));
      setLoading(false);
    }
  };

  return (
    <div className="upload-page">
      <a className="profile-link" href="/profile">
      </a>
      <div className="card">
        <img
          src="/icons/upload.png"
          alt="Upload Icon"
          className="upload-icon"
        />

        <h1>이미지 업로드</h1>

        <input
          id="file-input"
          type="file"
          accept="image/*"
          multiple
          onChange={onFileChange}
          className="file-input"
        />
        <label htmlFor="file-input" className="file-label">
          파일 선택
        </label>

        <div className="preview-container">
          {previews.map((src, i) => (
            <img
              key={i}
              src={src}
              alt={`preview ${i+1}`}
              className="preview"
            />
          ))}
        </div>

        <button
          className="submit-button"
          onClick={onUpload}
          disabled={loading}
        >
          파일 생성 시작
        </button>
      </div>

      {/* 로딩 오버레이 */}
      {loading && (
        <div className="loader-overlay">
          <div className="loader" />
          <div className="loading-text">파일 생성 중...</div>
        </div>
      )}
    </div>
  );
}

import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion'; // ✅ 추가
import './UploadPage.css';

export default function UploadPage() {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [processedCount, setProcessedCount] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const startTimeRef = useRef(null);
  const navigate = useNavigate();

  // ✅ 백엔드 준비 상태 추가
  const [isBackendReady, setIsBackendReady] = useState(false);

  // ✅ 백엔드 준비 상태 확인을 위한 폴링
  useEffect(() => {
    let intervalId;
    const checkBackend = async () => {
      try {
        await axios.get('http://localhost:5000/results'); // 간단한 엔드포인트로 확인
        setIsBackendReady(true);
        clearInterval(intervalId); // 준비되면 폴링 중지
      } catch (error) {
        // console.log('Backend not ready yet...');
      }
    };

    intervalId = setInterval(checkBackend, 1000); // 1초마다 확인

    // 컴포넌트 언마운트 시 인터벌 정리
    return () => clearInterval(intervalId);
  }, []);

  const onFileChange = e => {
    const list = Array.from(e.target.files);
    setFiles(list);
    setPreviews(list.map(f => URL.createObjectURL(f)));
    setTotalCount(0);
    setProcessedCount(0);
    setProcessingProgress(0);
    setLoading(false);
  };

  const formatTime = ms => {
    const sec = Math.ceil(ms / 1000);
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}분 ${s}초`;
  };

  const onUpload = async () => {
    if (!files.length) return alert('파일을 선택하세요');
    setLoading(true);
    setTotalCount(files.length);
    setProcessedCount(0);
    setProcessingProgress(0);
    startTimeRef.current = Date.now();

    for (let i = 0; i < files.length; i++) {
      const fd = new FormData();
      fd.append('image', files[i]);
      try {
        await axios.post('http://localhost:5000/upload', fd);
      } catch (err) {
        console.error('처리 중 에러:', err);
      }
      const done = i + 1;
      setProcessedCount(done);
      setProcessingProgress(Math.round((done * 100) / files.length));
    }

    navigate('/results');
  };

  const elapsed = startTimeRef.current ? Date.now() - startTimeRef.current : 0;
  const remaining = totalCount && processedCount
    ? elapsed / processedCount * (totalCount - processedCount)
    : 0;

  return (
    <motion.div
      className="upload-page"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="card">
        <img src="./icons/upload.png" alt="Upload Icon" className="upload-icon" />
        <h1>이미지 업로드</h1>

        <input
          id="file-input"
          type="file"
          accept="image/*,.zip"
          multiple
          onChange={onFileChange}
          className="file-input"
        />
        <label htmlFor="file-input" className="file-label">
          파일 선택
        </label>

        <div className="preview-container">
          {previews.map((src, idx) => (
            <img key={idx} src={src} alt={`preview ${idx + 1}`} className="preview" />
          ))}
        </div>

        <button
          className="submit-button"
          onClick={onUpload}
          disabled={loading || !isBackendReady} // 백엔드가 준비되지 않으면 비활성화
        >
          {isBackendReady ? '파일 생성 시작' : 'OCR 준비 중...'} {/* 텍스트 변경 */}
        </button>
      </div>

      {loading && (
        <div className="loader-overlay">
          <div className="overlay-box">
            <div className="overlay-title">처리 중...</div>

            {/* ✅ 단계 표시 (가로형) */}
            <div className="phase-progress-horizontal">
              {['이미지 업로드', 'OCR 진행 중', '결과 처리 중', '완료'].map((label, idx, arr) => (
                <div className="horizontal-step" key={idx}>
                  <div className={`phase-circle ${idx <= Math.floor(processingProgress / 25) ? 'active' : ''}`}>
                    {idx + 1}
                  </div>
                  <div className="phase-label">{label}</div>
                  {idx < arr.length - 1 && <div className="horizontal-line" />}
                </div>
              ))}
            </div>

            <div className="progress-bar">
              <div
                className="progress-fill processing"
                style={{ width: `${processingProgress}%` }}
              />
            </div>

            <div className="progress-label">
              {processedCount} / {totalCount} ({processingProgress}%)
              {processedCount > 0 && processedCount < totalCount && (
                <> · 예상 남은 시간: {formatTime(remaining)}</>
              )}
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}

import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import './UploadPage.css';

export default function UploadPage() {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [processedCount, setProcessedCount] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false); // ✅ 드래그 상태 추가
  const startTimeRef = useRef(null);
  const navigate = useNavigate();

  const [isBackendReady, setIsBackendReady] = useState(false);

  useEffect(() => {
    let intervalId;
    const checkBackend = async () => {
      try {
        await axios.get('http://localhost:5000/results');
        setIsBackendReady(true);
        clearInterval(intervalId);
      } catch (error) {
        // console.log('Backend not ready yet...');
      }
    };

    intervalId = setInterval(checkBackend, 1000);

    return () => clearInterval(intervalId);
  }, []);

  // ✅ 파일 처리 로직을 별도 함수로 분리
  const handleFiles = useCallback((fileList) => {
    const list = Array.from(fileList);
    setFiles(list);
    // ZIP 파일은 미리보기를 생성하지 않고, 일반 아이콘으로 대체
    setPreviews(list.map(f => f.name.toLowerCase().endsWith('.zip') ? './icons/zip-icon.svg' : URL.createObjectURL(f)));
    setTotalCount(0);
    setProcessedCount(0);
    setProcessingProgress(0);
    setLoading(false);
  }, []);

  const onFileChange = e => {
    handleFiles(e.target.files);
  };

  // ✅ 드래그 앤 드롭 핸들러
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
      e.dataTransfer.clearData();
    }
  }, [handleFiles]);

  const formatTime = ms => {
    const sec = Math.ceil(ms / 1000);
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}분 ${s}초`;
  };

  const AVERAGE_TIME_PER_IMAGE_MS = 300; // 이미지당 평균 처리 시간 (밀리초)
  const FAKE_PROGRESS_THRESHOLD = 95; // 가짜 진행률이 도달할 최대 퍼센트

  const onUpload = async () => {
    if (!files.length) return alert('파일을 선택하세요');
    setLoading(true);
    setProcessedCount(0);
    setProcessingProgress(0);
    startTimeRef.current = Date.now();

    let totalImagesToProcess = 0;
    const fileDetails = []; // 각 파일의 이미지 수를 저장할 배열

    // 1단계: 총 이미지 수 미리 계산 (ZIP 파일 내부 이미지 포함) 및 파일별 이미지 수 저장
    for (const file of files) {
      let imageCountForFile = 1; // 단일 파일의 경우 기본값
      if (file.name.toLowerCase().endsWith('.zip')) {
        const fd = new FormData();
        fd.append('file', file);
        try {
          const res = await axios.post('http://localhost:5000/get-zip-image-count', fd);
          imageCountForFile = res.data.image_count;
          if (imageCountForFile === 0) {
            alert(`ZIP 파일 (${file.name}) 내에 처리할 이미지가 없습니다.`);
            setLoading(false);
            return;
          }
        } catch (err) {
          console.error('ZIP 파일 이미지 수 계산 중 에러:', err);
          alert(`ZIP 파일 (${file.name}) 처리 중 오류가 발생했습니다.`);
          setLoading(false);
          return;
        }
      }
      totalImagesToProcess += imageCountForFile;
      fileDetails.push({ file, imageCount: imageCountForFile });
    }
    setTotalCount(totalImagesToProcess);

    let cumulativeProcessedCount = 0;
    let fakeProgressInterval = null; // Ref to store interval ID

    // 2단계: 실제 파일 업로드 및 진행률 업데이트
    for (let i = 0; i < fileDetails.length; i++) {
      const { file, imageCount } = fileDetails[i];
      const fd = new FormData();
      fd.append('image', file); 

      // ZIP 파일인 경우 가짜 진행률 시작
      if (file.name.toLowerCase().endsWith('.zip')) {
        const estimatedZipProcessingTime = imageCount * AVERAGE_TIME_PER_IMAGE_MS;
        const intervalDuration = 100; // 100ms마다 업데이트
        const incrementPerInterval = (FAKE_PROGRESS_THRESHOLD / (estimatedZipProcessingTime / intervalDuration));
        
        let currentFakeProgress = 0;
        fakeProgressInterval = setInterval(() => {
          currentFakeProgress = Math.min(currentFakeProgress + incrementPerInterval, FAKE_PROGRESS_THRESHOLD); // 최대 95%까지 증가
          // 현재 파일의 가짜 진행률을 전체 진행률에 반영
          const baseProgress = (cumulativeProcessedCount * 100) / totalImagesToProcess;
          const currentFileContribution = (currentFakeProgress / 100) * (imageCount * 100 / totalImagesToProcess);
          setProcessingProgress(Math.round(baseProgress + currentFileContribution));
        }, intervalDuration);
      }

      try {
        const res = await axios.post('http://localhost:5000/upload', fd);
        
        // 가짜 진행률 중지
        if (fakeProgressInterval) {
          clearInterval(fakeProgressInterval);
          fakeProgressInterval = null;
        }

        const { processed_count } = res.data;

        cumulativeProcessedCount += processed_count;

        setProcessedCount(cumulativeProcessedCount);
        const currentProgress = Math.round((cumulativeProcessedCount * 100) / totalImagesToProcess);
        setProcessingProgress(currentProgress);

      } catch (err) {
        console.error('처리 중 에러:', err);
        // 가짜 진행률 중지
        if (fakeProgressInterval) {
          clearInterval(fakeProgressInterval);
          fakeProgressInterval = null;
        }
        // 오류 발생 시에도 진행률 업데이트를 위해 1개 파일 처리로 간주 (정확도는 떨어질 수 있음)
        cumulativeProcessedCount += imageCount; // 오류난 파일의 이미지 수만큼 누적
        setProcessedCount(cumulativeProcessedCount);
        const currentProgress = Math.round((cumulativeProcessedCount * 100) / totalImagesToProcess);
        setProcessingProgress(currentProgress);
      }
    }

    // 모든 파일 처리 완료 후, 혹시 남아있을 수 있는 가짜 진행률 인터벌 정리
    if (fakeProgressInterval) {
      clearInterval(fakeProgressInterval);
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
      <div 
        className={`card dropzone ${isDragging ? 'dragging' : ''}`} // ✅ 드래그 상태에 따른 클래스 추가
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
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
          파일 선택 또는 드래그
        </label>

        <div className="preview-container">
          {previews.map((src, idx) => (
            <img key={idx} src={src} alt={`preview ${idx + 1}`} className="preview" />
          ))}
        </div>

        <button
          className="submit-button"
          onClick={onUpload}
          disabled={loading || !isBackendReady}
        >
          {isBackendReady ? '파일 생성 시작' : 'OCR 준비 중...'}
        </button>
      </div>

      {loading && (
        <div className="loader-overlay">
          <div className="overlay-box">
            <div className="overlay-title">처리 중...</div>

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

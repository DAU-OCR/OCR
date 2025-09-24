import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';

export default function ResultsPage() {
  const [rows, setRows] = useState([]);
  const [downloadingExcel, setDownloadingExcel] = useState(false);
  const [downloadingJson, setDownloadingJson] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [message, setMessage] = useState('');

  useEffect(() => {
    // Fetch initial results from the server
    axios.get('http://localhost:5000/results')
      .then(res => {
        // 이미지 이름으로 정렬하는 코드를 제거하여 업로드 순서를 유지합니다.
        setRows(res.data);
      })
      .catch(e => {
        console.error('ERROR GET /results', e);
        setMessage('결과를 가져오는 중 오류가 발생했습니다.');
      });
  }, []);

  const handleDownloadExcel = async () => {
    try {
      setDownloadingExcel(true);
      setMessage('');
      
      // 1. Send updated plate values to the server
      await axios.post('http://localhost:5000/update-plates', rows.map(r => ({
        image: r.image,
        plate: r.plate
      })));

      // 2. Create a filename based on the current date
      const now = new Date();
      const yyyy = now.getFullYear();
      const mm = String(now.getMonth() + 1).padStart(2, '0');
      const dd = String(now.getDate()).padStart(2, '0');
      const filename = `${yyyy}-${mm}-${dd}_차량번호판.xlsx`;

      let writable;
      if ('showSaveFilePicker' in window) {
        try {
          const handle = await window.showSaveFilePicker({
            suggestedName: filename,
            types: [{
              description: 'Excel 파일',
              accept: {
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
              }
            }]
          });
          writable = await handle.createWritable();
        } catch (e) {
          if (e.name === 'AbortError') return;
          throw e;
        }
      }

      // 3. Request the download from the server
      const res = await axios.get('http://localhost:5000/download', { responseType: 'blob' });
      const blob = new Blob([res.data], {
        type: res.headers['content-type'],
      });

      // 4. Save the file
      if (writable) {
        await writable.write(blob);
        await writable.close();
      } else {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      }
    } catch (e) {
      if (e.response?.status === 400) {
        setMessage('인식된 차량 번호판이 없습니다.');
        return;
      }
      setMessage('다운로드 중 오류가 발생했습니다.');
    } finally {
      setDownloadingExcel(false);
    }
  };

  const handleDownloadJson = async () => {
    try {
      setDownloadingJson(true);
      setMessage('');

      // 서버의 JSON 다운로드 엔드포인트를 호출
      const res = await axios.get('http://localhost:5000/download-json', { responseType: 'blob' });
      const blob = new Blob([res.data], {
        type: res.headers['content-type'],
      });

      // 파일명을 현재 날짜를 기반으로 생성
      const now = new Date();
      const yyyy = now.getFullYear();
      const mm = String(now.getMonth() + 1).padStart(2, '0');
      const dd = String(now.getDate()).padStart(2, '0');
      const filename = `${yyyy}-${mm}-${dd}_raw데이터.json`;

      // Use the File System Access API if available
      if ('showSaveFilePicker' in window) {
        try {
          const handle = await window.showSaveFilePicker({
            suggestedName: filename,
            types: [{
              description: 'JSON 파일',
              accept: {
                'application/json': ['.json']
              }
            }]
          });
          const writable = await handle.createWritable();
          await writable.write(blob);
          await writable.close();
        } catch (e) {
          if (e.name === 'AbortError') return;
          throw e;
        }
      } else {
        // Fallback for older browsers
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      }

      setMessage('JSON 파일 다운로드가 완료되었습니다.');
    } catch (e) {
      if (e.response?.status === 400) {
        setMessage('인식된 차량 번호판이 없습니다.');
        return;
      }
      setMessage('다운로드 중 오류가 발생했습니다.');
    } finally {
      setDownloadingJson(false);
    }
  };

  const handleReset = async () => {
    try {
      await axios.post('http://localhost:5000/reset');
      setRows([]);
      setMessage('결과가 초기화되었습니다.');
    } catch (e) {
      setMessage('초기화 중 오류가 발생했습니다.');
    }
  };

  const handlePlateChange = (index, newValue) => {
    const updated = [...rows];
    updated[index].plate = newValue;
    setRows(updated);
  };

  return (
    <motion.div
      className="results-page"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <style>
        {`
        /* ResultsPage.css (Updated for full-width card and preview box) */

        /* Override App.css page-container limits */
        .page-container {
          max-width: none !important;
          width: 100% !important;
          padding: 0 !important;
          margin: 0 !important;
        }

        /* Main wrapper */
        .results-page {
          display: flex;
          justify-content: center;
          align-items: flex-start;
          padding-top: 100px;
          background: none;
          width: 100%;
          box-sizing: border-box;
        }

        /* Card fills viewport width with max constraints */
        .results-page .card {
          width: 90vw; 
          max-width: 1200px; 
          background: #fff;
          border-radius: 8px;
          padding: 40px;
          box-shadow:
            0 4px 12px rgba(0,0,0,0.15),
            0 8px 24px rgba(0,0,0,0.10);
          margin: 0 auto;
          text-align: center;
          box-sizing: border-box;
        }

        /* Title */
        .preview-title {
          margin-bottom: 24px;
          font-size: 24px;
          font-weight: bold;
          color: #333;
        }

        /* Preview box expands full width inside card */
        .results-page .preview-box {
          width: 100%;
          height: 600px;
          margin: 0 auto 32px;
          padding: 20px;
          background: #fafafa;
          border: 1px solid #ddd;
          border-radius: 8px;
          overflow-y: auto;
          box-sizing: border-box;
        }

        /* Table styles */
        .results-table {
          width: 100%;
          border-collapse: collapse;
        }
        .results-table th,
        .results-table td {
          padding: 16px;
          text-align: center;
          border-bottom: 1px solid #eee;
        }

        /* 선택된 결과 강조 스타일 */
        .highlighted-plate {
          font-size: 18px;
          font-weight: bold;
          color: #007BFF; /* 파란색 (Bootstrap Primary Blue) */
        }
        .results-table th {
          background: #f0f0f0;
        }

        /* Image thumbnail */
        .row-image {
          width: 160px;
          height: auto;
          border-radius: 4px;
          object-fit: cover;
        }

        /* Buttons auto-width */
        .actions {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 16px;
          margin-top: 24px;
        }
        .download-button,
        .reset-button {
          display: inline-flex;
          align-items: center;
          padding: 10px 20px;
          border: none;
          border-radius: 4px;
          background: #fff;
          color: #333;
          font-size: 16px;
          cursor: pointer;
          box-shadow: 0 2px 6px rgba(0,0,0,0.1);
          width: auto;
        }
        .download-button:hover,
        .reset-button:hover {
          background: #f0f0f0;
        }

        /* Icon size */
        .download-icon {
          width: 24px;
          height: 24px;
          margin-right: 8px;
        }

        /* ✅ 확대 보기 오버레이 */
        .image-overlay {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: rgba(0,0,0,0.7);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 2000;
        }

        .image-popup {
          position: relative;
          background: white;
          padding: 16px;
          border-radius: 8px;
          max-width: 90vw;
          max-height: 90vh;
          overflow: auto;
          box-shadow: 0 0 16px rgba(0,0,0,0.3);
          animation: fadeIn 0.3s ease-out;
        }

        .enlarged-image {
          width: 100%;
          height: auto;
          max-height: 80vh;
          border-radius: 4px;
        }

        .close-button {
          position: absolute;
          top: 8px;
          right: 12px;
          background: none;
          border: none;
          font-size: 28px;
          font-weight: bold;
          color: #333;
          cursor: pointer;
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: scale(0.95); }
          to { opacity: 1; transform: scale(1); }
        }

        /* 선택된 결과 입력 필드 */
        .plate-input {
          font-size: 16px;
          padding: 6px 8px;
          width: 100%;
          max-width: 160px;
          border: 1px solid #ccc;
          border-radius: 4px;
          text-align: center;
          box-sizing: border-box;
        }
        `}
      </style>
      <div className="card">
        <h2 className="preview-title">미리보기</h2>
        <div className="preview-box">
          <table className="results-table">
            <thead>
              <tr>
                <th>연번</th>
                <th>차량사진</th>
                <th>모델1 결과</th>
                <th>모델2 결과</th>
                <th>선택된 결과</th>
              </tr>
            </thead>
            <tbody>
              {rows.length > 0 ? (
                rows.map((r, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>
                      <img
                        src={r.visual || r.image}
                        alt={`차량 ${i + 1}`}
                        className="row-image"
                        onClick={() => setSelectedImage(r.visual || r.image)}
                      />
                    </td>
                    <td>{r.text1 || '-'}</td>
                    <td>{r.text2 || '-'}</td>
                    <td>
                      <input
                        type="text"
                        className="plate-input"
                        value={r.plate || ''}
                        onChange={e => handlePlateChange(i, e.target.value)}
                      />
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5" className="no-data">데이터가 없습니다.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="actions">
          <button 
            className="download-button" 
            onClick={handleDownloadExcel} 
            disabled={downloadingExcel}
          >
            {downloadingExcel ? '다운로드 중…' : (
              <>
                <img src="/icons/download.png" alt="다운로드" className="download-icon" />
                <span>엑셀 파일 다운로드</span>
              </>
            )}
          </button>
          <button 
            className="download-button" 
            onClick={handleDownloadJson} 
            disabled={downloadingJson}
          >
            {downloadingJson ? '다운로드 중…' : (
              <>
                <img src="/icons/download.png" alt="다운로드" className="download-icon" />
                <span>JSON 파일 다운로드</span>
              </>
            )}
          </button>
          <button className="reset-button" onClick={handleReset}>
            초기화
          </button>
        </div>
      </div>

      {/* 확대 보기 */}
      {selectedImage && (
        <div className="image-overlay" onClick={() => setSelectedImage(null)}>
          <div className="image-popup" onClick={(e) => e.stopPropagation()}>
            <button className="close-button" onClick={() => setSelectedImage(null)}>&times;</button>
            <img src={selectedImage} alt="확대 보기" className="enlarged-image" />
          </div>
        </div>
      )}
    </motion.div>
  );
}

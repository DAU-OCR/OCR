import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import './ResultsPage.css';

export default function ResultsPage() {
  const [rows, setRows] = useState([]);
  const [downloading, setDownloading] = useState(false);
  const [selectedRow, setSelectedRow] = useState(null); 

  useEffect(() => {
    axios.get('http://localhost:5000/results')
      .then(res => {
        const sorted = res.data
          .map(r => ({
            ...r,
            plate: r.plate?.trim() || '인식 실패'
          }))
          .sort((a, b) => a.image.localeCompare(b.image));
        setRows(sorted);
      })
      .catch(e => {
        console.error('ERROR GET /results', e);
        alert('결과를 가져오는 중 오류가 발생했습니다.');
      });
  }, []);

  // ✅ 키보드 이벤트 핸들러 추가 및 Page Up/Down 스크롤 방지
  useEffect(() => {
    const handleKeyDown = (e) => {
      // 팝업이 열려 있는 경우에만 작동
      if (selectedRow) {
        const currentIndex = rows.findIndex(r => r.image === selectedRow.image);
        
        if (e.key === 'PageDown') {
          e.preventDefault(); // 스크롤 방지
          const nextIndex = currentIndex + 1;
          if (nextIndex < rows.length) {
            setSelectedRow(rows[nextIndex]);
          }
        } else if (e.key === 'PageUp') {
          e.preventDefault(); // 스크롤 방지
          const prevIndex = currentIndex - 1;
          if (prevIndex >= 0) {
            setSelectedRow(rows[prevIndex]);
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedRow, rows]);

  const handleDownload = async () => {
    try {
      await axios.post('/update-plates', rows.map(r => ({
        image: r.image,
        plate: r.plate?.trim() || '인식 실패'
      })));

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

      setDownloading(true);
      const res = await axios.get('/download', { responseType: 'blob' });
      const blob = new Blob([res.data], {
        type: res.headers['content-type'],
      });

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
        alert('인식된 차량 번호판이 없습니다.');
        return;
      }
      alert('다운로드 중 오류가 발생했습니다.');
    } finally {
      setDownloading(false);
    }
  };

  const handleReset = async () => {
    if (!window.confirm('정말 결과를 초기화하시겠습니까?')) return;
    try {
      await axios.post('http://localhost:5000/reset');
      setRows([]);
    } catch (e) {
      alert('초기화 중 오류가 발생했습니다.');
    }
  };

  const handlePlateChange = (index, newValue) => {
    const updated = [...rows];
    updated[index].plate = newValue;
    setRows(updated);
  };
  
  const handlePopupPlateChange = (newValue) => {
    if (!selectedRow) return;
    const index = rows.findIndex(r => r.image === selectedRow.image);
    if (index !== -1) {
      const updatedRows = [...rows];
      updatedRows[index].plate = newValue;
      setRows(updatedRows);
      setSelectedRow({
        ...selectedRow,
        plate: newValue
      });
    }
  };

  return (
    <motion.div
      className="results-page"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
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
                        src={`http://localhost:5000${r.visual || r.image}`}
                        alt={`차량 ${i + 1}`}
                        className="row-image"
                        onClick={() => setSelectedRow(r)}
                      />
                    </td>
                    <td>{r.text1 || '-'}</td>
                    <td>{r.text2 || '-'}</td>
                    <td>
                      <input
                        type="text"
                        className="plate-input"
                        value={r.plate?.trim() || '인식 실패'}
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
          <button className="download-button" onClick={handleDownload} disabled={downloading}>
            {downloading ? '다운로드 중…' : (
              <>
                <img src="/icons/download.png" alt="다운로드" className="download-icon" />
                <span>엑셀 파일 다운로드</span>
              </>
            )}
          </button>
          <button className="reset-button" onClick={handleReset}>
            초기화
          </button>
        </div>
      </div>

      {selectedRow && (
        <div className="image-overlay" onClick={() => setSelectedRow(null)}>
          <div className="image-popup" onClick={(e) => e.stopPropagation()}>
            <button className="close-button" onClick={() => setSelectedRow(null)}>×</button>
            <div className="popup-content">
              <img src={`http://localhost:5000${selectedRow.visual || selectedRow.image}`} alt="확대 보기" className="enlarged-image" />
              <div className="popup-details">
                <h3>인식 결과 수정</h3>
                <div className="result-group">
                  <span className="result-label">모델1 결과:</span>
                  <span className="result-value">{selectedRow.text1 || '없음'}</span>
                </div>
                <div className="result-group">
                  <span className="result-label">모델2 결과:</span>
                  <span className="result-value">{selectedRow.text2 || '없음'}</span>
                </div>
                <div className="result-group">
                  <span className="result-label">최종 결과:</span>
                  <input
                    type="text"
                    className="popup-plate-input"
                    value={selectedRow.plate?.trim() || '인식 실패'}
                    onChange={e => handlePopupPlateChange(e.target.value)}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}
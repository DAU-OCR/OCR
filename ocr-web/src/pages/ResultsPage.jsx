import React, { useEffect, useState } from 'react';
import axios from 'axios';
axios.defaults.baseURL = 'http://localhost:5000'; // axios 기본 URL 설정
import { motion } from 'framer-motion';
import './ResultsPage.css';

export default function ResultsPage() {
  const [rows, setRows] = useState([]);
  const [downloading, setDownloading] = useState(false);
  const [selectedRow, setSelectedRow] = useState(null); 
  const [focusedIndex, setFocusedIndex] = useState(-1); // For keyboard navigation
  const inputRefs = React.useRef([]); // To focus input elements

  // Keyboard navigation logic
  React.useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setFocusedIndex(prev => Math.min(prev + 1, rows.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setFocusedIndex(prev => Math.max(prev - 1, 0));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [rows.length]);

  // Effect to focus the input when focusedIndex changes
  React.useEffect(() => {
    if (focusedIndex !== -1 && inputRefs.current[focusedIndex]) {
      inputRefs.current[focusedIndex].focus();
      inputRefs.current[focusedIndex].select();
    }
  }, [focusedIndex]); 

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

  const handleDownload = async () => {
    try {
      // ✅ 1. 수정된 plate 값 서버에 전송
      await axios.post('/update-plates', rows.map(r => ({
        image: r.image,
        plate: r.plate?.trim() || '인식 실패' // 빈 문자열인 경우 '인식 실패'로 변환하여 전송
      })));

      // ✅ 2. 날짜 기반 파일 이름 생성
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

      // ✅ 3. 다운로드 요청
      setDownloading(true);
      const res = await axios.get('/download', { responseType: 'blob' });
      const blob = new Blob([res.data], {
        type: res.headers['content-type'],
      });

      // ✅ 4. 파일 저장
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
  
  // 팝업 내부에서 입력값 변경을 처리하는 새로운 함수
  const handlePopupPlateChange = (newValue) => {
    if (!selectedRow) return;

    // 현재 팝업에 표시된 행의 인덱스를 찾아 수정
    const index = rows.findIndex(r => r.image === selectedRow.image);
    if (index !== -1) {
      const updatedRows = [...rows];
      updatedRows[index].plate = newValue;
      setRows(updatedRows);
      
      // 팝업에 표시되는 상태도 업데이트
      setSelectedRow({
        ...selectedRow,
        plate: newValue
      });
    }
  };

  const handleJsonDownload = async () => {
    setDownloading(true);
    try {
      const res = await axios.get('http://localhost:5000/download-json', { responseType: 'blob' });
      const blob = new Blob([res.data], {
        type: 'application/json;charset=utf-8',
      });

      const contentDisposition = res.headers['content-disposition'];
      let filename = `ocr_results_${new Date().toISOString().slice(0,10)}.json`;
      if (contentDisposition) {
          const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
          if (filenameMatch && filenameMatch.length > 1) {
              filename = filenameMatch[1];
          }
      }

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

    } catch (e) {
      if (e.response?.status === 400) {
        alert('내보낼 데이터가 없습니다.');
        return;
      }
      console.error('JSON Download Error:', e);
      alert('JSON 다운로드 중 오류가 발생했습니다.');
    } finally {
      setDownloading(false);
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
                  <tr
                    key={i}
                    className={i === focusedIndex ? 'focused-row' : ''}
                    onClick={() => setFocusedIndex(i)}
                  >
                    <td>{i + 1}</td>
                    <td>
                      <img
                        src={`http://localhost:5000${r.visual || r.image}`}
                        alt={`차량 ${i + 1}`}
                        className="row-image"
                        onDoubleClick={() => setSelectedRow(r)} // Double click to open popup
                      />
                    </td>
                    <td>{r.text1 || '-'}</td>
                    <td>{r.text2 || '-'}</td>
                    <td>
                      <input
                        ref={el => (inputRefs.current[i] = el)}
                        type="text"
                        className="plate-input"
                        value={r.plate} // '인식 실패'일 때 '인식 실패' 표시, 사용자 입력 유지
                        onChange={e => handlePlateChange(i, e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter') {
                            e.preventDefault();
                            setFocusedIndex(prev => Math.min(prev + 1, rows.length - 1));
                          }
                        }}
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
                <img src="./icons/download.png" alt="다운로드" className="download-icon" />
                <span>엑셀 파일 다운로드</span>
              </>
            )}
          </button>
          <button className="download-button" onClick={handleJsonDownload} disabled={downloading}>
            {downloading ? '다운로드 중…' : 'JSON으로 내보내기'}
          </button>
          <button className="reset-button" onClick={handleReset}>
            초기화
          </button>
        </div>
      </div>

      {/* ✅ 확대 보기 팝업 (수정 기능 추가) */}
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
                    value={selectedRow.plate} // '인식 실패'일 때 '인식 실패' 표시, 사용자 입력 유지
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
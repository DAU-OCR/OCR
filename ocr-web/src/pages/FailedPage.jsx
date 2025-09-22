import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { motion } from 'framer-motion';
import './FailedPage.css';

export default function FailedPage() {
  const [rows, setRows] = useState([]);
  // ✅ 클릭된 행의 전체 데이터를 저장하도록 변경
  const [selectedRow, setSelectedRow] = useState(null); 
  const navigate = useNavigate();

  useEffect(() => {
    axios.get('/results')
      .then(res => {
        const failed = res.data
          .filter(r => !r.matched)
          .map(r => ({
            ...r,
            // plate 값이 없으면 '인식 실패'로 초기화
            plate: r.plate?.trim() || '인식 실패'
          }))
          .sort((a, b) => a.image.localeCompare(b.image));
        setRows(failed);
      })
      .catch(e => {
        console.error('실패한 사진 조회 오류', e);
        alert('실패한 사진을 가져오는 중 오류가 발생했습니다.');
      });
  }, []);

  const handlePlateChange = (index, newValue) => {
    const updated = [...rows];
    updated[index].plate = newValue;
    setRows(updated);
  };
  
  // 팝업 내부에서 입력값 변경을 처리하는 함수
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

  const handleSaveAndGoBack = async () => {
    try {
      // ✅ 수정된 plate 값 서버에 전송
      await axios.post('/update-plates', rows.map(r => ({
        image: r.image,
        plate: r.plate?.trim() || '인식 실패'
      })));
      alert('수정된 내용이 저장되었습니다.');
      navigate(-1);
    } catch (e) {
      console.error('수정 내용 저장 오류', e);
      alert('수정 내용을 저장하는 중 오류가 발생했습니다.');
    }
  };

  return (
    <motion.div
      className="results-page failed-page"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="card">
        <h2 className="preview-title">실패한 사진 모음</h2>
        <div className="preview-box">
          {rows.length > 0 ? (
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
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>
                      <img
                        src={r.visual || r.image}
                        alt={`실패 ${i + 1}`}
                        className="row-image"
                        // ✅ 클릭 시 전체 행 객체를 전달하도록 변경
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
                ))}
              </tbody>
            </table>
          ) : (
            <div className="no-data">실패한 사진이 없습니다.</div>
          )}
        </div>
        <div className="actions">
          {/* ✅ 버튼 기능 변경: 수정 내용을 저장하고 뒤로가기 */}
          <button className="save-button" onClick={handleSaveAndGoBack}>
            수정 내용 저장 후 뒤로가기
          </button>
        </div>
      </div>

      {/* ✅ 확대 보기 팝업 (수정 기능 추가) */}
      {selectedRow && (
        <div className="image-overlay" onClick={() => setSelectedRow(null)}>
          <div className="image-popup" onClick={(e) => e.stopPropagation()}>
            <button className="close-button" onClick={() => setSelectedRow(null)}>×</button>
            <div className="popup-content">
              <img src={selectedRow.visual || selectedRow.image} alt="확대 보기" className="enlarged-image" />
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
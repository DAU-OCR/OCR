import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { motion } from 'framer-motion';
import './FailedPage.css';

export default function FailedPage() {
  const [rows, setRows] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null); // ✅ 클릭된 이미지 상태
  const navigate = useNavigate();

  useEffect(() => {
    axios.get('/results')
      .then(res => {
        const failed = res.data.filter(r => !r.matched);
        setRows(failed);
      })
      .catch(e => {
        console.error('실패한 사진 조회 오류', e);
        alert('실패한 사진을 가져오는 중 오류가 발생했습니다.');
      });
  }, []);

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
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>
                      <img
                        src={r.image}
                        alt={`실패 ${i + 1}`}
                        className="row-image"
                        onClick={() => setSelectedImage(r.image)} // ✅ 클릭 시 설정
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
          <button className="reset-button" onClick={() => navigate(-1)}>
            뒤로가기
          </button>
        </div>
      </div>

      {/* ✅ 확대 이미지 오버레이 */}
      {selectedImage && (
        <div className="image-overlay" onClick={() => setSelectedImage(null)}>
          <div className="image-popup" onClick={(e) => e.stopPropagation()}>
            <button className="close-button" onClick={() => setSelectedImage(null)}>×</button>
            <img src={selectedImage} alt="확대 보기" className="enlarged-image" />
          </div>
        </div>
      )}
    </motion.div>
  );
}

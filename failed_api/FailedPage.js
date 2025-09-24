import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { motion } from 'framer-motion';
import './FailedPage.css';

// axios의 기본 URL을 백엔드 서버 주소로 설정합니다.
// index.js나 App.js에서 설정했다면 이 줄은 삭제해도 됩니다.
axios.defaults.baseURL = 'http://localhost:5000';

export default function FailedPage() {
  const [rows, setRows] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const navigate = useNavigate();

  // 서버에서 실패한 사진 목록을 가져오는 함수
  const fetchFailedImages = () => {
    // 서버의 '/results' API를 호출합니다.
    axios.get('/results')
      .then(res => {
        // 서버에서 받은 데이터 중 'matched'가 false인 항목만 필터링합니다.
        const failed = res.data.filter(r => !r.matched);
        setRows(failed);
      })
      .catch(e => {
        console.error('실패한 사진 조회 오류:', e);
        alert('실패한 사진을 가져오는 중 오류가 발생했습니다.');
      });
  };

  useEffect(() => {
    fetchFailedImages();
  }, []);

  // 사진 삭제 처리 함수
  const handleDelete = async (imageURL) => {
    if (!window.confirm('정말 이 사진을 삭제하시겠습니까?')) {
      return;
    }
    try {
      // 이미지 URL의 시작 부분에 있는 '/'를 제거하여 중복 경로를 방지합니다.
      const path = imageURL.startsWith('/') ? imageURL.slice(1) : imageURL;
      
      // 서버에 삭제 요청을 보냅니다.
      await axios.delete(`/image/${path}`); 

      alert('사진이 성공적으로 삭제되었습니다.');
      // 삭제 후 UI를 즉시 업데이트합니다.
      setRows(prevRows => prevRows.filter(row => row.image !== imageURL));
    } catch (error) {
      console.error('사진 삭제 오류:', error);
      alert('사진 삭제 중 오류가 발생했습니다. 서버의 API 설정을 확인하세요.');
    }
  };

  // CSV 다운로드 처리 함수
  const handleDownloadCSV = () => {
    if (rows.length === 0) {
      alert("저장할 데이터가 없습니다.");
      return;
    }

    const headers = ["연번", "이미지URL"];
    
    const csvContent = [
      headers.join(","),
      ...rows.map((r, i) => `${i + 1},"${r.image}"`)
    ].join("\n");

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    
    const link = document.createElement("a");
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute("href", url);
      link.setAttribute("download", "failed_images.csv");
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
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
                  <th>관리</th>
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
                        onClick={() => setSelectedImage(r.image)}
                      />
                    </td>
                    <td>
                      <div className="action-buttons">
                        <button onClick={() => handleDelete(r.image)} className="action-delete">삭제</button>
                      </div>
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
          <button className="download-button" onClick={handleDownloadCSV}>
            CSV 다운로드
          </button>
        </div>
      </div>

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

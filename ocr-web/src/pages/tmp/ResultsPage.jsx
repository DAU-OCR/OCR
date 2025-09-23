// src/pages/ResultsPage.js

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './ResultsPage.css';

export default function ResultsPage() {
  const [rows, setRows] = useState([]);
  const [downloading, setDownloading] = useState(false);

  // 결과 불러오기
  useEffect(() => {
    axios.get('/results')
      .then(res => {
        // 파일명 기준 정렬
        const sorted = res.data.sort((a, b) => a.image.localeCompare(b.image));
        setRows(sorted);
      })
      .catch(e => {
        console.error('ERROR GET /results', e);
        alert('결과를 가져오는 중 오류가 발생했습니다.');
      });
  }, []);

  // 엑셀 다운로드
 const handleDownload = async () => {
  try {
    const now = new Date();
    const yyyy = now.getFullYear();
    const mm = String(now.getMonth() + 1).padStart(2, '0');
    const dd = String(now.getDate()).padStart(2, '0');
    const filename = `${yyyy}-${mm}-${dd}_차량번호판.xlsx`;

    // (1) 사용자에게 저장 위치 먼저 요청
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
        if (e.name === 'AbortError') {
          console.log('사용자가 저장을 취소했습니다.');
          return;
        }
        throw e;
      }
    }

    // (2) 서버에 엑셀 요청
    setDownloading(true);
    const res = await axios.get('/download', { responseType: 'blob' });
    const blob = new Blob([res.data], {
      type: res.headers['content-type'],
    });

    // (3) 저장
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
    console.error('ERROR GET /download', e);
    alert('다운로드 중 오류가 발생했습니다.');
  } finally {
    setDownloading(false);
  }
  };


  // 결과 초기화
  const handleReset = async () => {
    if (!window.confirm('정말 결과를 초기화하시겠습니까?')) return;
    try {
      await axios.post('http://localhost:5000/reset'); // 정확한 Flask 포트
      setRows([]);
    } catch (e) {
      console.error('ERROR POST /reset', e);
      alert('초기화 중 오류가 발생했습니다.');
    }
  };

  return (
    <div className="results-page">
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
                      />
                    </td>
                    <td>{r.text1 || '-'}</td>
                    <td>{r.text2 || '-'}</td>
                    <td>{r.plate || '-'}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5" className="no-data">
                    데이터가 없습니다.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="actions">
          <button
            className="download-button"
            onClick={handleDownload}
            disabled={downloading}
          >
            {downloading ? '다운로드 중…' : (
              <>
                <img
                  src="./icons/download.png"
                  alt="다운로드"
                  className="download-icon"
                />
                <span>엑셀 파일 다운로드</span>
              </>
            )}
          </button>
          <button
            className="reset-button"
            onClick={handleReset}
          >
            초기화
          </button>
        </div>
      </div>
    </div>
  );
}
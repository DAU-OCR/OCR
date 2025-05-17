// src/pages/ResultsPage.js

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './ResultsPage.css';

export default function ResultsPage() {
  const [rows, setRows] = useState([]);
  const [downloading, setDownloading] = useState(false);

  // 결과 불러오기
  const fetchResults = () => {
    axios.get('/results')
      .then(res => setRows(res.data))
      .catch(e => {
        console.error('ERROR /results', e);
        alert('결과를 가져오는 중 오류가 발생했습니다.');
      });
  };

  useEffect(fetchResults, []);

  // 엑셀 다운로드 + 저장 위치/파일명 선택
  const handleDownload = async () => {
    setDownloading(true);
    try {
      // 백엔드에서 엑셀 blob 받아오기
      const res = await axios.get('/download', { responseType: 'blob' });
      const blob = new Blob([res.data], { type: res.headers['content-type'] });

      // 오늘 날짜 기반 기본 파일명
      const now = new Date();
      const yyyy = now.getFullYear();
      const mm   = String(now.getMonth() + 1).padStart(2, '0');
      const dd   = String(now.getDate()).padStart(2, '0');
      const defaultName = `${yyyy}-${mm}-${dd} 차량번호판`;

      // 사용자에게 파일명 입력 받기 (기본값: defaultName)
      const filename = window.prompt('저장할 파일명 (확장자 제외)', defaultName);
      if (!filename) {
        setDownloading(false);
        return;
      }

      // File System Access API 사용 가능 시
      if ('showSaveFilePicker' in window) {
        const opts = {
          suggestedName: `${filename}.xlsx`,
          types: [{
            description: 'Excel 파일',
            accept: {
              'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
            }
          }]
        };
        const handle = await window.showSaveFilePicker(opts);
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
      } else {
        // fallback: a 태그 다운로드
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${filename}.xlsx`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      }
    } catch (e) {
      console.error('ERROR /download', e);
      alert('다운로드 중 오류가 발생했습니다.');
    } finally {
      setDownloading(false);
    }
  };

  // 결과 초기화
  const handleReset = () => {
    if (!window.confirm('정말 결과를 초기화하시겠습니까?')) return;
    axios.post('/reset')
      .then(() => setRows([]))
      .catch(e => {
        console.error('ERROR /reset', e);
        alert('초기화 중 오류가 발생했습니다.');
      });
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
                <th>차량번호1</th>
                <th>차량번호2</th>
              </tr>
            </thead>
            <tbody>
              {rows.length > 0 ? rows.map((r, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>
                    <img
                      src={r.image}
                      alt={`차량 ${i + 1}`}
                      className="row-image"
                    />
                  </td>
                  <td>{r.raw}</td>
                  <td>{r.plate}</td>
                </tr>
              )) : (
                <tr>
                  <td colSpan="4" className="no-data">
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
            {downloading
              ? '다운로드 중…'
              : (
                <>
                  <img
                    src="/icons/download.png"
                    alt="다운로드"
                    className="download-icon"
                  />
                  <span>엑셀 파일 다운로드</span>
                </>
              )
            }
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

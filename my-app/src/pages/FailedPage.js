import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './FailedPage.css';

export default function FailedPage() {
  const [rows, setRows] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    axios.get('/results')
      .then(res => {
        // matched=false (실패)만 필터
        const failed = res.data.filter(r => !r.matched);
        setRows(failed);
      })
      .catch(e => {
        console.error(e);
        alert('실패한 사진을 가져오는 중 오류가 발생했습니다.');
      });
  }, []);

  return (
    <div className="failed-page">
      <div className="card">
        <h2 className="preview-title">실패한 사진</h2>

        <div className="preview-box">
          {rows.length > 0 ? (
            <table className="results-table">
              <thead>
                <tr>
                  <th>연번</th>
                  <th>차량사진</th>
                  <th>Raw 텍스트</th>
                  <th>Plate 텍스트</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>
                      <img
                        src={r.image}
                        alt={`실패 ${i+1}`}
                        className="row-image"
                      />
                    </td>
                    <td>{r.raw || '—'}</td>
                    <td>{r.plate || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="no-data">실패한 사진이 없습니다.</div>
          )}
        </div>

        <div className="actions">
          <button className="back-button" onClick={() => navigate(-1)}>
            뒤로가기
          </button>
        </div>
      </div>
    </div>
  );
}

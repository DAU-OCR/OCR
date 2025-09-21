import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './FailedPage.css';

export default function FailedPage() {
  const [rows, setRows] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    axios.get('http://localhost:5000/results')
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
    <div className="results-page failed-page">
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
                        src={`http://localhost:5000${r.image}`}
                        alt={`실패 ${i + 1}`}
                        className="row-image"
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
    </div>
  );
}
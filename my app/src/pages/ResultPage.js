import React, { useEffect, useState } from 'react';
import axios from 'axios';

export default function ResultsPage() {
  const [rows, setRows] = useState([]);

  useEffect(() => {
    axios.get('/results')
      .then(res => setRows(res.data))
      .catch(e => {
        console.error(e);
        alert('결과를 가져오는 중 오류가 발생했습니다.');
      });
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h1>결과 미리보기 (Excel)</h1>
      <table border="1" cellPadding="5">
        <thead>
          <tr>
            <th>Raw</th>
            <th>Plate</th>
            <th>Matched</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.raw}</td>
              <td>{r.plate}</td>
              <td>{r.matched ? '✅' : '❌'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 20 }}>
        <a href="/download">
          <button>엑셀 파일 다운로드</button>
        </a>
      </div>
    </div>
  );
}

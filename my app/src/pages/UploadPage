import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const navigate = useNavigate();

  const onFileChange = e => {
    const f = e.target.files[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const onUpload = async () => {
    if (!file) return alert('이미지를 선택하세요');
    const fd = new FormData();
    fd.append('image', file);
    try {
      await axios.post('/upload', fd);
      navigate('/results');
    } catch (e) {
      console.error(e);
      alert('서버 오류: ' + (e.response?.statusText || e.message));
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>이미지 업로드</h1>
      <input type="file" accept="image/*" onChange={onFileChange} />
      {preview && (
        <img
          src={preview}
          alt="preview"
          style={{ width: 300, marginTop: 20, display: 'block' }}
        />
      )}
      <div style={{ marginTop: 20 }}>
        <button onClick={onUpload}>파일 생성 시작</button>
      </div>
    </div>
  );
}

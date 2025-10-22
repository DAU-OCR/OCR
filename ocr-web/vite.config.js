import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    proxy: {
      '/results': 'http://localhost:5000',
      '/upload': 'http://localhost:5000',
      '/download': 'http://localhost:5000',
      '/reset': 'http://localhost:5000',
      '/update-plates': 'http://localhost:5000',
      '/uploads': 'http://localhost:5000',
    }
  }
})

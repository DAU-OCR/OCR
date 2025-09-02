const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1200, 
    height: 800,
  });

  // React 앱을 로드합니다.
  // 현재는 실행 중인 Vite 개발 서버 주소를 가리킵니다.
  const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process'); // Add this line

let pythonProcess = null; // To hold the reference to the Python process

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      // We might need a preload script later for IPC, but for now, let's keep it simple.
      // preload: path.join(__dirname, 'preload.js')
    }
  });

  // Load the React app.
  if (process.env.NODE_ENV === 'development') {
    // In development, load from Vite dev server
    mainWindow.loadURL('http://localhost:5173');
  } else {
    // In production, load the built React app from the local filesystem
    mainWindow.loadFile(path.join(__dirname, 'dist/index.html'));
  }

  // Optional: Open DevTools for debugging.
  // mainWindow.webContents.openDevTools();
}

// Function to start the Python backend
function startPythonBackend() {
  // Path to the packaged Python executable
  // electron-builder's extraFiles puts them at the root of the packaged app.
  const serverExecutable = path.join(app.getAppPath(), 'ocr_server.exe');

  console.log(`Attempting to start backend from: ${serverExecutable}`);

  pythonProcess = spawn(serverExecutable, [], {
    stdio: 'inherit' // This will pipe stdout/stderr to the main Electron process console
  });

  pythonProcess.on('error', (err) => {
    console.error('Failed to start Python backend:', err);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python backend exited with code ${code}`);
  });
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
app.whenReady().then(() => {
  createWindow(); // Call createWindow first
  startPythonBackend(); // Then start the backend

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Quit when all windows are closed, except on macOS.
app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

// Kill the Python process when the Electron app is about to quit
app.on('will-quit', () => {
  if (pythonProcess) {
    console.log('Killing Python backend...');
    pythonProcess.kill(); // Send SIGTERM
    pythonProcess = null;
  }
});

  // (선택사항) 개발자 도구를 엽니다.
  // mainWindow.webContents.openDevTools();
}

// Electron이 준비되면 창을 생성합니다.
app.whenReady().then(createWindow);

// 모든 창이 닫혔을 때 앱을 종료합니다. (macOS 제외)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// 앱이 활성화되었을 때 창이 없으면 새로 생성합니다. (macOS)
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

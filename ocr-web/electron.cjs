const { app, BrowserWindow, Menu, MenuItem } = require('electron');
const path = require('path');
const { spawn, exec, execSync } = require('child_process');
const fs = require('fs');

let pythonProcess = null;

// Function to determine the correct path to the backend executable
function getBackendPath() {
  // In a packaged app, the executable is next to the main app executable.
  if (app.isPackaged) {
    return path.join(path.dirname(app.getPath('exe')), 'ocr_server.exe');
  }
  // In development, we point to the file in the server's dist folder.
  return path.join(__dirname, '..', 'server', 'dist', 'ocr_server.exe');
}

// Function to start the Python backend
function startPythonBackend() {
  const serverExecutable = getBackendPath();
  const serverDir = path.dirname(serverExecutable);

  pythonProcess = spawn(serverExecutable, [], {
    windowsHide: true,
    cwd: serverDir
    // stdio: 'ignore' // Remove this to capture output
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
  });
}

  function createWindow() {
    const mainWindow = new BrowserWindow({
      width: 1200,
      height: 800,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
      }
    });

    // 개발자 도구 메뉴 추가
    const menu = Menu.buildFromTemplate([
      {
        label: '개발자',
        submenu: [
          {
            label: '개발자 도구 토글',
            accelerator: process.platform === 'darwin' ? 'Alt+Command+I' : 'Ctrl+Shift+I',
            click (item, focusedWindow) {
              if (focusedWindow) focusedWindow.webContents.toggleDevTools();
            }
          }
        ]
      }
    ]);
    Menu.setApplicationMenu(menu);

    // Load the React app.
    const startUrl = app.isPackaged
      ? `file://${path.join(__dirname, 'dist/index.html')}`
      // In development, load from Vite dev server
      : 'http://localhost:5173';
    
    mainWindow.loadURL(startUrl);
  }
  app.whenReady().then(() => {
    startPythonBackend();
    createWindow();

    app.on('activate', function () {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
  });

  // Make sure to kill the backend process when the app quits.
app.on('will-quit', () => {
  if (pythonProcess) {
    // Windows에서 강제 종료를 위해 taskkill 사용
    if (process.platform === 'win32') {
      try {
        execSync(`taskkill /pid ${pythonProcess.pid} /f /t`);
      } catch (err) {
        // Ignore errors, we are quitting anyway
      }
    } else {
      pythonProcess.kill();
    }
    pythonProcess = null;
  }
});
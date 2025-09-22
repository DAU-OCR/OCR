const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

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

  console.log(`Attempting to start backend from: ${serverExecutable}`);

  pythonProcess = spawn(serverExecutable, [], {
    stdio: 'inherit', // Pipe stdout/stderr to the main Electron process console for debugging
    windowsHide: true
  });

  pythonProcess.on('error', (err) => {
    console.error('Failed to start Python backend:', err);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python backend exited with code ${code}`);
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
    console.log('Killing Python backend...');
    pythonProcess.kill();
    pythonProcess = null;
  }
});
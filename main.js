const { app, BrowserWindow } = require('electron')

//for dual graphics user
app.commandLine.appendSwitch('force_high_performance_gpu', '1')
app.commandLine.appendSwitch('no-sandbox')

function createWindow() {
  const win = new BrowserWindow({
    
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
    },
  })

  win.loadURL('http://127.0.0.1:5000/'); // Update with your Flask app URL

  // Open the DevTools.
  // win.webContents.openDevTools();

  win.on('closed', () => {
    win.destroy();
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

D:\Projects\OCR\ocr-web>npm run electron:build

> ocr-web@0.0.0 electron:build
> npm run build && electron-builder


> ocr-web@0.0.0 build
> vite build

vite v6.4.0 building for production...
✓ 492 modules transformed.
dist/index.html                     0.50 kB │ gzip:   0.33 kB
dist/assets/favicon-Dks8GlWI.svg    5.34 kB │ gzip:   2.30 kB
dist/assets/index-CIySghHh.css      7.63 kB │ gzip:   2.06 kB
dist/assets/index-Cv6ZjnpD.js     391.09 kB │ gzip: 128.41 kB
✓ built in 1.15s
  • electron-builder  version=26.0.12 os=10.0.26100
  • loaded configuration  file=package.json ("build" field)
  • description is missed in the package.json  appPackageFile=D:\Projects\OCR\ocr-web\package.json
  • author is missed in the package.json  appPackageFile=D:\Projects\OCR\ocr-web\package.json
  • writing effective config  file=release\builder-effective-config.yaml
  • executing @electron/rebuild  electronVersion=30.5.1 arch=x64 buildFromSource=false appDir=./
  • installing native dependencies  arch=x64
  • completed installing native dependencies
  • packaging       platform=win32 arch=x64 electron=30.5.1 appOutDir=release\win-unpacked
  • downloading     url=https://github.com/electron/electron/releases/download/v30.5.1/electron-v30.5.1-win32-x64.zip size=109 MB parts=8
  • downloaded      url=https://github.com/electron/electron/releases/download/v30.5.1/electron-v30.5.1-win32-x64.zip duration=2.404s
  • updating asar integrity executable resource  executablePath=release\win-unpacked\ocr-web.exe
  • signing with signtool.exe  path=release\win-unpacked\ocr-web.exe
  • signing with signtool.exe  path=release\win-unpacked\ocr_server.exe
  • building        target=nsis file=release\ocr-web Setup 0.0.0.exe archs=x64 oneClick=true perMachine=false
  • signing with signtool.exe  path=release\win-unpacked\resources\elevate.exe
  • signing with signtool.exe  path=release\__uninstaller-nsis-ocr-web.exe
  ⨯ C:\Users\ygchun123\AppData\Local\electron-builder\Cache\nsis\nsis-3.0.4.1\Bin\makensis.exe process failed ERR_ELECTRON_BUILDER_CANNOT_EXECUTE
Exit code:
1
Output:
Command line defined: "APP_ID=com.yourcompany.ocrapp"
Command line defined: "APP_GUID=90f776fb-767d-5536-b63c-7be3afc1ca8e"
Command line defined: "UNINSTALL_APP_KEY=90f776fb-767d-5536-b63c-7be3afc1ca8e"
Command line defined: "PRODUCT_NAME=ocr-web"
Command line defined: "PRODUCT_FILENAME=ocr-web"
Command line defined: "APP_FILENAME=ocr-web"
Command line defined: "APP_DESCRIPTION="
Command line defined: "VERSION=0.0.0"
Command line defined: "PROJECT_DIR=D:\Projects\OCR\ocr-web"
Command line defined: "BUILD_RESOURCES_DIR=D:\Projects\OCR\ocr-web\assets"
Command line defined: "APP_PACKAGE_NAME=ocr-web"
Command line defined: "MUI_ICON=D:\Projects\OCR\ocr-web\public\icons\logo.ico"
Command line defined: "MUI_UNICON=D:\Projects\OCR\ocr-web\public\icons\logo.ico"
Command line defined: "APP_64=D:\Projects\OCR\ocr-web\release\ocr-web-0.0.0-x64.nsis.7z"
Command line defined: "APP_64_NAME=ocr-web-0.0.0-x64.nsis.7z"
Command line defined: "APP_64_HASH=E9F18F1D50295CF0A6CB61A528325A8C24EAB7A8C54929F9A8000B2BE895529E99EEF794049F4C6949DF22AAAFA53A2FCA164856071997FCBACC77BFA79619D2"
Command line defined: "APP_64_UNPACKED_SIZE=2805135"
Command line defined: "APP_INSTALLER_STORE_FILE=ocr-web-updater\installer.exe"
Command line defined: "COMPRESSION_METHOD=7z"
Command line defined: "ONE_CLICK"
Command line defined: "RUN_AFTER_FINISH"
Command line defined: "SHORTCUT_NAME=ocr-web"
Command line defined: "UNINSTALL_DISPLAY_NAME=ocr-web 0.0.0"
Command line defined: "ESTIMATED_SIZE=2805134"
Command line defined: "COMPRESS=auto"
Command line defined: "UNINSTALLER_OUT_FILE=D:\Projects\OCR\ocr-web\release\__uninstaller-nsis-ocr-web.exe"
Processing config: C:\Users\ygchun123\AppData\Local\electron-builder\Cache\nsis\nsis-3.0.4.1\nsisconf.nsh
Processing script file: "<stdin>" (UTF8)

Error output:
File: failed creating mmap of "D:\Projects\OCR\ocr-web\release\ocr-web-0.0.0-x64.nsis.7z"
Error in macro x64_app_files on macroline 1
Error in macro compute_files_for_current_arch on macroline 7
Error in macro extractEmbeddedAppPackage on macroline 8
Error in macro installApplicationFiles on macroline 79
!include: error in script: "installSection.nsh" on line 66
Error in script "<stdin>" on line 191 -- aborting creation process
  failedTask=build stackTrace=Error: C:\Users\ygchun123\AppData\Local\electron-builder\Cache\nsis\nsis-3.0.4.1\Bin\makensis.exe process failed ERR_ELECTRON_BUILDER_CANNOT_EXECUTE
Exit code:
1
Output:
Command line defined: "APP_ID=com.yourcompany.ocrapp"
Command line defined: "APP_GUID=90f776fb-767d-5536-b63c-7be3afc1ca8e"
Command line defined: "UNINSTALL_APP_KEY=90f776fb-767d-5536-b63c-7be3afc1ca8e"
Command line defined: "PRODUCT_NAME=ocr-web"
Command line defined: "PRODUCT_FILENAME=ocr-web"
Command line defined: "APP_FILENAME=ocr-web"
Command line defined: "APP_DESCRIPTION="
Command line defined: "VERSION=0.0.0"
Command line defined: "PROJECT_DIR=D:\Projects\OCR\ocr-web"
Command line defined: "BUILD_RESOURCES_DIR=D:\Projects\OCR\ocr-web\assets"
Command line defined: "APP_PACKAGE_NAME=ocr-web"
Command line defined: "MUI_ICON=D:\Projects\OCR\ocr-web\public\icons\logo.ico"
Command line defined: "MUI_UNICON=D:\Projects\OCR\ocr-web\public\icons\logo.ico"
Command line defined: "APP_64=D:\Projects\OCR\ocr-web\release\ocr-web-0.0.0-x64.nsis.7z"
Command line defined: "APP_64_NAME=ocr-web-0.0.0-x64.nsis.7z"
Command line defined: "APP_64_HASH=E9F18F1D50295CF0A6CB61A528325A8C24EAB7A8C54929F9A8000B2BE895529E99EEF794049F4C6949DF22AAAFA53A2FCA164856071997FCBACC77BFA79619D2"
Command line defined: "APP_64_UNPACKED_SIZE=2805135"
Command line defined: "APP_INSTALLER_STORE_FILE=ocr-web-updater\installer.exe"
Command line defined: "COMPRESSION_METHOD=7z"
Command line defined: "ONE_CLICK"
Command line defined: "RUN_AFTER_FINISH"
Command line defined: "SHORTCUT_NAME=ocr-web"
Command line defined: "UNINSTALL_DISPLAY_NAME=ocr-web 0.0.0"
Command line defined: "ESTIMATED_SIZE=2805134"
Command line defined: "COMPRESS=auto"
Command line defined: "UNINSTALLER_OUT_FILE=D:\Projects\OCR\ocr-web\release\__uninstaller-nsis-ocr-web.exe"
Processing config: C:\Users\ygchun123\AppData\Local\electron-builder\Cache\nsis\nsis-3.0.4.1\nsisconf.nsh
Processing script file: "<stdin>" (UTF8)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  Error output:
File: failed creating mmap of "D:\Projects\OCR\ocr-web\release\ocr-web-0.0.0-x64.nsis.7z"
Error in macro x64_app_files on macroline 1
Error in macro compute_files_for_current_arch on macroline 7
Error in macro extractEmbeddedAppPackage on macroline 8
Error in macro installApplicationFiles on macroline 79
!include: error in script: "installSection.nsh" on line 66
Error in script "<stdin>" on line 191 -- aborting creation process
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      at ChildProcess.<anonymous> (D:\Projects\OCR\ocr-web\node_modules\builder-util\src\util.ts:259:14)
    at Object.onceWrapper (node:events:633:26)
    at ChildProcess.emit (node:events:518:28)
    at ChildProcess.cp.emit (D:\Projects\OCR\ocr-web\node_modules\cross-spawn\lib\enoent.js:34:29)
    at maybeClose (node:internal/child_process:1101:16)
    at Process.ChildProcess._handle.onexit (node:internal/child_process:304:5)

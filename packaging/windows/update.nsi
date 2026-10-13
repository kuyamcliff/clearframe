; ClipFX Enhance app-only update.
;
; A small installer that replaces just the application in an existing installation
; (FFmpeg and other resources stay as they are), so updates download a few MB instead of
; the full installer. The app runs it as:  update.exe /S /D=<install folder>
;
; Build:  makensis -DVERSION=1.1.0 -DAPP_EXE=path\to\ClipFXEnhance.exe -DOUT=ClipFX-Enhance-1.1.0-x64-update.exe update.nsi

Unicode true
RequestExecutionLevel admin
SetCompressor /SOLID lzma

!ifndef VERSION
  !error "pass -DVERSION=x.y.z"
!endif
!ifndef APP_EXE
  !error "pass -DAPP_EXE=path to ClipFXEnhance.exe"
!endif
!ifndef OUT
  !define OUT "ClipFX-Enhance-${VERSION}-x64-update.exe"
!endif

Name "ClipFX Enhance ${VERSION} update"
OutFile "${OUT}"
InstallDir "$PROGRAMFILES64\ClipFX Enhance"
ShowInstDetails show
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "ClipFX Enhance"
VIAddVersionKey "FileDescription" "ClipFX Enhance update"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "MIT License"

Page instfiles

Section "Update"
  ; Only ever update an existing installation.
  IfFileExists "$INSTDIR\ClipFXEnhance.exe" +3 0
    MessageBox MB_ICONSTOP|MB_OK "ClipFX Enhance was not found in $INSTDIR. Download the full installer instead." /SD IDOK
    Abort

  ; Wait (up to a minute) for the running app to close, keeping the old copy until the
  ; new one is in place so a failed update leaves a working app.
  StrCpy $0 0
  wait:
    ClearErrors
    Delete "$INSTDIR\ClipFXEnhance.exe.old"
    Rename "$INSTDIR\ClipFXEnhance.exe" "$INSTDIR\ClipFXEnhance.exe.old"
    IfErrors 0 replace
    IntOp $0 $0 + 1
    IntCmp $0 120 timeout
    Sleep 500
    Goto wait
  timeout:
    MessageBox MB_ICONSTOP|MB_OK "Please close ClipFX Enhance and run the update again." /SD IDOK
    Abort

  replace:
    SetOutPath "$INSTDIR"
    ClearErrors
    File "/oname=ClipFXEnhance.exe" "${APP_EXE}"
    IfErrors 0 done
      ; Roll back to the previous version.
      Rename "$INSTDIR\ClipFXEnhance.exe.old" "$INSTDIR\ClipFXEnhance.exe"
      MessageBox MB_ICONSTOP|MB_OK "The update could not be written. Your previous version was kept." /SD IDOK
      Abort
  done:
    Delete /REBOOTOK "$INSTDIR\ClipFXEnhance.exe.old"
    ; Keep "Apps & features" showing the right version.
    ReadRegStr $1 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClipFX Enhance" "DisplayName"
    StrCmp $1 "" +2 0
      WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClipFX Enhance" "DisplayVersion" "${VERSION}"
    ; Start the new version as the signed-in user, not as administrator.
    Exec '"$WINDIR\explorer.exe" "$INSTDIR\ClipFXEnhance.exe"'
SectionEnd

; ClipFX Enhance NSIS installer hooks (included by Tauri's installer template).
;
; What the installer owns: the program folder under Program Files, Start menu
; and desktop shortcuts, file associations and the uninstall registry entry.
;
; What it never touches unless the user explicitly ticks "Delete the
; application data" in the uninstaller:
;   %LOCALAPPDATA%\ClipFX Enhance   settings, database, logs, default models and cache
;   %APPDATA%\ClipFX Enhance        (reserved for future roaming settings)
;
; Upgrades run the previous uninstaller with /UPDATE, which never deletes data.
; Silent uninstalls (/S) leave the checkbox unticked and therefore keep data.
; Models or cache folders the user moved elsewhere (Preferences > Storage) and
; exported videos and projects in the user's own folders are never deleted by
; the uninstaller, even when the box is ticked.

!macro NSIS_HOOK_PREINSTALL
  ; Running exports are stopped by Tauri's CheckIfAppIsRunning prompt (the
  ; processing workers are ClipFXEnhance.exe processes). Unfinished exports resume
  ; from their last completed segment after the upgrade.
!macroend

!macro NSIS_HOOK_POSTINSTALL
  ; Nothing to do: all per-user folders are created on first launch, in the
  ; context of the user who runs ClipFX Enhance (never the elevated installer).
!macroend

!macro NSIS_HOOK_PREUNINSTALL
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $DeleteAppDataCheckboxState = 1
  ${AndIf} $UpdateMode <> 1
    SetShellVarContext current
    RMDir /r "$LOCALAPPDATA\ClipFX Enhance"
    RMDir /r "$APPDATA\ClipFX Enhance"
  ${EndIf}
!macroend

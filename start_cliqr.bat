set "MF=%USERPROFILE%\AppData\Local\miniforge3"
call "%MF%\Scripts\activate.bat" "%MF%"
call conda activate cliqr
cd /d "%SUERPROFILE%\cliqr"
jupyter-lab
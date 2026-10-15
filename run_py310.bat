@echo off
REM 用 Python 3.10 运行指定脚本 (PowerShell 的 python 指向 msys64, 没装 pip)
set PY310=C:\Users\86139\AppData\Local\Programs\Python\Python310\python.exe
%PY310% %*

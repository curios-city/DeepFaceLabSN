@echo off

CD /D %~dp0
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" videoed cut-video ^
       --input-file %1

pause
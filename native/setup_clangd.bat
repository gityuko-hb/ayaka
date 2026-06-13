@echo off
REM Generate compile_flags.txt with project includes and CUDA path (if available)
setlocal enabledelayedexpansion
set "OUTFILE=%~dp0compile_flags.txt"
echo Generating %OUTFILE%...

> "%OUTFILE%" (
    echo -Iinclude
    echo -Isrc
    echo -Icsrc
    echo -Istubs
)
if defined CUDA_PATH (
    set "CUDA_INC=%CUDA_PATH:\=/%/include"
    set "CUDA_ROOT=%CUDA_PATH:\=/%"
    >> "%OUTFILE%" echo -I!CUDA_INC!
    >> "%OUTFILE%" echo --cuda-path=!CUDA_ROOT!
    echo CUDA include: !CUDA_INC!
    echo CUDA root:    !CUDA_ROOT!
) else (
    echo WARNING: CUDA_PATH not set. CUDA headers may not be found.
)
echo Done.

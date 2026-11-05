@echo off
setlocal

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 run_dev.py --install-frontend-deps --api-host 127.0.0.1 --api-port 8000 --front-host 127.0.0.1 --front-port 3000
) else (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 (
    python run_dev.py --install-frontend-deps --api-host 127.0.0.1 --api-port 8000 --front-host 127.0.0.1 --front-port 3000
  ) else (
    echo [HATA] Python bulunamadi.
    pause
    exit /b 1
  )
)

if %ERRORLEVEL% neq 0 (
  echo.
  echo [HATA] run_dev.py baslatilamadi.
  echo Python kurulu degilse veya bagimliliklar ile ilgili bir sorun olabilir.
  pause
)

endlocal

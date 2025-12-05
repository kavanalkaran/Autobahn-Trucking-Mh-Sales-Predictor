@echo off
REM Go to the folder where the app.py file is located
cd /d "C:\Users\Training\OneDrive - Autobahn Trucking Corporation - Doors Dubai\Desktop\mh_sales_forecast"

REM Run the Streamlit app (python -m ensures it works even if 'streamlit' is not on PATH)
python -m streamlit run "app.py"

echo.
echo Streamlit has stopped. Press any key to close this window...
pause >nul

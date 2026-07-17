@echo off
rem Serve the repo root so the app can reach ../../artifacts, and open the page.
rem Run dist\ instead if you want what actually deploys: python -m http.server 8201 --directory dist
cd /d "%~dp0"
start "" http://localhost:8200/web/app/
python -m http.server 8200

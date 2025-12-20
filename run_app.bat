:: project_root/run_app.bat
@echo off
:: バッチファイルのあるディレクトリにカレントディレクトリを移動
cd /d %~dp0

echo ========================================================
echo   ScholarScope 文献探索システム (Portable Mode)
echo ========================================================

:: --- ポータブル環境設定エリア ---

:: 1. キャッシュディレクトリをプロジェクト内に固定 (C:\Users\... を汚さない)
set TEMP_DIR=%~dp0temp
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

set HF_HOME=%TEMP_DIR%\hf_cache
set SENTENCE_TRANSFORMERS_HOME=%TEMP_DIR%\st_cache

:: 2. Pythonライブラリのオフラインモード強制
:: (モデルが見つからない場合に勝手にダウンロードしに行くのを防ぐ)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

:: 3. Ollama設定 (ユーザー環境のOllamaを使うため、ここはデフォルトのまま)
:: もしポートを変更したい場合は以下を有効化
:: set OLLAMA_HOST=127.0.0.1:11434

echo 環境設定完了:
echo   Cache Dir : %TEMP_DIR%
echo   Offline   : ON
echo.

echo 仮想環境 .venv をアクティブ化します...
CALL .\.venv\Scripts\activate.bat

echo Streamlitアプリケーションを起動します...
streamlit run src/app.py --server.fileWatcherType none

echo.
echo アプリケーションが終了しました。
pause
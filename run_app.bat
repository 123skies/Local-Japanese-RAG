:: project_root/run_app.bat
@echo off
setlocal
cd /d %~dp0

echo ========================================================
echo   ScholarScope: 文献探索支援システム
echo ========================================================

:: --- 環境設定エリア ---

:: 1. キャッシュディレクトリをプロジェクト内に固定 (ポータブル性の維持)
:: ユーザーのCドライブを汚さず、モデルをプロジェクト内に保存します。
set "TEMP_DIR=%~dp0temp"
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"
set "HF_HOME=%TEMP_DIR%\hf_cache"
set "SENTENCE_TRANSFORMERS_HOME=%TEMP_DIR%\st_cache"

:: 2. オフライン設定の変更
:: 公開用では「初回ダウンロード」が必要なため、OFFLINE=1 は削除または 0 にします。
set HF_HUB_OFFLINE=0
set TRANSFORMERS_OFFLINE=0

:: 3. Python環境の確認
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] 仮想環境 .venv が見つかりません。
    echo 以下のコマンドでセットアップを行ってください:
    echo python -m venv .venv
    echo .venv\Scripts\activate
    echo pip install -r requirements.txt
    pause
    exit /b
)

echo 仮想環境を起動しています...
call .venv\Scripts\activate.bat

:: 4. Ollamaの起動確認（オプション: メッセージのみ）
echo [INFO] Ollamaが起動していることを確認してください。
echo (モデル: qwen2.5 などがインストールされている必要があります)

:: 5. アプリケーションの起動
:: 以下のように引数でワークスペースの場所を指定することもできます。
:: streamlit run src/app.py --server.fileWatcherType none -- --workspace "C:\my_workspace"
echo Streamlitを起動します...
:: --server.fileWatcherType none はリソース節約のため維持
streamlit run src/app.py --server.fileWatcherType none

echo.
echo アプリケーションが終了しました。
pause
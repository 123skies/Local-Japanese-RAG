:: run_app.bat
@echo off
echo 仮想環境 .venv をアクティブ化します...

:: 仮想環境のアクティブ化
CALL .\.venv\Scripts\activate.bat

echo Streamlitアプリケーションを起動します...

:: Streamlitアプリケーションの実行

:: Workspaceは以下のように引数でも指定可
:: streamlit run src/app.py --server.fileWatcherType none -- --workspace "C:\my_workspace"
streamlit run src/app.py --server.fileWatcherType none

echo Streamlitサーバーが起動しました。終了するにはコマンドプロンプトでCtrl+Cを押してください。
echo 通常、自動的にブラウザが開きますが、開かない場合は http://localhost:8501 にアクセスしてください。

pause
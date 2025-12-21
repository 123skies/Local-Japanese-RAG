# ▼▼▼▼▼ モンキーパッチ ▼▼▼▼▼
import os
import sys

# 【重要】StreamlitとPyTorchの競合エラーを回避するパッチ
# Streamlitが torch.classes をフォルダとしてスキャンしようとしてクラッシュするのを防ぎます。
# 他のモジュール（engineなど）が読み込まれる前に、ここで確実に無効化します。
import torch
if hasattr(torch, 'classes'):
    torch.classes.__path__ = []

# ▲▲▲▲▲ ここまで ▲▲▲▲▲

# project_root/src/app.py
import streamlit as st
import os
import time
import datetime
import json
import tiktoken
from modules import highlight
import logging
import logging.handlers
from engine import ScholarScopeEngine
from modules.date_standardizer import DateStandardizer
from pathlib import Path
import re
import argparse

# --- ロガーの取得のみを先に行う ---
logger = logging.getLogger('scholarscope_lite')

def setup_logging(engine: ScholarScopeEngine):
    if logger.hasHandlers():
        return

    log_file_path = engine.log_file_path  # エンジンから正しいログファイルパスを取得

    # --- config.tomlからログレベルを取得 ---
    console_level_str = engine.console_log_level
    file_level_str = engine.file_log_level
    # 文字列をloggingレベルに変換 (不正な値の場合はデフォルト値を使用)
    console_level = getattr(logging, console_level_str, logging.INFO)
    file_level = getattr(logging, file_level_str, logging.WARNING)

    logger.propagate = False
    # ロガー自体のレベルは最も低いレベルに設定し、ハンドラでフィルタリングする
    logger.setLevel(min(console_level, file_level, logging.INFO))
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(file_level)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_level)
    logger.addHandler(stream_handler)

    logger.info("--- Application Logger Initialized (PID: %s) ---", os.getpid())
    logger.info(f"ログは '{log_file_path}' に出力されます。(ファイルレベル: {file_level_str})")
    logger.info(f"コンソールログレベル: {console_level_str}")

# --- ページ設定 ---
st.set_page_config(page_title="ScholarScope", layout="wide")

# ### 変更 ###: ハードコーディングされたプロンプトを削除
# QA_SYSTEM_PROMPT = """..."""
# QGEN_SYSTEM_PROMPT = """..."""

# --- セッションステート初期化 ---
def initialize_session_state():
    if 'initialized' not in st.session_state:
        default_session_state = {
            'engine': None,
            'initialized': False,
            'uploaded_file_info_list': [],
            'executed_and_query_for_highlight': "",
            'executed_bm25_query_for_highlight': "",
            'executed_semantic_query_for_highlight': "",
            'last_executed_must_keywords': "",
            'last_executed_semantic_query': "",
            'last_executed_full_text_query': "",
            'last_executed_doc_name_filter': "",
            'and_search_results': [],
            'keyword_search_results': [],
            'vector_search_results': [],
            'reranked_search_results': [],
            'max_context_tokens_for_prompt': 7000, # あとでengineから上書き
            'show_ai_question_pane': True,
            'search_result_meta': {}, # 追加: 検索結果のメタ情報（has_moreなど）
            'current_ai_question_text': "",
            'initialization_status': "pending",
            'rebuild_decision_made': False,
            'user_chose_to_rebuild': None,
            'user_chose_to_incrementally_update': None,
            'contexts_for_prompt_docs_precomputed': [],
            'contexts_string_precomputed': "",
            'current_total_tokens_precomputed': 0,
            'last_used_max_context_tokens_for_prompt_in_contexts_string': 7000, # あとでengineから上書き
            'last_query_for_precompute_v2': None,
            'prompt_display_text_right_pane': "",
            'current_prompt_type': None,
            'llm_response': "",
            'user_prompt_for_ollama': "",
            'citation_map': {},
            'is_streaming': False,
            'ui_state': 'idle',  # idle, streaming_answer, getting_citations, done
            'prompt_citation_map': {},
            'clean_answer': "",
            'final_answer': "",
            'contexts_for_citation': "",
            'system_prompt_for_ollama': "",
            'skip_rerank': False,
            'is_deep_search_mode': False,
            'search_request': None,
            'executed_optimized_bm25_query': None,
            'last_search_duration': 0.0,
            'is_searching': False,
            'search_was_cancelled': False, # 追加: 検索中断フラグ
        }
        for key, value in default_session_state.items():
            if key not in st.session_state: st.session_state[key] = value

initialize_session_state()

def reset_application_state():
    """アプリケーションの状態を初期化（入力クリア＆結果クリア）"""
    # 検索結果のクリア
    st.session_state.and_search_results = []
    st.session_state.keyword_search_results = []
    st.session_state.vector_search_results = []
    st.session_state.reranked_search_results = []
    st.session_state.search_was_cancelled = False
    st.session_state.search_result_meta = {}
    st.session_state.llm_response = ""
    st.session_state.clean_answer = ""
    st.session_state.final_answer = ""
    st.session_state.prompt_display_text_right_pane = ""
    
    # 履歴変数のクリア
    st.session_state.last_executed_must_keywords = ""
    st.session_state.last_executed_semantic_query = ""
    st.session_state.last_executed_full_text_query = ""
    st.session_state.last_executed_doc_name_filter = ""
    st.session_state.current_ai_question_text = ""
    
    # 入力フォームのクリア（session_stateのキー経由）
    keys_to_clear = ["input_full_text", "input_doc_name_filter", "input_semantic_query"]
    for k in keys_to_clear:
        if k in st.session_state:
            st.session_state[k] = ""
    
    # UI状態のリセット
    st.session_state.reranked_results_expanded_state = None
    st.session_state.and_results_expanded_state = None
    st.session_state.bm25_results_expanded_state = None
    st.session_state.vector_results_expanded_state = None
    st.session_state.show_ai_question_pane = True
    st.session_state.ui_state = 'idle'

    # 最適化・ハイライト情報のクリア
    st.session_state.executed_optimized_bm25_query = None
    st.session_state.executed_rerank_query = None
    st.session_state.executed_bm25_query_for_highlight = ""
    st.session_state.executed_semantic_query_for_highlight = ""
    st.session_state.executed_and_query_for_highlight = ""

    logger.info("ユーザー操作により条件と結果を初期化しました。")

def save_search_history(history_entry: dict):
    search_history_file = st.session_state.engine.search_history_path
    try:
        with open(search_history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(history_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"検索履歴の保存中にエラー: {e}", exc_info=True)

def render_right_column():
    with right_column:
        col_header_right, col_close_btn_right = st.columns([0.9, 0.1])
        with col_header_right: st.subheader("検索結果を使ったAI分析")
        with col_close_btn_right:
            if st.button("✖️", key="close_ai_pane_button_right_key", help="AI質問パネルを隠す", use_container_width=True):
                st.session_state.show_ai_question_pane = False; logger.debug("AI質問パネルを非表示にしました。"); st.rerun()

        # 質問入力エリア
        edited_question = st.text_area(
            "質問・関心事 (レポート生成用):",
            value=st.session_state.current_ai_question_text,
            key="ai_question_text_area_right_key_unique",
            height=150,
            help="「レポート生成」で使用します。検索結果を元に、ここに入力した質問に沿ったレポートをAIが生成します。"
        )
        if edited_question != st.session_state.current_ai_question_text:
            logger.debug(f"AI質問文が編集されました。新: '{edited_question[:50]}...'")
        st.session_state.current_ai_question_text = edited_question

        # アクションボタン
        any_search_results = st.session_state.reranked_search_results or \
                             st.session_state.and_search_results or \
                             st.session_state.keyword_search_results or \
                             st.session_state.vector_search_results
        action_buttons_enabled = any_search_results and st.session_state.current_ai_question_text.strip()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 レポートを生成", key="generate_ai_answer_button", help="検索結果と上の質問文を元に、詳細なレポートを生成します。", disabled=not action_buttons_enabled, use_container_width=True, type="primary"):
                logger.info("「レポート生成」ボタンが押されました。")
                st.session_state.ui_state = 'streaming_answer'
                st.session_state.clean_answer = ""
                st.session_state.final_answer = ""
                st.session_state.llm_response = "" # 旧レスポンスをクリア

                with st.spinner("AIへの質問準備中..."):
                    # リランキングスキップ時はbm25とvectorの結果もコンテキスト生成に含める
                    search_results_for_prompt = {
                        "reranked": st.session_state.reranked_search_results,
                        "and": st.session_state.and_search_results,
                        "bm25": st.session_state.keyword_search_results,
                        "vector": st.session_state.vector_search_results
                    }
                    prompt_data = st.session_state.engine.generate_prompt_for_llm(
                        search_results_dict=search_results_for_prompt,
                        question_text=st.session_state.current_ai_question_text,
                        max_tokens=st.session_state.max_context_tokens_for_prompt
                    )
                    st.session_state.contexts_for_citation = prompt_data["contexts_string"]
                    st.session_state.prompt_citation_map = prompt_data["citation_map"]
                    user_prompt_content = f"""**提供されたコンテキスト:**\n{prompt_data["contexts_string"]}\n\n---\n**ユーザーの質問・関心:**\n{st.session_state.current_ai_question_text}"""
                    st.session_state.user_prompt_for_ollama = user_prompt_content
                    # system_promptはエンジン側で固定されるため、ここでは設定不要
                    st.session_state.prompt_display_text_right_pane = f"{st.session_state.engine.qa_system_prompt}\n---\n{user_prompt_content}"

                st.rerun()

        with col2:
            if st.button("🔍 論点/トピックを抽出", key="extract_points_button", help="検索結果を元に、調査可能な新たな論点やトピックをAIが提案します。上の質問文は使用しません。", disabled=not any_search_results, use_container_width=True):
                logger.info("「論点/トピックを抽出」ボタンが押されました。")
                st.session_state.ui_state = 'idle'
                st.session_state.llm_response = ""

                with st.spinner("プロンプト用コンテキストを生成中..."):
                    search_results_for_prompt = {
                        "reranked": st.session_state.reranked_search_results,
                        "and": st.session_state.and_search_results,
                        "bm25": st.session_state.keyword_search_results,
                        "vector": st.session_state.vector_search_results
                    }
                    prompt_data = st.session_state.engine.generate_prompt_for_llm(
                        search_results_dict=search_results_for_prompt, question_text="",
                        max_tokens=st.session_state.max_context_tokens_for_prompt
                    )
                st.session_state.user_prompt_for_ollama = f"""**提供されたコンテキスト:**\n{prompt_data["contexts_string"]}"""
                st.session_state.system_prompt_for_ollama = st.session_state.engine.qgen_system_prompt
                st.session_state.prompt_display_text_right_pane = f"{st.session_state.system_prompt_for_ollama}\n---\n{st.session_state.user_prompt_for_ollama}"
                st.session_state.is_streaming = True
                st.rerun()

        # --- 詳細設定 Expander ---
        with st.expander("生成されたプロンプト詳細"):
            st.text_area(
                "生成されたプロンプト（外部LLMへのコピー用）:",
                value=st.session_state.prompt_display_text_right_pane,
                height=300, key="prompt_display_text_area_right_pane_key",
                help="ここに生成されたプロンプトが表示されます。外部LLMで試す場合はここからコピーしてください。",
            )

            if st.button("全検索結果からプロンプトを再生成", key="regenerate_prompt_with_all_contexts_button", help="現在表示されている全ての検索結果をコンテキストとして利用し、プロンプトを再生成します。"):
                logger.info("「全検索結果からプロンプトを再生成」ボタンが押されました。")
                with st.spinner("全検索結果からコンテキストを生成中..."):
                    all_search_results_for_prompt = {
                        "reranked": st.session_state.reranked_search_results,
                        "and": st.session_state.and_search_results,
                        "bm25": st.session_state.keyword_search_results,
                        "vector": st.session_state.vector_search_results
                    }
                    all_contexts_string, _ = st.session_state.engine.get_all_deduplicated_contexts_string(all_search_results_for_prompt)
                    user_prompt_content = f"""**提供されたコンテキスト:**\n{all_contexts_string}\n\n---\n**ユーザーの質問・関心:**\n{st.session_state.current_ai_question_text}"""
                    # 外部LLM用のプロンプトを使用する
                    system_prompt_for_all_contexts = st.session_state.engine.qa_system_prompt_for_all_contexts
                    st.session_state.prompt_display_text_right_pane = f"{system_prompt_for_all_contexts}\n---\n{user_prompt_content}"
                st.rerun()

        with st.expander("高度な設定"):
            def update_max_context_tokens():
                st.session_state.max_context_tokens_for_prompt = st.session_state.max_context_tokens_for_prompt_input_key
                logger.info(f"最大コンテキストトークン数がUIから変更されました。新: {st.session_state.max_context_tokens_for_prompt}")
            st.number_input(
                "最大コンテキストトークン数:", min_value=500, max_value=30000,
                value=st.session_state.max_context_tokens_for_prompt,
                key="max_context_tokens_for_prompt_input_key", step=100,
                help="AIへの質問時に含めることができる検索結果の最大トークン数です。",
                on_change=update_max_context_tokens
            )

        # --- レポート表示エリア (2段階プロセス対応) ---
        if st.session_state.ui_state != 'idle':
            st.markdown("---")
            if st.session_state.ui_state == 'streaming_answer':
                # 生成中断ボタンをテキストエリアより先に配置（位置固定のため）
                if st.button("⏹️ 生成をストップ", key="stop_generation_btn", type="primary"):
                    st.session_state.ui_state = 'idle'
                    st.rerun()
            # 免責事項の表示
            st.caption("⚠️ **AIによる生成結果は不正確な場合があります。あくまで調査の補助としてご利用ください。**")
            # 質問・関心事を表示
            st.markdown(f"##### 【質問・関心事】 {st.session_state.current_ai_question_text}")

            # テキスト表示用のプレースホルダをボタンの後に作成
            answer_placeholder = st.empty()

            if st.session_state.ui_state == 'streaming_answer':
                with st.spinner("AIがレポートを生成中..."):
                    response_generator = st.session_state.engine.stream_clean_answer(
                        user_prompt=st.session_state.user_prompt_for_ollama
                    )
                    st.session_state.clean_answer = answer_placeholder.write_stream(response_generator)

                # 日付表記の統一処理 (和暦・西暦の補完)
                with st.spinner("日付表記を統一しています..."):
                    try:
                        date_converter = DateStandardizer()
                        st.session_state.clean_answer = date_converter.process_text(st.session_state.clean_answer)
                    except Exception as e:
                        logger.error(f"日付正規化処理中にエラーが発生しました: {e}", exc_info=True)

                st.session_state.ui_state = 'getting_citations'
                st.rerun()

            if st.session_state.ui_state == 'getting_citations':
                answer_placeholder.markdown(st.session_state.clean_answer, unsafe_allow_html=True)
                with st.spinner("出典を検証中です..."):
                    llm_citation_map = st.session_state.engine.get_citation_map_from_answer(
                        answer_text=st.session_state.clean_answer,
                        contexts_string=st.session_state.contexts_for_citation
                    )

                # 箇条書きに出典を付与するロジック
                final_answer_parts = []
                lines = st.session_state.clean_answer.strip().split('\n')
                bullet_pattern = re.compile(r"^(\s*([\*\-]|(?:\d+\.))\s+)")
                header_pattern = re.compile(r"^\s*【.*?】")
                item_counter = 1

                for line in lines:
                    stripped_line = line.strip()
                    if not stripped_line:
                        final_answer_parts.append("") # 空行を維持
                        continue

                    if header_pattern.match(stripped_line):
                        final_answer_parts.append(line)
                        continue

                    match = bullet_pattern.match(line)
                    if match:
                        item_key = f"A{item_counter}"
                        context_indices = llm_citation_map.get(item_key, [])
                        citation_str = ""

                        # LLMが「情報の不在(0)」と判定した場合
                        if 0 in context_indices:
                            citation_str = ""
                        
                        # LLMが根拠を見つけられなかった場合（空リスト）
                        elif not context_indices:
                            citation_str = " **[不明な出典]**"
                        
                        # 通常の出典がある場合
                        else:
                            show_numbers = st.session_state.engine.show_citation_context_numbers
                            if show_numbers:
                                citations = []
                                for idx in sorted(context_indices):
                                    filename = st.session_state.prompt_citation_map.get(idx, "不明なファイル")
                                    citations.append(f"{idx}: {filename}")
                                citation_str = f" **[{', '.join(citations)}]**"
                            else:
                                # 番号なし (ファイル名のみ、重複排除)
                                filenames = set()
                                for idx in sorted(context_indices):
                                    filename = st.session_state.prompt_citation_map.get(idx, "不明なファイル")
                                    filenames.add(filename)
                                citation_str = f" **[{', '.join(sorted(list(filenames)))}]**"

                        final_answer_parts.append(line.rstrip() + citation_str)
                        item_counter += 1
                    else:
                        final_answer_parts.append(line)

                st.session_state.final_answer = "\n".join(final_answer_parts)
                st.session_state.ui_state = 'done'
                st.rerun()

            if st.session_state.ui_state == 'done':
                answer_placeholder.markdown(st.session_state.final_answer, unsafe_allow_html=True)
                
                # --- AIレポートログ保存 (自動) ---
                report_log_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "query_semantic": st.session_state.current_ai_question_text,
                    "query_keywords": st.session_state.last_executed_must_keywords, # 検索時のキーワード
                    "search_query_semantic": st.session_state.last_executed_semantic_query, # 検索時の意味クエリ
                    "settings": {
                        "deep_search": st.session_state.is_deep_search_mode,
                        "skip_rerank": st.session_state.skip_rerank
                    },
                    "final_answer": st.session_state.final_answer,
                    "citation_contexts": list(st.session_state.prompt_citation_map.values())
                }
                st.session_state.engine.save_ai_report(report_log_data)
                
                # ダウンロードボタン
                timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                report_filename = f"report_{timestamp_str}.txt"
                report_content = f"質問・関心事:\n{st.session_state.current_ai_question_text}\n\nAIによるレポート:\n{st.session_state.final_answer}"
                
                st.download_button(
                    label="📥 レポートをダウンロード",
                    data=report_content,
                    file_name=report_filename,
                    mime="text/plain",
                    key="download_report_button"
                )

                with st.expander("このレポートの根拠となったコンテキスト詳細"):
                    st.text_area("参照コンテキスト", value=st.session_state.contexts_for_citation, height=300, label_visibility="collapsed")

        # --- 従来のストリーミング処理（検索結果概要用） ---
        elif st.session_state.is_streaming:
            # 生成中断ボタン（論点抽出用）
            if st.button("⏹️ 生成をストップ", key="stop_extraction_btn", type="primary"):
                st.session_state.is_streaming = False
                st.rerun()
            
            with st.spinner("AIがレポートを生成中..."):
                # stream_llm_responseは廃止されたため、Ollamaを直接呼び出す
                # この機能は典拠不要のため、stream_clean_answerは使わない
                import ollama
                stream = ollama.chat(
                    model=st.session_state.engine.ollama_model_name,
                    messages=[
                        {'role': 'system', 'content': st.session_state.system_prompt_for_ollama},
                        {'role': 'user', 'content': st.session_state.user_prompt_for_ollama},
                    ],
                    stream=True,
                    options={'temperature': 0.2}
                )
                response_generator = (chunk['message']['content'] for chunk in stream)
                st.session_state.llm_response = st.write_stream(response_generator)
                st.session_state.is_streaming = False
                st.rerun()

        elif st.session_state.llm_response:
            st.markdown("---")
            st.markdown("##### AIによるレポート:")
            st.markdown(st.session_state.llm_response, unsafe_allow_html=True)

if not st.session_state.initialized:
    logger.info("アプリケーション初期化シーケンス開始。")
    st.info("アプリケーションを初期化中...")

    if st.session_state.engine is None:
        try:
            # configファイルのパスをプロジェクトルートからの相対パスで指定
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.toml')
            
            # コマンドライン引数の解析 (ワークスペースの上書き用)
            parser = argparse.ArgumentParser()
            parser.add_argument('--workspace', type=str, default=None, help='ワークスペースディレクトリのパス')
            args, _ = parser.parse_known_args()

            st.session_state.engine = ScholarScopeEngine(config_path=config_path, workspace_dir=args.workspace)
            setup_logging(st.session_state.engine)
            logger.info("ScholarScopeEngineのインスタンス化成功。")

            # --- エンジンの設定値でセッションステートを更新 ---
            engine_config = st.session_state.engine.config
            st.session_state.max_context_tokens_for_prompt = engine_config['settings']['default_max_context_tokens']
            st.session_state.last_used_max_context_tokens_for_prompt_in_contexts_string = engine_config['settings']['default_max_context_tokens']

        except Exception as e:
            print(f"CRITICAL: ScholarScopeEngineの初期化に失敗: {e}")
            st.error(f"コアエンジンの初期化に失敗しました: {e}")
            st.session_state.initialization_status = "error"
            st.stop()

    st.session_state.initialized = True
    logger.info("アプリケーション初期化シーケンス完了。")


# --- ユーザー確認とインデックス処理 (ロジックは変更なし) ---
if not st.session_state.rebuild_decision_made:
    logger.info("インデックス再構築判断シーケンス開始。")
    action, reason = st.session_state.engine.check_and_decide_build_action()

    if action == "STOP":
        st.error(reason)
        st.stop()

    elif action == "LOAD_EMPTY":
        st.info(reason)
        st.session_state.rebuild_decision_made = True
        st.session_state.user_chose_to_rebuild = False

    elif action == "FORCE_REBUILD":
        st.info(reason)
        st.session_state.rebuild_decision_made = True
        st.session_state.user_chose_to_rebuild = True

    elif action == "LOAD":
        st.info(reason)
        st.session_state.rebuild_decision_made = True
        st.session_state.user_chose_to_rebuild = False

    elif action == "ASK_INCREMENTAL":
        st.warning(reason)
        st.info("インデックスの更新方法を選択してください。")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("差分更新を実行", key="incremental_update_button", use_container_width=True, type="primary"):
                st.session_state.user_chose_to_incrementally_update = True
                st.session_state.rebuild_decision_made = True
                st.rerun()
        with col2:
            if st.button("インデックスを全再構築", key="force_rebuild_button_incremental", use_container_width=True):
                st.session_state.user_chose_to_rebuild = True
                st.session_state.rebuild_decision_made = True
                st.rerun()
        st.stop() # ユーザーの選択を待つ

    elif action == "ASK_REBUILD":
        st.warning(reason)
        st.info("インデックスの再構築が必要です。")
        if st.button("インデックスを再構築", key="force_rebuild_button_rebuild", use_container_width=True, type="primary"):
            st.session_state.user_chose_to_rebuild = True
            st.session_state.rebuild_decision_made = True
            st.rerun()
        st.stop() # ユーザーの選択を待つ

if st.session_state.rebuild_decision_made and st.session_state.initialization_status not in ["ready", "error"]:

    action_taken = False
    build_params = {}

    if st.session_state.user_chose_to_incrementally_update:
        st.info("インデックスを差分更新します。")
        build_params = {"incremental_update": True}
        action_taken = True
    elif st.session_state.user_chose_to_rebuild:
        st.info("インデックスを構築/再構築します。")
        build_params = {"force_rebuild": True}
        action_taken = True
    elif st.session_state.user_chose_to_rebuild is False:
        st.info("既存インデックスを利用します。")
        build_params = {}
        action_taken = True

    if action_taken:
        with st.spinner("インデックスを準備しています..."):
            result = st.session_state.engine.build_index(**build_params)

        if result.get("status") == "success":
            st.success(result.get("message", "処理が完了しました。"))
        else:
            st.warning(result.get("message", "処理中に問題が発生しました。"))

        st.session_state.initialization_status = "ready"
        st.rerun()

# --- 初期化後のメインUI描画 ---
if st.session_state.initialization_status != "ready":
    if st.session_state.initialization_status == "error":
        logger.error("初期化中にエラーが発生したため、エラーメッセージを表示して停止します。")
        st.error("初期化中にエラーが発生しました。アプリケーションを再起動するか、管理者にご連絡ください。詳細はログファイルを確認してください。")
    elif st.session_state.initialization_status in ["user_confirm_rebuild", "user_confirm_incremental"]:
        logger.debug(f"ユーザー確認待機中 ({st.session_state.initialization_status})。UI描画スキップ。")
        pass
    else:
        logger.info(f"アプリケーション準備中 ({st.session_state.initialization_status})。待機メッセージ表示。")
        st.info(f"アプリケーションの準備が完了するまでお待ちください... (状態: {st.session_state.initialization_status})")
    st.stop()

logger.info("アプリケーション準備完了。メインUI描画開始。")

st.markdown("""
<style>
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        height: calc(100vh - 10rem); overflow-y: auto; padding-right: 15px; padding-left: 5px;
    }
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 1rem; padding-right: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- ヘッダー ---
header_cols = st.columns([0.7, 0.3])
st.markdown("<hr style='margin: 0; border-top: 1px solid #E0E0E0;'>", unsafe_allow_html=True)
with header_cols[0]:
    st.header("文献探索支援システム", divider=False)

with header_cols[1]:
    # ワークスペース情報をコンパクトに表示
    workspace_path = st.session_state.engine.workspace_dir
    st.markdown(f"<div style='font-size: 0.8rem; text-align: right;'><b>Workspace:</b> <code>{workspace_path}</code></div>", unsafe_allow_html=True)

    # ロード済みファイル情報をPopoverで表示
    loaded_files_info = st.session_state.engine.get_loaded_files_info()
    if loaded_files_info:
        with st.popover(f"ロード済みファイル: {len(loaded_files_info)}件", use_container_width=True):
            st.markdown("##### ロード済みファイル一覧")
            for finfo in loaded_files_info:
                st.markdown(f"- {finfo['name']}")
    else:
        documents_folder_name = st.session_state.engine.config['paths']['documents_folder']
        st.info(f"ファイルがありません。`{documents_folder_name}`にファイルを追加してください。")

if st.session_state.show_ai_question_pane:
    main_cols = st.columns([0.25, 0.45, 0.30])
    left_column, center_column, right_column = main_cols[0], main_cols[1], main_cols[2]
    render_right_column()
else:
    main_cols = st.columns([0.35, 0.65])
    left_column, center_column = main_cols[0], main_cols[1]
    right_column = None

# --- 左ペイン (検索フォーム) ---
with left_column:
    st.subheader("検索設定")
    with st.form(key="search_form_left_pane"):
        full_text_query_input = st.text_input(
            "全文検索（絞り込み）",
            value=st.session_state.last_executed_full_text_query,
            key="input_full_text",
            placeholder=st.session_state.engine.config.get('ui', {}).get('must_keywords_placeholder', "スペース区切りでAND検索、-で除外検索"),
            help="スペース区切りでAND検索。単語の前に `-` を付けると除外検索になります。"
        )
        with st.expander("絞り込みを追加"):
            doc_name_filter_input = st.text_input(
                "ドキュメント名",
                value=st.session_state.last_executed_doc_name_filter,
                key="input_doc_name_filter",
                placeholder=st.session_state.engine.config.get('ui', {}).get('must_keywords_placeholder', "スペース区切りでAND検索、-で除外検索"),
                help="ドキュメント名で絞り込みます。スペース区切りでAND検索、-で除外検索ができます。"
            )
        with st.expander("意味・文脈で検索（AI利用）"):

            semantic_query_input = st.text_area(
                "質問・関心事",
                value=st.session_state.last_executed_semantic_query,
                key="input_semantic_query",
                height=150,
                placeholder=st.session_state.engine.config.get('ui', {}).get('semantic_query_placeholder', "例: 明治20年頃の条約改正の動きについて知りたい"),
                help="質問・関心事を自然な文章で入力してください。\n\n💡 **便利な機能**\n- **和暦/西暦の自動補完**: 「明治20年」と入力すると、内部で「明治20年（1887年）」のように変換され、どちらの表記でもヒットするようになります。"
            )
            use_ai_optimization = st.toggle(
                "検索クエリの最適化",
                value=True,
                help="【ON】AIが質問の意図を解釈し、各検索エンジン（キーワード・ベクトル・リランク）に合わせて最適な形に書き換えます。曖昧な質問でもヒットしやすくなります。\n\n【OFF】入力文をそのまま使用します（キーワード検索用の形態素解析は行われます）。\n専門用語を厳密に検索したい場合や、AIによる意訳・要約で重要な単語が省略されてしまうのを防ぎたい場合は、OFFの方が良い結果が得られます。"
            )
            
            # 設定値を取得（表示用）
            std_count = st.session_state.engine.reranker_input_count
            deep_count = st.session_state.engine.reranker_input_count_deep

            # UI簡略化: リランキングのON/OFFのみにする (GPU有効化に伴いDeepでも高速なため)
            is_rerank_active = st.toggle(
                "関連度順ソート（AI利用）",
                value=not st.session_state.skip_rerank,
                help=f"【ON】検索候補の上位（最大{deep_count*2}件）について、AIが本文を読み込んで質問との関連性を判定・並べ替えを行います。より的確な結果が得られます。\n\n【OFF】キーワードの出現頻度や単純な類似度順に表示します。高速ですが、AIによる詳細な精査は行われません。"
            )

        search_submit_button = st.form_submit_button("検索実行", type="primary", disabled=st.session_state.is_searching)

    if search_submit_button:
        full_text_query = full_text_query_input.strip()
        doc_name_filter = doc_name_filter_input.strip()
        semantic_query = semantic_query_input.strip()

        # 全文検索から必須キーワードと除外キーワードをパース
        all_keywords = full_text_query.split()
        must_keywords_list = [kw for kw in all_keywords if not kw.startswith('-')]
        exclude_keywords_list = [kw[1:] for kw in all_keywords if kw.startswith('-') and len(kw) > 1]
        must_keywords = " ".join(must_keywords_list)

        if not must_keywords and not semantic_query:
            st.warning("「必須キーワード」または「質問・関心事」のいずれかを入力してください。")
        else:
            # ラジオボタンの選択値からフラグを設定
            skip_rerank_req = False
            # トグルがOFFならスキップ、ONならDeepモード(最大件数)で実行
            if not is_rerank_active:
                skip_rerank_req = True
            # Deepモードを常に有効にする（件数はconfigで制御）
            deep_search_req = True

            # 検索実行時に結果をクリアし、処理中フラグを立てる
            st.session_state.and_search_results = []
            st.session_state.keyword_search_results = []
            st.session_state.vector_search_results = []
            st.session_state.reranked_search_results = []
            st.session_state.search_result_meta = {}
            st.session_state.llm_response = ""
            st.session_state.clean_answer = ""
            st.session_state.final_answer = ""
            st.session_state.search_was_cancelled = False # 新規検索時は中断フラグを下げる

            # 各検索結果の展開状態をリセット
            if 'reranked_results_expanded_state' in st.session_state: st.session_state.reranked_results_expanded_state = None
            if 'and_results_expanded_state' in st.session_state: st.session_state.and_results_expanded_state = None
            if 'bm25_results_expanded_state' in st.session_state: st.session_state.bm25_results_expanded_state = None
            if 'vector_results_expanded_state' in st.session_state: st.session_state.vector_results_expanded_state = None
            if 'and_results_expanded_state_skip' in st.session_state: st.session_state.and_results_expanded_state_skip = None

            # 検索リクエストをセッションステートに保存
            st.session_state.search_request = {
                "must_keywords": must_keywords,
                "exclude_keywords": exclude_keywords_list,
                "doc_name_filter": doc_name_filter,
                "semantic_query": semantic_query,
                "skip_rerank": skip_rerank_req,
                "full_text_query": full_text_query, # UI表示用に保存
                "use_ai_optimization": use_ai_optimization,
                "deep_search": deep_search_req
            }
            st.session_state.is_searching = True # 処理開始フラグ
            logger.info("検索リクエストを受付。処理を開始します。")
            st.rerun() # UIを即時更新して結果をクリアし、ボタンを無効化

    # --- 条件クリア / 検索中止 ボタン (フォーム外) ---
    if not st.session_state.is_searching:
        # on_clickを使うことで、次のウィジェット描画前に値をリセットできる（エラー回避）
        st.button("🗑️ 条件・検索結果をクリア", use_container_width=True, on_click=reset_application_state)

    # --- 検索リクエストの処理 ---
    if st.session_state.is_searching and st.session_state.search_request:
        req = st.session_state.search_request
        must_keywords = req["must_keywords"]
        exclude_keywords = req["exclude_keywords"]
        doc_name_filter = req["doc_name_filter"]
        semantic_query = req["semantic_query"]
        skip_rerank_val = req["skip_rerank"]
        full_text_query = req["full_text_query"]
        use_ai_optimization_val = req.get("use_ai_optimization", False)
        deep_search_val = req.get("deep_search", False)

        st.session_state.skip_rerank = skip_rerank_val
        st.session_state.is_deep_search_mode = deep_search_val
        logger.info(f"検索実行。必須: '{must_keywords}', 質問: '{semantic_query}', Deep: {deep_search_val}, スキップ: {skip_rerank_val}")
        st.session_state.last_executed_must_keywords = must_keywords
        st.session_state.last_executed_full_text_query = full_text_query
        st.session_state.last_executed_doc_name_filter = doc_name_filter
        st.session_state.last_executed_semantic_query = semantic_query
        
        # レポート生成用の質問文にも日付正規化を適用してセットする
        raw_question_text = semantic_query or must_keywords
        if raw_question_text:
            st.session_state.current_ai_question_text = st.session_state.engine.date_standardizer.process_text(raw_question_text)
        else:
            st.session_state.current_ai_question_text = ""
            
        st.session_state.executed_and_query_for_highlight = must_keywords
        st.session_state.executed_bm25_query_for_highlight = semantic_query
        st.session_state.executed_semantic_query_for_highlight = semantic_query
        st.session_state.executed_optimized_bm25_query = None # リセット
        st.session_state.executed_rerank_query = None # リセット

        # --- 進捗表示用コールバック定義 ---
        with center_column:
            st.subheader("検索結果")
            search_status_container = st.status("検索プロセス実行中...", expanded=True)
            status_placeholder = search_status_container.empty()
            progress_bar_placeholder = search_status_container.empty()

            if st.button("⛔ 検索を中止", key="stop_search_center", type="primary", use_container_width=True):
                st.session_state.is_searching = False
                st.session_state.search_request = None
                st.session_state.search_was_cancelled = True # 中断フラグを立てる
                logger.info("ユーザーにより検索が中断されました。")
                st.rerun()

        # 状態管理用辞書
        prog_state = {
            "steps": [
                {"key": "query_opt", "label": "🤔 質問・関心事の分析", "status": "pending", "detail": ""},
                {"key": "retrieval", "label": "📚 候補を広く収集", "status": "pending", "detail": ""},
                {"key": "filter", "label": "🧹 候補の整理・統合", "status": "pending", "detail": ""},
                {"key": "rerank", "label": "👀 内容の精査", "status": "pending", "detail": "", "progress": 0, "total": 0}
            ]
        }

        def render_progress():
            md_lines = []
            for step in prog_state["steps"]:
                icon = "⬜"
                if step["status"] == "running": icon = "🔄"
                elif step["status"] == "done": icon = "✅"
                elif step["status"] == "skipped": icon = "⏭️"
                
                # detailが空なら状態に応じたデフォルト文を表示
                msg = step["detail"]
                if not msg:
                    if step["status"] == "pending": msg = "待機中..."
                    elif step["status"] == "running": msg = "実行中..."
                
                md_lines.append(f"{icon} **{step['label']}**: {msg}")
            
            status_placeholder.markdown("\n\n".join(md_lines))
            
            # リランキングのプログレスバー制御
            rerank_step = prog_state["steps"][3]
            if rerank_step["status"] == "running" and rerank_step["total"] > 0:
                pct = min(rerank_step["progress"] / rerank_step["total"], 1.0)
                progress_bar_placeholder.progress(pct)
            else:
                progress_bar_placeholder.empty()

        def update_step_status(key, status, detail=None):
            for step in prog_state["steps"]:
                if step["key"] == key:
                    step["status"] = status
                    if detail is not None: step["detail"] = detail
            render_progress()

        def search_progress_callback(phase, status=None, detail=None, current=None, total=None):
            """
            phase: 'start', 'query_opt', 'retrieval', 'filter', 'rerank', 'done'
            status: 'running', 'done', 'skipped', 'failed' (phase='start'/'done'の場合は省略可)
            detail: 表示するメッセージ文字列
            """
            if phase == 'start':
                render_progress()
            
            elif phase == 'query_opt':
                if status == 'running':
                    # エンジンからのメッセージを上書きして、より親しみやすい表現に
                    update_step_status("query_opt", "running", "AIが検索の意図を解釈しています...")
                elif status == 'skipped':
                    update_step_status("query_opt", "skipped", "スキップ (元の入力をそのまま使用)")
                elif status == 'done':
                    update_step_status("query_opt", "done", detail)

            elif phase == 'retrieval':
                if status == 'running':
                    update_step_status("retrieval", "running", "関連しそうな箇所を集めています...")
                elif status == 'done':
                    update_step_status("retrieval", "done", detail)

            elif phase == 'filter':
                if status == 'running':
                    update_step_status("filter", "running", "重複や条件外のものを除外しています...")
                elif status == 'done':
                    update_step_status("filter", "done", detail)

            elif phase == 'rerank':
                step = prog_state["steps"][3]
                if current is not None and total is not None:
                    step["status"] = "running"
                    step["progress"] = current
                    step["total"] = total
                    # 進捗表示をより分かりやすく
                    step["detail"] = f"AIが本文を解析し、関連度順に並べています ({current} / {total} 件)"
                    render_progress()
                elif status == 'running':
                    update_step_status("rerank", "running", "AIが本文を解析し、関連度順に並べています...")
                elif status == 'done':
                    update_step_status("rerank", "done", "完了 (関連度順に並べ替え)")
                elif status == 'skipped':
                    update_step_status("rerank", "skipped", "スキップ (高速モード)")

            elif phase == 'done':
                # 全工程が完了したらコンテナを畳む
                search_status_container.update(label="検索処理完了", state="complete", expanded=False)

        search_start_time = time.time()

        try:
            optimized_queries = None
            if use_ai_optimization_val and semantic_query:
                search_progress_callback('start')
                search_progress_callback('query_opt', 'running', "分析中...")
                optimized_queries = st.session_state.engine.optimize_search_queries(semantic_query)
                
                if optimized_queries:
                    # シンプルな表示にする
                    q_summary = f"キーワード: {optimized_queries['bm25_query'][:15]}..., ベクトル: {optimized_queries['vector_query'][:15]}..."
                    search_progress_callback('query_opt', 'done', q_summary)
                    
                    # ハイライト用変数の更新
                    st.session_state.executed_bm25_query_for_highlight = optimized_queries['bm25_query']
                    st.session_state.executed_semantic_query_for_highlight = optimized_queries['vector_query']
                    st.session_state.executed_optimized_bm25_query = optimized_queries['bm25_query']
                    st.session_state.executed_rerank_query = optimized_queries.get('rerank_query')
                else:
                    search_progress_callback('query_opt', 'skipped') # 失敗時はスキップ扱い
            else:
                 # 最適化OFFの場合
                 search_progress_callback('start')
                 search_progress_callback('query_opt', 'skipped')

            search_results_dict = st.session_state.engine.search(
                must_keywords=must_keywords,
                semantic_query=semantic_query,
                doc_name_filter=doc_name_filter,
                exclude_keywords=exclude_keywords,
                skip_rerank=st.session_state.skip_rerank,
                optimized_queries=optimized_queries,
                deep_search=deep_search_val,
                callback=search_progress_callback
            )
            search_progress_callback('done')
        
        except Exception as e:
            search_status_container.update(label="検索中にエラーが発生しました", state="error")
            logger.error(f"検索処理中にエラー: {e}", exc_info=True)
            st.error(f"エラーが発生しました: {e}")
            st.stop()
            
        st.session_state.last_search_duration = time.time() - search_start_time

        # 検索結果をセッションステートに格納
        st.session_state.and_search_results = search_results_dict.get("and", [])
        st.session_state.keyword_search_results = search_results_dict.get("bm25", [])
        st.session_state.vector_search_results = search_results_dict.get("vector", [])
        st.session_state.reranked_search_results = search_results_dict.get("reranked", [])
        st.session_state.search_result_meta = search_results_dict.get("meta", {})

        try:
            search_hits_info = {
                "and": len(st.session_state.and_search_results),
                "bm25": len(st.session_state.keyword_search_results),
                "vector": len(st.session_state.vector_search_results),
                "reranked": len(st.session_state.reranked_search_results)
            }
            unique_hit_sources_for_history = set()
            all_docs_for_history = st.session_state.and_search_results + \
                                   [doc for doc, score in st.session_state.keyword_search_results] + \
                                   [doc for doc, score in st.session_state.vector_search_results]
            for doc in all_docs_for_history:
                if doc.metadata.get("source"):
                    unique_hit_sources_for_history.add(doc.metadata["source"])

            history_entry_to_save = {
                "timestamp": datetime.datetime.now().isoformat(),
                "must_keywords": must_keywords,
                "semantic_query": semantic_query,
                "hits_per_method": search_hits_info,
                "unique_source_documents_hit": len(unique_hit_sources_for_history)
            }
            save_search_history(history_entry_to_save)
        except Exception as e_hist:
            logger.error(f"検索履歴の作成または保存呼び出し中に予期せぬエラー: {e_hist}", exc_info=True)

        logger.info("検索処理完了。UIを再描画します。")
        st.session_state.is_searching = False # 処理完了
        st.session_state.search_request = None # リクエストをクリア
        st.rerun()

    # --- 最適化結果の事後表示 ---
    if st.session_state.executed_optimized_bm25_query and st.session_state.search_request is None:
         with st.expander("💡 AIによるクエリ最適化結果", expanded=False):
            st.markdown(f"**キーワード検索用:** `{st.session_state.executed_bm25_query_for_highlight}`")
            st.markdown(f"**ベクトル検索用:** `{st.session_state.executed_semantic_query_for_highlight}`")
            if not st.session_state.skip_rerank and st.session_state.get('executed_rerank_query'):
                st.markdown(f"**リランキング用:** `{st.session_state.executed_rerank_query}`")

    st.divider()
    st.subheader("表示設定")
    show_pane_checkbox_val_left = st.checkbox(
        "AI分析パネルを表示", value=st.session_state.show_ai_question_pane,
        key="cb_show_ai_pane_left_key", help="右側に検索結果を使ったAI分析パネルを表示/非表示します。"
    )
    if show_pane_checkbox_val_left != st.session_state.show_ai_question_pane:
        st.session_state.show_ai_question_pane = show_pane_checkbox_val_left
        logger.debug(f"AI概要パネル表示状態変更: {st.session_state.show_ai_question_pane}")
        st.rerun()

# --- 中央ペイン (検索結果表示) ---
with center_column:
    st.subheader("検索結果")
    show_individual_results = False # エラー回避のため事前に初期化
    any_query_executed = st.session_state.last_executed_must_keywords or st.session_state.last_executed_semantic_query
    if any_query_executed:
        all_hit_docs_for_stats = []
        if st.session_state.reranked_search_results: all_hit_docs_for_stats.extend([doc for doc, score in st.session_state.reranked_search_results])
        if st.session_state.and_search_results: all_hit_docs_for_stats.extend(st.session_state.and_search_results)
        if st.session_state.keyword_search_results: all_hit_docs_for_stats.extend([doc for doc, score in st.session_state.keyword_search_results])
        if st.session_state.vector_search_results: all_hit_docs_for_stats.extend([doc for doc, score in st.session_state.vector_search_results])

        unique_hit_source_names = set()
        if all_hit_docs_for_stats:
            for doc_for_stats in all_hit_docs_for_stats:
                source_name_for_stats = doc_for_stats.metadata.get("source")
                if source_name_for_stats:
                    unique_hit_source_names.add(Path(source_name_for_stats).name)

        loaded_files_info = st.session_state.engine.get_loaded_files_info()
        total_uploaded_docs = len(loaded_files_info)

        if total_uploaded_docs > 0:
            duration_html = f" <span style='color:gray; font-size:0.8rem; margin-left:10px;'>({st.session_state.last_search_duration:.2f}秒)</span>" if st.session_state.last_search_duration > 0 else ""
            st.markdown(f"**ヒット文書:** {len(unique_hit_source_names)} / {total_uploaded_docs} 件{duration_html}", unsafe_allow_html=True)
            if unique_hit_source_names:
                with st.expander("ヒットしたドキュメント一覧 (クリックで展開)"):
                    for name in sorted(list(unique_hit_source_names)): st.markdown(f"- {name}")
            st.markdown("---")

    active_search_results_exist = False

    # 1. リランキング結果 (最優先で表示)
    if st.session_state.reranked_search_results:
        active_search_results_exist = True

        # 表示するタイトルを動的に変更
        num_reranked = len(st.session_state.reranked_search_results)

        header_cols_reranked = st.columns([0.8, 0.2])
        with header_cols_reranked[0]:
            if st.session_state.last_executed_must_keywords:
                st.markdown(f"#### AIによる再評価：関連度の高い順 {num_reranked}件（絞込あり）", unsafe_allow_html=True)
            else:
                st.markdown(f"#### AIによる再評価：関連度の高い順 {num_reranked}件", unsafe_allow_html=True)
        with header_cols_reranked[1]:
            if 'reranked_results_expanded_state' not in st.session_state:
                st.session_state.reranked_results_expanded_state = None # None: 初期, True: 全展開, False: 全閉じる
            button_label_reranked = "すべて展開" if st.session_state.reranked_results_expanded_state in [None, False] else "すべて閉じる"
            if st.button(button_label_reranked, key="toggle_reranked", use_container_width=True):
                st.session_state.reranked_results_expanded_state = not st.session_state.reranked_results_expanded_state in [True]
                st.rerun()

        # ハイライト用のクエリを決定
        if st.session_state.executed_optimized_bm25_query:
            query_for_highlight = st.session_state.executed_optimized_bm25_query
        else:
            query_for_highlight = st.session_state.last_executed_semantic_query or st.session_state.last_executed_must_keywords

        query_terms_for_highlight = highlight.get_query_terms_for_highlight(
            query_for_highlight,
            search_type="bm25", # 意味検索のハイライトはBM25方式が適している
            tokenizer_instance=st.session_state.engine.tokenizer
        )

        num_to_expand = st.session_state.engine.initially_expanded_results_count

        for i, (chunk_doc, score) in enumerate(st.session_state.reranked_search_results): # reranker_input_count * 2 件が最大
            source_full_path = chunk_doc.metadata.get("source", "N/A")
            source_filename = Path(source_full_path).name
            page_num_str = str(chunk_doc.metadata.get("page", "N/A"))
            page_num_info = f", Page: {page_num_str}" if page_num_str.isdigit() else ""

            # オレンジ -> 薄いオレンジに変更
            highlighted_content = highlight.highlight_text(chunk_doc.page_content, query_terms_for_highlight, default_color="#FFE0B2", search_type="bm25")

            # --- expandedロジック ---
            is_expanded = False
            if st.session_state.reranked_results_expanded_state is True:
                is_expanded = True
            elif st.session_state.reranked_results_expanded_state is False:
                is_expanded = False
            else: # 初期状態
                is_expanded = (i < num_to_expand)
            # --- ここまで ---

            with st.expander(f"**{i+1}.** {source_filename}{page_num_info} (関連度スコア: **{score:.4f}**)", expanded=is_expanded):
                st.markdown(highlighted_content, unsafe_allow_html=True)

        # --- 「その他の候補」の表示ロジック ---
        # リランキングに使われなかった残りの候補を表示
        other_bm25_results = st.session_state.keyword_search_results[st.session_state.engine.reranker_input_count:]
        other_vector_results = st.session_state.vector_search_results[st.session_state.engine.reranker_input_count:]

        if other_bm25_results or other_vector_results:
            st.markdown("---")
            header_cols_other = st.columns([0.8, 0.2])
            with header_cols_other[0]:
                st.markdown("#### その他の候補")
            with header_cols_other[1]:
                if 'other_results_expanded_state' not in st.session_state:
                    st.session_state.other_results_expanded_state = None
                button_label_other = "すべて展開" if st.session_state.other_results_expanded_state in [None, False] else "すべて閉じる"
                if st.button(button_label_other, key="toggle_other", use_container_width=True):
                    st.session_state.other_results_expanded_state = not st.session_state.other_results_expanded_state in [True]
                    st.rerun()

            is_expanded_other = st.session_state.other_results_expanded_state is True

            if other_bm25_results:
                # 黒文字 + 水色背景バッジ風
                st.markdown(f"##### <span style='background-color:#B2EBF2; color:black; padding:2px 6px; border-radius:4px;'>キーワードスコア順</span>（上位{len(other_bm25_results)}件）", unsafe_allow_html=True)
                query_tokens_for_kw_highlight = highlight.get_query_terms_for_highlight(
                    st.session_state.executed_bm25_query_for_highlight, search_type="bm25",
                    tokenizer_instance=st.session_state.engine.tokenizer
                )
                for i, (chunk_doc, score) in enumerate(other_bm25_results):
                    source_full_path = chunk_doc.metadata.get("source", "N/A")
                    source_filename = Path(source_full_path).name
                    page_num_str = str(chunk_doc.metadata.get("page", "N/A"))
                    page_num_info = f", Page: {page_num_str}" if page_num_str != "N/A" and page_num_str.isdigit() else ""
                    highlighted_content_kw = highlight.highlight_text(chunk_doc.page_content, query_tokens_for_kw_highlight, search_type="bm25")
                    with st.expander(f"BM25-{i+st.session_state.engine.reranker_input_count+1}. {source_filename}{page_num_info} (キーワードスコア: {score:.4f})", expanded=is_expanded_other):
                        st.markdown(highlighted_content_kw, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            if other_vector_results:
                # 黒文字 + 薄紫背景バッジ風
                st.markdown(f"##### <span style='background-color:#E1BEE7; color:black; padding:2px 6px; border-radius:4px;'>意味が近い順</span>（上位{len(other_vector_results)}件）", unsafe_allow_html=True)
                query_terms_for_vec_highlight = highlight.get_query_terms_for_highlight(st.session_state.executed_semantic_query_for_highlight, search_type="vector")
                for i, (chunk_doc, score) in enumerate(other_vector_results):
                    source_full_path = chunk_doc.metadata.get("source", "N/A")
                    source_filename = Path(source_full_path).name
                    page_num_str = str(chunk_doc.metadata.get("page", "N/A"))
                    page_num_info = f", Page: {page_num_str}" if page_num_str != "N/A" and page_num_str.isdigit() else ""
                    highlighted_content_vec = highlight.highlight_text(chunk_doc.page_content, query_terms_for_vec_highlight, default_color="#E1BEE7", search_type="vector")
                    with st.expander(f"Vector-{i+st.session_state.engine.reranker_input_count+1}. {source_filename}{page_num_info} (AI類似度スコア: {score:.4f})", expanded=is_expanded_other):
                        st.markdown(highlighted_content_vec, unsafe_allow_html=True)

        st.markdown("---")

    # 2. リランキングがスキップされた場合の統合表示
    elif st.session_state.last_executed_semantic_query and st.session_state.skip_rerank:
        st.markdown("#### 検索結果（AIによる関連度順位付けはスキップされています）", unsafe_allow_html=True)

        # BM25の結果
        if st.session_state.keyword_search_results:
            header_cols_bm25 = st.columns([0.8, 0.2])
            with header_cols_bm25[0]:
                # 黒文字 + 水色背景バッジ風
                st.markdown(f"##### <span style='background-color:#B2EBF2; color:black; padding:2px 6px; border-radius:4px;'>キーワードスコア順</span>（上位{len(st.session_state.keyword_search_results)}件）", unsafe_allow_html=True)
            with header_cols_bm25[1]:
                if 'bm25_results_expanded_state' not in st.session_state:
                    st.session_state.bm25_results_expanded_state = None
                button_label_bm25 = "すべて展開" if st.session_state.bm25_results_expanded_state in [None, False] else "すべて閉じる"
                if st.button(button_label_bm25, key="toggle_bm25_skip", use_container_width=True):
                    st.session_state.bm25_results_expanded_state = not st.session_state.bm25_results_expanded_state in [True]
                    st.rerun()
            query_tokens_for_kw_highlight = highlight.get_query_terms_for_highlight(
                st.session_state.executed_bm25_query_for_highlight,
                search_type="bm25",
                tokenizer_instance=st.session_state.engine.tokenizer
            )
            num_to_expand = st.session_state.engine.initially_expanded_results_count
            for i, (chunk_doc, score) in enumerate(st.session_state.keyword_search_results): # スライスを削除し全件表示
                source_full_path = chunk_doc.metadata.get("source", "N/A")
                source_filename = Path(source_full_path).name
                page_num_str = str(chunk_doc.metadata.get("page", "N/A"))
                page_num_info = f", Page: {page_num_str}" if page_num_str != "N/A" and page_num_str.isdigit() else ""
                highlighted_content_kw = highlight.highlight_text(chunk_doc.page_content, query_tokens_for_kw_highlight, search_type="bm25")

                # --- expandedロジック ---
                is_expanded = False
                if st.session_state.bm25_results_expanded_state is True: is_expanded = True
                elif st.session_state.bm25_results_expanded_state is False: is_expanded = False
                else: is_expanded = (i < num_to_expand)

                with st.expander(f"{i+1}. {source_filename}{page_num_info} (キーワードスコア: {score:.4f})", expanded=is_expanded):
                    st.markdown(highlighted_content_kw, unsafe_allow_html=True)
            active_search_results_exist = True
            st.markdown("<br>", unsafe_allow_html=True)

        # Vectorの結果
        if st.session_state.vector_search_results:
            header_cols_vector = st.columns([0.8, 0.2])
            with header_cols_vector[0]:
                st.markdown(f"##### <span style='background-color:#E1BEE7; color:black; padding:2px 6px; border-radius:4px;'>意味が近い順</span>（上位{len(st.session_state.vector_search_results)}件）", unsafe_allow_html=True)
            with header_cols_vector[1]:
                if 'vector_results_expanded_state' not in st.session_state:
                    st.session_state.vector_results_expanded_state = None
                button_label_vector = "すべて展開" if st.session_state.vector_results_expanded_state in [None, False] else "すべて閉じる"
                if st.button(button_label_vector, key="toggle_vector_skip", use_container_width=True):
                    st.session_state.vector_results_expanded_state = not st.session_state.vector_results_expanded_state in [True]
                    st.rerun()
            query_terms_for_vec_highlight = highlight.get_query_terms_for_highlight(st.session_state.executed_semantic_query_for_highlight, search_type="vector")
            num_to_expand = st.session_state.engine.initially_expanded_results_count
            for i, (chunk_doc, score) in enumerate(st.session_state.vector_search_results): # スライスを削除し全件表示
                source_full_path = chunk_doc.metadata.get("source", "N/A")
                source_filename = Path(source_full_path).name
                page_num_str = str(chunk_doc.metadata.get("page", "N/A"))
                page_num_info = f", Page: {page_num_str}" if page_num_str != "N/A" and page_num_str.isdigit() else ""
                highlighted_content_vec = highlight.highlight_text(chunk_doc.page_content, query_terms_for_vec_highlight, default_color="#E1BEE7", search_type="vector")

                # --- expandedロジック ---
                is_expanded = False
                if st.session_state.vector_results_expanded_state is True: is_expanded = True
                elif st.session_state.vector_results_expanded_state is False: is_expanded = False
                else: is_expanded = (i < num_to_expand)

                with st.expander(f"{i+1}. {source_filename}{page_num_info} (AI類似度スコア: {score:.4f})", expanded=is_expanded):
                    st.markdown(highlighted_content_vec, unsafe_allow_html=True)
            active_search_results_exist = True

        if active_search_results_exist:
            st.markdown("---")

    # 3. AND検索のみの結果 (リランキングがなく、スキップもされていない場合)
    elif st.session_state.and_search_results:
        header_cols_and = st.columns([0.8, 0.2])
        
        # 件数表示のロジック: メタ情報を見て "+" をつけるか判断
        hit_count_str = f"{len(st.session_state.and_search_results)}"
        if st.session_state.search_result_meta.get("and_has_more"):
            st.caption(f"※ 表示上限（{hit_count_str}件）に達したため、一部の結果は表示されません。")
            hit_count_str += "+"

        with header_cols_and[0]:
            st.markdown(f"#### <span style='background-color:#C8E6C9; color:black; padding:2px 6px; border-radius:4px;'>完全一致</span>： {hit_count_str}件（キーワードを含む箇所）", unsafe_allow_html=True)
        with header_cols_and[1]:
            if 'and_results_expanded_state' not in st.session_state:
                st.session_state.and_results_expanded_state = None # 初期状態: None, 全展開: True, 全閉じる: False
            button_label_and = "すべて展開" if st.session_state.and_results_expanded_state in [None, False] else "すべて閉じる"
            if st.button(button_label_and, key="toggle_and", use_container_width=True):
                st.session_state.and_results_expanded_state = not st.session_state.and_results_expanded_state in [True]
                st.rerun()

        query_terms_for_and_highlight = highlight.get_query_terms_for_highlight(st.session_state.executed_and_query_for_highlight, search_type="and")
        num_to_expand = st.session_state.engine.initially_expanded_results_count

        for i, chunk_doc in enumerate(st.session_state.and_search_results):
            source_full_path = chunk_doc.metadata.get("source", "N/A")
            source_filename = Path(source_full_path).name
            page_num_str = str(chunk_doc.metadata.get("page", "N/A"))
            page_num_info = f", Page: {page_num_str}" if page_num_str != "N/A" and page_num_str.isdigit() else ""
            highlighted_content_and = highlight.highlight_text(chunk_doc.page_content, query_terms_for_and_highlight, default_color="#C8E6C9", search_type="and")

            # --- expandedロジック ---
            is_expanded = False
            if st.session_state.and_results_expanded_state is True:
                is_expanded = True
            elif st.session_state.and_results_expanded_state is False:
                is_expanded = False
            else: # 初期状態
                is_expanded = (i < num_to_expand)

            with st.expander(f"{i+1}. {source_filename}{page_num_info} (完全一致)", expanded=is_expanded):
                st.markdown(highlighted_content_and, unsafe_allow_html=True)
        active_search_results_exist = True
        st.markdown("---")

    # 4. 検索未実行または結果なしの場合のメッセージ (変更なし)
    if not any_query_executed:
        if not st.session_state.engine.get_loaded_files_info():
            documents_folder_name = st.session_state.engine.config['paths']['documents_folder']
            st.info(f"検索対象の文献ファイルがありません。\n`{documents_folder_name}` フォルダに直接ファイルを追加し、アプリを再起動してください。")
        else:
            st.info("左パネルの検索フォームから検索を実行してください。")
    elif any_query_executed and not active_search_results_exist:
        if st.session_state.search_was_cancelled:
            st.warning("検索処理が中断されました。")
        else:
            st.info("検索条件に一致する情報は見つかりませんでした。")
# project_root/src/engine.py
import os
import json
import logging
import toml
import re
import time
from pathlib import Path
from modules import file_processing, search_engines, reranker
from langchain.schema import Document
from modules import utils
import ollama
import lancedb
from modules.date_standardizer import DateStandardizer
from modules.kanji_converter import normalize_kanji

logger = logging.getLogger('scholarscope_lite')

class ScholarScopeEngine:
    def __init__(self, config_path="config.toml", workspace_dir=None):
        logger.info("ScholarScopeEngineの初期化を開始...")

        self.config = self._load_config(config_path)

        # --- プロジェクトルートとパス解決ヘルパーの定義 ---
        # config_path (src/../configs/config.toml) からプロジェクトルート (configsの親) を特定
        self.project_root = Path(config_path).resolve().parent.parent

        def _resolve_project_path(path_str):
            """プロジェクトルート基準でパスを解決する内部ヘルパー"""
            p = Path(path_str)
            if p.is_absolute():
                return p
            return (self.project_root / p).resolve()

        # --- パス設定 ---
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir).resolve()
            logger.info(f"ワークスペースディレクトリを引数から設定しました: {self.workspace_dir}")
        else:
            self.workspace_dir = _resolve_project_path(self.config['workspace_directory'])
            
        self.documents_dir = self.workspace_dir / self.config['paths']['documents_folder']
        self.app_data_dir = self.workspace_dir / self.config['paths']['app_data_folder']

        self.vectorstore_persist_dir = self.app_data_dir / self.config['paths']['vectorstore_folder']
        self.bm25_index_dir = self.app_data_dir / self.config['paths']['bm25_index_folder']
        self.bm25_index_path = self.bm25_index_dir / "bm25_data.pkl" # ファイル名を固定

        self.metadata_file_path = self.app_data_dir / self.config['paths']['metadata_file']

        # --- ログ・履歴・デバッグ出力先の分離 (ワークスペース外へ) ---
        # configで指定されたパスを使用。指定がなければデフォルト値(logs)を使用
        self.system_log_dir = _resolve_project_path(self.config['paths'].get('system_log_directory', 'logs'))
        self.search_history_path = _resolve_project_path(self.config['paths'].get('search_history_file', 'logs/search_history.jsonl'))
        
        self.log_dir = self.system_log_dir # 互換性のためエイリアス
        self.log_file_path = self.system_log_dir / self.config['paths']['log_file']

        # --- ログ設定の読み込み ---
        log_config = self.config.get('logging', {})
        self.console_log_level = log_config.get('console_log_level', 'INFO').upper()
        self.file_log_level = log_config.get('file_log_level', 'WARNING').upper()
        self.save_ai_reports = log_config.get('save_ai_reports', True) # デフォルトON

        # --- モデル・設定の読み込み ---
        self.deduplication_threshold = self.config['settings']['deduplication_threshold_for_prompt']
        self.ollama_model_name = self.config['models']['ollama_chat']
        
        # リランカーモデルのパス解決 (ローカルパスかHuggingFace IDかの判定)
        raw_reranker_path = self.config['models']['reranker']
        resolved_reranker_path = _resolve_project_path(raw_reranker_path)
        if resolved_reranker_path.exists():
            # ローカルにフォルダが存在する場合は、その絶対パスを使用
            self.reranker_model_name = str(resolved_reranker_path)
            logger.info(f"リランキングモデルをローカルパスとして設定: {self.reranker_model_name}")
        else:
            # 存在しない場合は設定値をそのまま使用 (HuggingFace Hubからのダウンロードなどを想定)
            self.reranker_model_name = raw_reranker_path
        
        # 類似度計算モデル（SimCSE）のパス解決
        # configに未定義の場合はデフォルトのHuggingFace IDを使用（後方互換）
        default_hf_id = "pkshatech/simcse-ja-bert-base-clcmlp"
        raw_sim_path = self.config['models'].get('similarity', default_hf_id)
        resolved_sim_path = _resolve_project_path(raw_sim_path)

        if resolved_sim_path.exists():
            self.similarity_model_name = str(resolved_sim_path)
            logger.info(f"類似度モデルをローカルパスとして設定: {self.similarity_model_name}")
        else:
            # ローカルに存在しない場合、もし設定値がデフォルトのローカルパス(ai_models/...)なら、
            # ユーザーの配置忘れと判断してHFのIDにフォールバックする（ネット経由でDLさせる）
            if "ai_models" in str(raw_sim_path):
                logger.warning(f"指定されたローカルモデル '{resolved_sim_path}' が見つかりません。Hugging Face上のモデル '{default_hf_id}' を使用します。")
                self.similarity_model_name = default_hf_id
            else:
                self.similarity_model_name = raw_sim_path
            
        self.embedding_model_name = self.config['models']['embedding']
        self.embedding_batch_size = self.config['settings'].get('embedding_batch_size', 32)

        self.bm25_retrieval_count = self.config['settings']['bm25_retrieval_count']
        self.vector_retrieval_count = self.config['settings']['vector_retrieval_count']
        self.reranker_input_count = self.config['settings']['reranker_input_count']
        # 設定ファイルにキーがない場合は、標準設定の2倍をデフォルトとする
        self.reranker_input_count_deep = self.config['settings'].get('reranker_input_count_deep', self.reranker_input_count * 2)
        self.and_search_retrieval_count = self.config['settings']['and_search_retrieval_count']
        self.and_search_context_window_chars = self.config['settings'].get('and_search_context_window_chars', 150)

        # --- デバッグ設定 ---
        self.debug_enabled = self.config.get('debug', {}).get('enabled', False)
        self.debug_save_content = self.config.get('debug', {}).get('save_chunk_content', True)
        self.debug_include_query = self.config.get('debug', {}).get('include_query_in_filename', True)

        self.initially_expanded_results_count = self.config['settings']['initially_expanded_results_count']
        self.rerank_batch_size = self.config['settings'].get('rerank_batch_size', 10)

        self.vector_chunk_size = self.config['settings']['chunking']['vector']['size']
        self.vector_chunk_overlap = self.config['settings']['chunking']['vector']['overlap']
        self.bm25_chunk_size = self.config['settings']['chunking']['bm25']['size']
        self.bm25_chunk_overlap = self.config['settings']['chunking']['bm25']['overlap']

        # バックフィル（リランキング後の穴埋め）判定基準
        self.backfill_bm25_score_drop_ratio = self.config['settings'].get('backfill_bm25_score_drop_ratio', 0.6)
        self.backfill_vector_distance_expansion_ratio = self.config['settings'].get('backfill_vector_distance_expansion_ratio', 1.5)

        self.qa_system_prompt = self.config['prompts']['qa_system_prompt']
        self.qgen_system_prompt = self.config['prompts']['qgen_system_prompt']
        self.citation_mapping_system_prompt = self.config['prompts']['citation_mapping_system_prompt']
        # 外部LLM用のプロンプトを追加（存在しない場合は通常のQAプロンプトをフォールバックとして使用）
        self.qa_system_prompt_for_all_contexts = self.config['prompts'].get('qa_system_prompt_for_all_contexts', self.qa_system_prompt)
        # クエリ最適化用プロンプト
        self.query_optimization_system_prompt = self.config['prompts'].get('query_optimization_system_prompt')
        if not self.query_optimization_system_prompt:
            logger.warning("設定ファイルに 'query_optimization_system_prompt' が見つかりません。クエリ最適化機能が正しく動作しない可能性があります。")

        # --- UI設定 ---
        self.show_citation_context_numbers = self.config.get('ui', {}).get('show_citation_context_numbers', True)

        self.reranker_model = reranker.get_reranker_model(self.reranker_model_name)
        self.tokenizer = search_engines.get_sudachi_tokenizer_instance()
        
        # 日付正規化用インスタンスの初期化
        self.date_standardizer = DateStandardizer()

        self.embedding_model = search_engines.get_embedding_model(self.embedding_model_name)
        # パス解決済みのモデル名を渡してインスタンス化
        self.similarity_model = search_engines.get_similarity_model_instance(self.similarity_model_name)

        self.loaded_file_info_list = []
        self.all_loaded_documents = []
        self.document_chunks_for_bm25 = []
        self.document_chunks_for_vector_search = []
        self.bm25_model = None
        self.vector_table = None
        if not self.tokenizer or not self.embedding_model or not self.similarity_model:
            logger.error("必須モデルのロードに失敗しました。エンジンは正常に動作しない可能性があります。")

        # --- 必要なディレクトリをすべて作成 ---
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_persist_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_index_dir.mkdir(parents=True, exist_ok=True)
        self.system_log_dir.mkdir(parents=True, exist_ok=True) # システムログ用ディレクトリ作成
        self.search_history_path.parent.mkdir(parents=True, exist_ok=True) # 履歴ファイル用親ディレクトリ作成

        # OpenCCは日本語への変換精度に難があるため使用せず、
        # 独自の辞書ベース置換を使用します (normalize_kanjiメソッド参照)

        logger.info("ScholarScopeEngineの初期化完了。")

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = toml.load(f) # ここを変更
                logger.info(f"設定ファイル '{config_path}' を正常に読み込みました。")
                return config
        except FileNotFoundError:
            logger.critical(f"設定ファイル '{config_path}' が見つかりません。")
            raise
        except Exception as e:
            logger.critical(f"設定ファイル '{config_path}' の読み込み中にエラーが発生しました: {e}", exc_info=True)
            raise

    def _get_current_files_metadata(self):
        if not self.documents_dir.exists(): return []
        metadata = []
        for item in self.documents_dir.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                if item.suffix.lower() in ['.pdf', '.txt', '.md']:
                    metadata.append((item.name, item.stat().st_size))
        return sorted(metadata)

    def _get_indexed_files_from_vector_store(self):
        if not self.vector_table: return set()
        try:
            # テーブルの全データを一度Pandas DataFrameとしてメモリに読み込む
            all_data_df = self.vector_table.to_pandas()
            # 必要な 'source' 列だけを抜き出す
            df = all_data_df[["source"]]
            return set(Path(p).name for p in df["source"].unique())
        except Exception as e:
            logger.warning(f"ベクトルストアからのインデックス済みファイル取得中にエラー: {e}", exc_info=True)
        return set()

    def build_index(self, force_rebuild=False, incremental_update=False):
        logger.info(f"インデックス構築開始。force_rebuild: {force_rebuild}, incremental_update: {incremental_update}")

        current_file_info_list = []
        if self.documents_dir.exists():
            for item in self.documents_dir.iterdir():
                if item.is_file() and not item.name.startswith('.'):
                    current_file_info_list.append({"name": item.name, "path": str(item)})

        self.loaded_file_info_list = current_file_info_list

        if not self.loaded_file_info_list:
            logger.info("検索対象ファイルなし。インデックスをクリアします。")
            self.bm25_model = None
            self.document_chunks_for_bm25 = []
            self.vector_table = None
            return {"status": "success", "message": "検索対象ファイルがありません。"}

        docs_to_process = []
        existing_bm25_data = None
        table_name = "scholarscope_collection"
        
        if incremental_update and not force_rebuild:
            logger.info("差分更新モード: 既存のインデックス状況を確認します。")
            bm25_model, bm25_chunks, bm25_tokens = search_engines.load_bm25_data(self.bm25_index_path)
            
            temp_table = None
            try:
                db = lancedb.connect(str(self.vectorstore_persist_dir))
                if table_name in db.table_names():
                    temp_table = db.open_table(table_name)
            except Exception as e:
                logger.warning(f"既存ベクトルストアへのアクセスに失敗: {e}")

            if not bm25_model or not bm25_chunks or temp_table is None:
                logger.warning("既存インデックスが不完全なため、差分更新を中止し、全再構築に切り替えます。")
                incremental_update = False
                force_rebuild = True
            else:
                existing_bm25_data = (bm25_model, bm25_chunks, bm25_tokens)
                indexed_files = {Path(c.metadata.get("source", "")).name for c in bm25_chunks}
                new_files_info = [f for f in self.loaded_file_info_list if f["name"] not in indexed_files]

                if not new_files_info:
                    logger.info("追加すべき新規ファイルはありません。インデックスは最新です。")
                    self.bm25_model = bm25_model
                    self.document_chunks_for_bm25 = bm25_chunks
                    self.vector_table = temp_table
                    return {"status": "success", "message": "新規ファイルがないため更新をスキップしました。"}

                logger.info(f"差分更新: {len(new_files_info)} 件の新規ファイルをロードします。")
                for file_info in new_files_info:
                    docs = file_processing.load_document_from_path(file_info['path'])
                    if docs: docs_to_process.extend(docs)

        if force_rebuild:
            logger.info("全ファイルをロードして再構築します。")
            docs_to_process = []
            for file_info in self.loaded_file_info_list:
                docs = file_processing.load_document_from_path(file_info['path'])
                if docs: docs_to_process.extend(docs)
        elif not incremental_update:
             logger.info("既存インデックスを利用します（新規ロードなし）。")

        if not docs_to_process and force_rebuild:
            logger.warning("有効なドキュメントをロードできませんでした。")
            return {"status": "warning", "message": "有効なドキュメントをロードできませんでした。"}

        logger.info(f"BM25インデックス更新処理開始 (対象ドキュメント数: {len(docs_to_process)})")
        self.bm25_model, self.document_chunks_for_bm25 = search_engines.initialize_bm25_engine(
            documents=docs_to_process,
            bm25_index_path=self.bm25_index_path,
            chunk_size=self.bm25_chunk_size,
            chunk_overlap=self.bm25_chunk_overlap,
            force_rebuild=force_rebuild,
            incremental_base_data=existing_bm25_data
        )
        if not self.bm25_model: logger.warning("BM25検索エンジンの準備に失敗しました。")

        if not self.embedding_model:
            logger.error("埋め込みモデル未ロードのため、ベクトルストア関連処理をスキップします。")
            return {"status": "error", "message": "Embeddingモデルが利用できず、ベクトル検索を準備できません。"}

        chunks_for_vector = None
        if docs_to_process:
            chunks_for_vector = search_engines.get_document_chunks(
                docs_to_process,
                chunk_size=self.vector_chunk_size,
                chunk_overlap=self.vector_chunk_overlap
            )

        # --- LanceDBテーブルの初期化/更新/ロード ---
        self.vector_table = search_engines.initialize_lancedb(
            document_chunks=chunks_for_vector,
            embedding_function=self.embedding_model,
            persist_directory=str(self.vectorstore_persist_dir),
            collection_name=table_name,
            batch_size=self.embedding_batch_size,
            force_rebuild=force_rebuild
        )

        if not self.vector_table:
            logger.warning("ベクトルストアの準備に失敗しました。")

        try:
            current_metadata = self._get_current_files_metadata()
            self._save_metadata(current_metadata)
        except IOError as e:
            logger.error(f"メタデータの保存に失敗: {e}")
            return {"status": "warning", "message": f"インデックスの準備は完了しましたが、メタデータの保存に失敗しました: {e}"}

        logger.info("インデックス構築/ロード処理完了。")
        return {"status": "success", "message": "インデックスの準備が完了しました。"}

    def _filter_results(self, search_results, must_keywords=None, exclude_keywords=None, doc_name_filter=None):
        """検索結果を各種条件でフィルタリングするヘルパー関数"""
        # search_results can be list[Document] or list[tuple[Document, float]]
        has_score = len(search_results) > 0 and isinstance(search_results[0], tuple)

        if not must_keywords and not exclude_keywords and not doc_name_filter:
            return search_results

        filtered = []
        keywords = must_keywords or []

        for item in search_results:
            if has_score:
                doc, score = item
            else:
                doc, score = item, 0.0 # score is dummy

            content = doc.page_content
            match = True

            # 1. Must Keywords Check
            for kw in keywords:
                try:
                    if not re.search(re.escape(kw), content, re.IGNORECASE):
                        match = False
                        break
                except re.error as e:
                    logger.warning(f"キーワード '{kw}' でのフィルタリング中に正規表現エラー: {e}")
                    match = False
                    break
            if not match: continue

            # 2. Exclude Keywords Check
            if exclude_keywords:
                for ex_kw in exclude_keywords:
                    if ex_kw in content:
                        match = False
                        break
            if not match: continue

            # 3. Document Name Filter Check
            if doc_name_filter:
                source_name = Path(doc.metadata.get("source", "")).name.lower()
                doc_filter_terms = doc_name_filter.lower().split()
                doc_must_terms = [term for term in doc_filter_terms if not term.startswith('-')]
                doc_exclude_terms = [term[1:] for term in doc_filter_terms if term.startswith('-') and len(term) > 1]

                for term in doc_must_terms:
                    if term not in source_name:
                        match = False
                        break
                if not match: continue

                for term in doc_exclude_terms:
                    if term in source_name:
                        match = False
                        break

            if not match: continue

            # If all checks passed
            if has_score:
                filtered.append((doc, score))
            else:
                filtered.append(doc)

        return filtered

    def normalize_kanji(self, text):
        """簡体字や繁体字を日本語の標準的な漢字に変換する"""
        return normalize_kanji(text) # モジュールの関数に委譲

    def optimize_search_queries(self, user_query):
        """ユーザー入力をAIで分析し、3種類のクエリに変換する"""
        if not user_query: return None
        
        # 和暦・西暦の正規化と補完 (例: 明治21年 -> 明治21年（1888年）)
        user_query = self.date_standardizer.process_text(user_query)
        
        logger.info(f"クエリ最適化を開始: {user_query}")
        try:
            response = ollama.chat(
                model=self.ollama_model_name,
                messages=[
                    {'role': 'system', 'content': self.query_optimization_system_prompt},
                    {'role': 'user', 'content': user_query},
                ],
                options={'temperature': 0.0},
                stream=False
            )
            content = response['message']['content']
            # JSONブロックの抽出を試みる
            json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
            json_str = json_match.group(1) if json_match else content
            
            # 単純なクリーニング（Markdownの残骸など）
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            # 簡体字・繁体字対策 (日本語漢字へ正規化)
            json_str = self.normalize_kanji(json_str)
            
            optimized = json.loads(json_str)
            # 必要なキーがあるか確認
            if all(k in optimized for k in ["bm25_query", "vector_query", "rerank_query"]):
                logger.info(f"クエリ最適化成功: {optimized}")
                return optimized
            else:
                logger.warning(f"クエリ最適化結果に必須キーが不足しています: {optimized}")
                return None
        except Exception as e:
            logger.error(f"クエリ最適化中にエラー: {e}", exc_info=True)
            return None

    def _build_lancedb_filter(self, must_keywords, exclude_keywords, doc_name_filter):
        """LanceDB用のSQLフィルタ条件式を構築する"""
        conditions = []

        def escape_sql(text):
            return text.replace("'", "''")

        # Must Keywords (AND)
        if must_keywords:
            for kw in must_keywords:
                conditions.append(f"text LIKE '%{escape_sql(kw)}%'")

        # Exclude Keywords (NOT)
        if exclude_keywords:
            for kw in exclude_keywords:
                conditions.append(f"text NOT LIKE '%{escape_sql(kw)}%'")

        # Doc Name Filter (source path)
        if doc_name_filter:
            doc_filter_terms = doc_name_filter.lower().split()
            for term in doc_filter_terms:
                safe_term = escape_sql(term.replace("-", ""))
                if term.startswith('-') and len(term) > 1:
                    conditions.append(f"LOWER(source) NOT LIKE '%{safe_term}%'")
                else:
                    conditions.append(f"LOWER(source) LIKE '%{safe_term}%'")

        return " AND ".join(conditions) if conditions else None

    def search(self, must_keywords, semantic_query, doc_name_filter=None, exclude_keywords=None, skip_rerank=False, deep_search=False, optimized_queries=None, callback=None):
        logger.info(f"検索実行。DeepMode: {deep_search}, must_keywords: '{must_keywords[:100]}...', semantic_query: '{semantic_query[:100]}...'")
        # startコールバックは呼び出し側で制御するため削除、または無視される
        
        # 意味検索クエリの日付正規化（最適化OFF時やバックフィルキーワード抽出のために実施）
        # すでにoptimize_search_queriesで変換済みの場合でも、DateStandardizerは冪等性があるため問題ない
        if semantic_query:
            semantic_query = self.date_standardizer.process_text(semantic_query)
        
        # Deep Searchモードの場合は件数を拡張
        current_rerank_count = self.reranker_input_count_deep if deep_search else self.reranker_input_count
        current_bm25_top_n = max(self.bm25_retrieval_count, current_rerank_count)
        current_vector_top_n = max(self.vector_retrieval_count, current_rerank_count)

        results = {}
        results["meta"] = {"and_has_more": False} # メタ情報初期化
        must_keyword_list = [kw.strip() for kw in must_keywords.split() if kw.strip()] if must_keywords else []

        # クエリ最適化が使われていない場合、ベクトル検索のバックフィルチェック用に
        # セマンティッククエリを単語リスト化しておく（簡易的なキーワード抽出）
        semantic_keywords = []
        if semantic_query:
             semantic_keywords = semantic_query.replace("　", " ").split()

        # --- ケース1: 「絞り込みキーワード」のみ (AND検索) ---
        if must_keywords and not semantic_query:
            logger.info("AND検索フローを実行します。")
            if self.document_chunks_for_bm25:
                # 1. まず件数制限なしで全ての結果を取得
                all_and_results = search_engines.search_and_完全一致(
                    must_keywords, self.document_chunks_for_bm25,
                    case_sensitive=False, context_window_chars=self.and_search_context_window_chars,
                    top_n=self.and_search_retrieval_count # search_engines側で+1件取得する
                )
                # 2. 次にフィルタリングを実行
                if callback: callback('filter', 'running', "条件フィルタリング中...")
                filtered_results = self._filter_results(
                    all_and_results, exclude_keywords=exclude_keywords, doc_name_filter=doc_name_filter
                )
                # 3. 最後に件数制限を適用
                if len(filtered_results) > self.and_search_retrieval_count:
                    results["meta"]["and_has_more"] = True
                    results["and"] = filtered_results[:self.and_search_retrieval_count]
                else:
                    results["and"] = filtered_results
            else:
                # AND検索パスでもUIの整合性のためにダミーコールバックなどを送るのが理想だが、今回は結果のみ
                if callback: callback('retrieval', 'done', f"AND検索で {len(results['and'])} 件ヒット")
            return results

        # --- ケース2: 「質問・関心事」が入力されている (意味検索) ---
        if semantic_query:
            logger.info("意味検索フローを実行します。")
            bm25_results, vector_results = [], []

            # 最適化されたクエリがある場合はそれを使用、なければ元のクエリを使用
            bm25_query_text = semantic_query
            vector_query_text = semantic_query
            rerank_query_text = semantic_query

            if optimized_queries:
                bm25_query_text = optimized_queries.get('bm25_query', semantic_query)
                vector_query_text = optimized_queries.get('vector_query', semantic_query)
                rerank_query_text = optimized_queries.get('rerank_query', semantic_query)

            if callback: callback('retrieval', 'running', "実行中...")

            # Pre-filtering準備: フィルタがある場合は検索段階で考慮する
            has_filter = bool(must_keyword_list or exclude_keywords or doc_name_filter)
            
            # BM25: フィルタがある場合は全件取得(top_n=None)してから絞る (確実性重視)
            target_bm25_top_n = None if has_filter else current_bm25_top_n
            
            # Vector: LanceDBのNative Filter (where句) を使用
            lancedb_filter = self._build_lancedb_filter(must_keyword_list, exclude_keywords, doc_name_filter) if has_filter else None

            # 1. ベースとなる意味検索を実行（広めに取得する）
            # リランキングの機会損失を防ぐため、設定値より多め(例: 30件)に取得してから統合する
            # ここでは config の値またはDeep Search用の拡張値を使う
            if self.bm25_model and self.document_chunks_for_bm25:
                bm25_results = search_engines.search_bm25(bm25_query_text, self.bm25_model, self.document_chunks_for_bm25, self.tokenizer, top_n=target_bm25_top_n)

            if self.vector_table:
                vector_results = search_engines.search_vector(vector_query_text, self.vector_table, self.embedding_model, top_n=current_vector_top_n, filter_condition=lancedb_filter)

            if callback:
                callback('retrieval', 'done', f"キーワード検索: {len(bm25_results)} 件, ベクトル検索: {len(vector_results)} 件")

            # 2. フィルタリングを適用
            logger.info("意味検索結果を各種条件でフィルタリングします。")
            if callback: callback('filter', 'running', "条件フィルタリングと候補統合を実行中...")
            
            bm25_results = self._filter_results(bm25_results, must_keywords=must_keyword_list, exclude_keywords=exclude_keywords, doc_name_filter=doc_name_filter)
            vector_results = self._filter_results(vector_results, must_keywords=must_keyword_list, exclude_keywords=exclude_keywords, doc_name_filter=doc_name_filter)

            # BM25で全件取得していた場合、ここで本来の件数にスライスする
            if has_filter:
                bm25_results = bm25_results[:current_bm25_top_n]

            # フィルタ後の結果を格納
            results["bm25"] = bm25_results
            results["vector"] = vector_results

            # フィルタ完了通知
            if callback: callback('filter', 'done', f"候補絞り込み (一次取得数: {len(bm25_results)+len(vector_results)}件)")
            # 3. リランキングを実行 (スキップしない場合)
            if not skip_rerank and self.reranker_model:
                # --- 候補の最大化 ---
                # 単純に「上位10件+10件」ではなく、統合して重複排除したリストから
                # 「リランキング定員(例:20件)」を埋めることで、スロットの無駄をなくす。
                
                # リランキング定員 = current_rerank_count * 2 (例: 30 * 2 = 60)
                rerank_budget = current_rerank_count * 2
                
                # 両方の結果を統合（BM25優先で並べる）
                docs_to_rerank_bm25 = [doc for doc, score in bm25_results]
                docs_to_rerank_vector = [doc for doc, score in vector_results]
                
                # 単純結合だとBM25がリストの前半を独占し、Vector結果がリランク定員から溢れるため、
                # 交互(ラウンドロビン)に並べてから重複排除を行う
                docs_to_rerank = []
                max_len = max(len(docs_to_rerank_bm25), len(docs_to_rerank_vector))
                for i in range(max_len):
                    if i < len(docs_to_rerank_bm25):
                        docs_to_rerank.append(docs_to_rerank_bm25[i])
                    if i < len(docs_to_rerank_vector):
                        docs_to_rerank.append(docs_to_rerank_vector[i])

                if docs_to_rerank:
                    # 重複排除 (Text contentベース)
                    unique_docs_to_rerank = list({doc.page_content: doc for doc in docs_to_rerank}.values())
                    
                    # 定員までカット
                    docs_for_reranker = unique_docs_to_rerank[:rerank_budget]
                    
                    logger.info(f"リランキング処理開始。候補数: {len(unique_docs_to_rerank)} -> 実行対象: {len(docs_for_reranker)}")
                    if callback: callback('rerank', 'running', f"上位 {len(docs_for_reranker)} 件の精査を開始（Deep Mode: {'ON' if deep_search else 'OFF'}）")
                    
                    def rerank_progress_wrapper(curr, total):
                        if callback: callback('rerank', 'running', current=curr, total=total)
                    
                    reranked_results = reranker.rerank_documents(
                        query=rerank_query_text,
                        documents=docs_for_reranker,
                        reranker=self.reranker_model,
                        top_n=len(docs_for_reranker),
                        progress_callback=rerank_progress_wrapper,
                        batch_size=self.rerank_batch_size
                    )
                    results["reranked"] = reranked_results
                    if callback: callback('rerank', 'done', "リランキング完了。関連順に並べ替えました。")
                else:
                    if callback: callback('rerank', 'done', "リランク対象となる候補がありませんでした。")
                    results["reranked"] = []
            else:
                if callback: callback('rerank', 'skipped')
                results["reranked"] = []

            # デバッグログ出力
            if self.debug_enabled:
                self._save_debug_log(
                    must_keywords, semantic_query, optimized_queries, 
                    deep_search, skip_rerank, results, 
                    current_rerank_count
                )

            return results

        # --- ケース3: どちらも入力なし (UI側でブロックされるが念のため) ---
        logger.warning("検索キーワードが両方とも空です。")
        return {}

    def check_index_health(self):
        logger.info("インデックスのヘルスチェックを開始...")
        start_time_total = time.time()
        if not self.embedding_model:
            logger.error("ヘルスチェック中止: 埋め込み関数が利用できません。")
            return {"vector_store_status": "error_no_embedding_model"}
        vector_store_status = "unknown"
        try:
            # 1. DBクライアント初期化
            start_time_connect = time.time()
            db = lancedb.connect(str(self.vectorstore_persist_dir))
            elapsed_connect = (time.time() - start_time_connect) * 1000
            logger.info(f"ヘルスチェック: DB接続完了 ({elapsed_connect:.2f} ms)")

            table_name = "scholarscope_collection"
            if table_name in db.table_names():
                start_time_open = time.time()
                table = db.open_table(table_name)
                elapsed_open = (time.time() - start_time_open) * 1000
                logger.info(f"ヘルスチェック: テーブルオープン完了 ({elapsed_open:.2f} ms)")

                if table.count_rows() > 0:
                    vector_store_status = "healthy"
                    logger.info("ヘルスチェック結果: healthy (テーブルにデータが存在します)")
                else:
                    vector_store_status = "empty_collection"
                    logger.info("ヘルスチェック結果: empty_collection (テーブルは空です)")
            else:
                vector_store_status = "not_found"
                logger.warning(f"ヘルスチェック結果: not_found (テーブル '{table_name}' が見つかりません)")

        except Exception as e:
            vector_store_status = "error_accessing_db"
            logger.error(f"ヘルスチェック中に予期せぬエラーが発生しました: {e}", exc_info=True)

        elapsed_total = (time.time() - start_time_total) * 1000
        logger.info(f"インデックスのヘルスチェック完了。最終ステータス: '{vector_store_status}', 所要時間: {elapsed_total:.2f} ms")
        return {"vector_store_status": vector_store_status}

    def generate_prompt_for_llm(self, search_results_dict, question_text, max_tokens):
        # --- 新しいプロンプト生成ロジック ---
        # UI表示件数とは独立して、利用可能な全ての検索結果を候補とする

        candidate_list = []
        reranked_res = search_results_dict.get("reranked", [])
        bm25_res = search_results_dict.get("bm25", [])
        vector_res = search_results_dict.get("vector", [])
        and_res = search_results_dict.get("and", [])

        # 重複チェック用のセット（テキスト内容で判定）
        seen_contents = set()

        # 優先度1: AND検索の結果 (「絞り込みキーワード」のみで検索した場合)
        if and_res:
            for doc in and_res:
                if doc.page_content not in seen_contents:
                    candidate_list.append((doc, 1.0)) # スコアはダミー
                    seen_contents.add(doc.page_content)

        # 優先度2: リランキングの結果 (Top-K保証 + 足切り)
        if reranked_res:
            logger.info("リランキング結果を適用（Top-K保証 + 足切り）")
            GUARANTEE_COUNT = 5     # 上位5件は無条件採用
            FILTER_THRESHOLD = 0.01 # それ以降はスコア0.01以下で足切り
            
            for i, (doc, score) in enumerate(reranked_res):
                if doc.page_content in seen_contents: continue
                
                # --- 【改善 Phase 2】 フィルタリング ---
                if i < GUARANTEE_COUNT:
                    # 保証枠: そのまま追加
                    candidate_list.append((doc, score))
                    seen_contents.add(doc.page_content)
                elif score > FILTER_THRESHOLD:
                    # 通常枠: 閾値チェック
                    candidate_list.append((doc, score))
                    seen_contents.add(doc.page_content)
                else:
                    # 足切り対象
                    pass

        # 優先度3: BM25とVectorのバックフィル (守りのバックフィル)
        if bm25_res or vector_res:
            logger.info("残りのBM25/Vector結果でバックフィルを行います。")
            
            # クエリに含まれるキーワード（ベクトル検索の安全性チェック用）
            # SudachiPyを使って厳密にキーワードを抽出
            query_keywords = search_engines.tokenize_text_sudachi_util(
                question_text, self.tokenizer, 
                source_info="バックフィル用キーワード抽出",
                pos_filter_active=True
            )

            # --- ボーダーラインスコアの取得 (相対スコアフィルタ用) ---
            # リランキング対象候補群(上位層)のスコアを基準にする
            bm25_top_score = bm25_res[0][1] if bm25_res else 0
            vector_top_dist = vector_res[0][1] if vector_res else 0
            
            max_len = max(len(bm25_res), len(vector_res))
            for i in range(max_len):
                # --- 守りのバックフィル ---
                
                # 1. BM25を優先 (キーワード確実 + スコアの崖チェック)
                if i < len(bm25_res):
                    doc, score = bm25_res[i]
                    if doc.page_content not in seen_contents:
                        # 相対スコアチェック: 基準よりガクッと落ちていないか
                        if score >= bm25_top_score * self.backfill_bm25_score_drop_ratio:
                            candidate_list.append((doc, score))
                            seen_contents.add(doc.page_content)
                        else:
                            # スコアが閾値を下回ったため、これ以降(i+1...)も全て下回るはずなので実質的に終了
                            pass
                
                # 2. Vectorは「距離チェック」AND「キーワードチェック」に合格した場合のみ採用
                if i < len(vector_res):
                    doc, score = vector_res[i]
                    if doc.page_content not in seen_contents:
                        
                        # A. 距離チェック (トップ層から離れすぎていないか)
                        is_distance_ok = False
                        if vector_top_dist == 0: # 完全一致があった場合
                            is_distance_ok = (score < 0.3) # 絶対値でガード
                        else:
                            is_distance_ok = (score <= vector_top_dist * self.backfill_vector_distance_expansion_ratio)
                        
                        if not is_distance_ok: continue

                        # キーワード簡易チェック: 質問文中の単語が一つでも含まれているか
                        has_keyword = False
                        if not query_keywords: # キーワードがない(質問がない)場合は許可せざるを得ない
                            has_keyword = True
                        else:
                            for kw in query_keywords:
                                if kw in doc.page_content:
                                    has_keyword = True
                                    break
                        
                        if has_keyword:
                            candidate_list.append((doc, score))
                            seen_contents.add(doc.page_content)

        # 最終的なリスト作成 (candidate_list は既に重複排除済み)
        unique_docs = [item[0] for item in candidate_list]

        # トークン数上限までコンテキストを選択
        context_docs, current_tokens = [], 0
        for doc in unique_docs:
            chunk_tokens = utils.count_tokens(doc.page_content)
            if current_tokens + chunk_tokens <= max_tokens:
                context_docs.append(doc)
                current_tokens += chunk_tokens
            else:
                break

        if not context_docs and unique_docs:
            logger.warning("プロンプト用コンテキストがありません (最初のチャンクがトークン上限を超えている可能性があります)。")

        # コンテキスト文字列と引用マップを生成
        context_parts = []
        citation_map = {} # {番号: ファイル名} のマップ
        for i, doc in enumerate(context_docs):
            source_filename = Path(doc.metadata.get('source', 'N/A')).name
            context_number = i + 1
            citation_map[context_number] = source_filename

            context_header = f"[{context_number}] 【出典: {source_filename}】"
            context_parts.append(f"{context_header}\n{doc.page_content}")

        contexts_string = "\n---\n".join(context_parts)
        logger.info(f"プロンプト用コンテキスト生成完了。{len(context_docs)}件、約{current_tokens}トークン。")

        return {
            "contexts_string": contexts_string,
            "context_docs": context_docs,
            "total_tokens": current_tokens,
            "citation_map": citation_map
        }

    def get_loaded_files_info(self):
        return self.loaded_file_info_list

    def get_all_deduplicated_contexts_string(self, search_results_dict):
        all_raw_docs = []
        if search_results_dict.get("reranked"): all_raw_docs.extend([doc for doc, score in search_results_dict["reranked"]])
        if search_results_dict.get("and"): all_raw_docs.extend(search_results_dict["and"])
        if search_results_dict.get("bm25"): all_raw_docs.extend([doc for doc, score in search_results_dict["bm25"]])
        if search_results_dict.get("vector"): all_raw_docs.extend([doc for doc, score in search_results_dict["vector"]])
        if not all_raw_docs: return "", 0

        temp_integrated_list, seen_identifiers = [], set()
        for doc in all_raw_docs:
            source_name = Path(doc.metadata.get("source", "N/A")).name
            page = str(doc.metadata.get("page", "N/A"))
            prefix = doc.page_content[:50]
            identifier = (source_name, page, prefix)
            if identifier not in seen_identifiers:
                temp_integrated_list.append(doc)
                seen_identifiers.add(identifier)

        # ここで self.similarity_model を渡すことで、オフラインロードしたモデルが使われる
        final_deduplicated_docs = search_engines.deduplicate_chunks(
            temp_integrated_list, 
            similarity_threshold=self.deduplication_threshold,
            model_instance=self.similarity_model
        )

        context_parts = []
        for i, doc_obj in enumerate(final_deduplicated_docs):
            source_filename = Path(doc_obj.metadata.get('source', 'N/A')).name
            context_number = i + 1
            context_header = f"[{context_number}] 出典: {source_filename}"
            context_parts.append(f"{context_header}\n---\n{doc_obj.page_content}")

        return "\n\n".join(context_parts), len(final_deduplicated_docs)

    def _load_metadata(self):
        if self.metadata_file_path.exists():
            try:
                with open(self.metadata_file_path, 'r', encoding='utf-8') as f:
                    return [tuple(item) for item in json.load(f)]
            except Exception as e:
                logger.warning(f"メタデータファイル ({self.metadata_file_path}) の読み込みに失敗: {e}")
                return []
        return []

    def _save_metadata(self, metadata_list):
        try:
            with open(self.metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
            logger.info(f"現在のファイルメタデータを ({self.metadata_file_path}) に保存しました。")
        except Exception as e:
            logger.error(f"メタデータファイル ({self.metadata_file_path}) の保存中にエラー: {e}", exc_info=True)
            raise IOError(f"メタデータファイルの保存に失敗しました: {e}")

    def check_and_decide_build_action(self):
        logger.info("インデックス構築/更新の要否を判断中...")

        last_saved_files_metadata = self._load_metadata()
        current_files_meta_on_disk = self._get_current_files_metadata()
        health_check_result = self.check_index_health()
        health_status = health_check_result.get("vector_store_status", "unknown")
        logger.info(f"方針判断: ヘルスチェック結果 = '{health_status}'")

        current_files_meta_set = set(current_files_meta_on_disk)
        last_saved_files_meta_set = set(last_saved_files_metadata)
        files_on_disk_names = {meta[0] for meta in current_files_meta_on_disk}
        last_saved_files_names = {meta[0] for meta in last_saved_files_metadata}
        added_files = files_on_disk_names - last_saved_files_names
        deleted_files = last_saved_files_names - files_on_disk_names
        changed_files = {f for f, s in current_files_meta_set if (f, s) not in last_saved_files_meta_set and f in last_saved_files_names}
        logger.info(f"方針判断: ファイル比較結果 - 追加: {len(added_files)}件, 削除: {len(deleted_files)}件, 変更: {len(changed_files)}件")

        if health_status == "error_no_embedding_model":
            logger.info("方針判断: 決定 = STOP (埋め込みモデルなし)")
            return "STOP", "ベクトルストアに必要なAIモデルがロードできませんでした。"

        if health_status in ["not_found", "error_accessing_db"] and current_files_meta_on_disk:
            logger.info("方針判断: 決定 = FORCE_REBUILD (インデックス不在/アクセス不可)")
            return "FORCE_REBUILD", "ベクトルストアが存在しないかアクセスできないため、インデックスの再構築が必要です。"

        if not current_files_meta_on_disk:
            logger.info("方針判断: 決定 = LOAD_EMPTY (ドキュメントフォルダが空)")
            return "LOAD_EMPTY", f"検索対象ファイルがありません ('{self.config['paths']['documents_folder']}' フォルダは空です)。"

        if current_files_meta_set == last_saved_files_meta_set:
            logger.info("方針判断: ファイル構成に変更なし。")
            if health_status == "healthy":
                logger.info("方針判断: 決定 = LOAD (インデックス正常)")
                return "LOAD", "ファイル構成に変更はなく、インデックスは正常です。"
            else:
                logger.info(f"方針判断: 決定 = ASK_REBUILD (インデックス不完全: {health_status})")
                return "ASK_REBUILD", f"ファイル構成は同じですが、インデックスが不完全です (状態: {health_status})。再構築を推奨します。"

        if added_files and not deleted_files and not changed_files:
            logger.info("方針判断: ファイルが追加されたのみ。")
            if health_status in ["healthy", "empty_collection"]:
                logger.info("方針判断: 決定 = ASK_INCREMENTAL (差分更新を提案)")
                return "ASK_INCREMENTAL", f"{len(added_files)}件のファイルが追加されました。差分更新または全再構築を選択してください。"
            else:
                logger.info(f"方針判断: 決定 = ASK_REBUILD (差分更新不可、インデックス不完全: {health_status})")
                return "ASK_REBUILD", f"ファイルが追加されましたが、既存インデックスが不完全なため、全再構築を推奨します。"

        details = []
        if added_files: details.append(f"{len(added_files)}件追加")
        if deleted_files: details.append(f"{len(deleted_files)}件削除")
        if changed_files: details.append(f"{len(changed_files)}件変更")
        logger.info(f"方針判断: 決定 = ASK_REBUILD (ファイル構成が複雑に変更: {', '.join(details)})")
        return "ASK_REBUILD", f"ファイル構成が変更されています ({', '.join(details)})。インデックスの全再構築を推奨します。"

    def stream_clean_answer(self, user_prompt, temperature=0.2):
        """LLM Call #1 を実行し、典拠なしの回答をストリーミングで生成する。"""
        try:
            stream = ollama.chat(
                model=self.ollama_model_name,
                messages=[
                    {'role': 'system', 'content': self.qa_system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                stream=True,
                options={'temperature': temperature}
            )
            
            def _stream_processor():
                for chunk in stream:
                    content = chunk['message']['content']
                    # 簡体字・繁体字対策
                    content = self.normalize_kanji(content)
                    yield content
            
            return _stream_processor()
        except Exception as e:
            error_msg = f"Ollamaとの通信中にエラーが発生しました: {e}"
            logger.error(error_msg, exc_info=True)
            def error_generator():
                yield error_msg
            return error_generator()

    def get_citation_map_from_answer(self, answer_text, contexts_string):
        """LLM Call #2 を実行し、回答とコンテキストの典拠マッピングを取得する。"""
        logger.info("LLM Call #2 (典拠マッピング) を開始します。")
        if not self.citation_mapping_system_prompt:
            logger.error("典拠マッピング用のシステムプロンプトがconfig.tomlに見つかりません。")
            return {}

        # 回答の各箇条書きにIDを付与（セクションヘッダーは無視）
        tagged_answer_parts = []
        # 箇条書きまたは番号付きリストの行にマッチする正規表現
        bullet_pattern = re.compile(r"^\s*([\*\-]|(?:\d+\.))\s+")
        # セクションヘッダーにマッチする正規表現
        header_pattern = re.compile(r"^\s*【.*?】")
        lines = answer_text.strip().split('\n')
        item_counter = 1

        for line in lines:
            line = line.strip()
            if not line or header_pattern.match(line):
                continue # 空行やヘッダー行はスキップ

            # 箇条書きの開始行かどうかを判定
            if bullet_pattern.match(line):
                # 箇条書きのマーカーを除いたテキスト部分を取得
                content_part = bullet_pattern.sub("", line)
                tagged_answer_parts.append(f"[A{item_counter}] {content_part.strip()}")
                item_counter += 1

        if not tagged_answer_parts:
            logger.warning("回答から典拠付与対象の箇条書き項目が見つかりませんでした。")
            return {}

        tagged_answer_string = "\n".join(tagged_answer_parts)
        user_prompt = f"""### 回答\n{tagged_answer_string}\n\n### コンテキスト\n{contexts_string}"""

        try:
            response = ollama.chat(
                model=self.ollama_model_name,
                messages=[
                    {'role': 'system', 'content': self.citation_mapping_system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                stream=False,
                options={'temperature': 0.0}
            )
            response_content = response['message']['content']
            # マークダウンのJSONブロックからJSON文字列を抽出
            json_match = re.search(r"```json\n(.*?)\n```", response_content, re.DOTALL)
            json_string = json_match.group(1) if json_match else response_content

            citation_map = json.loads(json_string)
            logger.info(f"典拠マッピング取得成功: {citation_map}")
            return citation_map
        except json.JSONDecodeError as e:
            logger.error(f"LLMからの応答JSONのパースに失敗しました: {e}\n応答内容: {response_content}", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"Ollamaとの通信中にエラーが発生しました (get_citation_map_from_answer): {e}", exc_info=True)
            return {}

    def save_ai_report(self, report_data):
        """AIレポートとメタデータをデバッグ用テキストファイルに保存する"""
        if not self.save_ai_reports:
            return

        try:
            timestamp_str = time.strftime('%Y%m%d_%H%M%S')

            # ファイル名用のクエリパート作成（検索時のクエリを使用）
            # 優先度: 検索時の意味クエリ > 検索時のキーワード > (無ければレポート用質問)
            q_source = report_data.get("search_query_semantic") or report_data.get("query_keywords") or report_data.get("query_semantic", "")

            # ファイル名に使えない文字を置換
            q = str(q_source)
            q = re.sub(r'[\\/*?:"<>|]', "", q)[:30] # 長くなりすぎないように制限
            query_part = f"_{q}" if q else ""

            # 検索デバッグログ(debug_search_...)と対になるようなファイル名
            filename = f"debug_ai_report_{timestamp_str}{query_part}.txt"
            filepath = self.system_log_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== AI REPORT DEBUG LOG ===\n")
                f.write(f"Date: {report_data.get('timestamp')}\n")
                f.write(f"Settings: DeepSearch={report_data['settings'].get('deep_search')}, SkipRerank={report_data['settings'].get('skip_rerank')}\n")
                
                f.write(f"\n--- Search Conditions ---\n")
                f.write(f"Must Keywords (Search): {report_data.get('query_keywords')}\n")
                f.write(f"Semantic Query (Search): {report_data.get('search_query_semantic')}\n")
                
                f.write(f"\n--- Report Request ---\n")
                f.write(f"User Question (Right Pane): {report_data.get('query_semantic')}\n")
                
                f.write(f"\n--- Final Answer ---\n")
                f.write(report_data.get('final_answer', ''))
                
                f.write(f"\n\n--- Citation Contexts ---\n")
                for ctx in report_data.get('citation_contexts', []):
                    f.write(f"- {ctx}\n")

            logger.info(f"AIレポートをデバッグファイルに保存しました: {filepath}")
        except Exception as e:
            logger.error(f"AIレポートの保存中にエラー: {e}")

    def _save_debug_log(self, must_keywords, semantic_query, optimized_queries, deep_search, skip_rerank, results, rerank_count):
        """検索結果の詳細なデバッグログをファイルに出力する"""
        try:
            timestamp_str = time.strftime('%Y%m%d_%H%M%S')
            query_part = ""
            if self.debug_include_query:
                q = semantic_query or must_keywords or "no_query"
                # ファイル名に使えない文字を置換
                q = re.sub(r'[\\/*?:"<>|]', "", q)[:30]
                query_part = f"_{q}"
            
            filename = f"debug_search_{timestamp_str}{query_part}.txt"
            filepath = self.system_log_dir / filename

            # データの統合と整形
            # Key=(Filename, Page, StartText) -> Value={Stats}
            stats = {}

            def _get_key(doc):
                src = doc.metadata.get("source", "N/A")
                filename = Path(src).name
                page = doc.metadata.get("page")
                # ページ番号の正規化（Noneや文字列のケア）
                if page is None or page == "":
                    page_str = ""
                else:
                    page_str = str(page)
                return (filename, page_str, doc.page_content[:30])

            # 初期化ヘルパー
            def _ensure_entry(k, doc):
                if k not in stats:
                    stats[k] = {
                        "doc": doc,
                        "rerank_rank": 9999, "rerank_score": None,
                        "bm25_rank": 9999, "bm25_score": None,
                        "vector_rank": 9999, "vector_score": None,
                        "is_and_hit": False
                    }

            # 1. BM25結果
            for rank, (doc, score) in enumerate(results.get("bm25", []), 1):
                k = _get_key(doc)
                _ensure_entry(k, doc)
                stats[k]["bm25_rank"] = rank
                stats[k]["bm25_score"] = score

            # 2. Vector結果
            for rank, (doc, score) in enumerate(results.get("vector", []), 1):
                k = _get_key(doc)
                _ensure_entry(k, doc)
                stats[k]["vector_rank"] = rank
                stats[k]["vector_score"] = score

            # 3. Rerank結果 (常に更新)
            for rank, (doc, score) in enumerate(results.get("reranked", []), 1):
                k = _get_key(doc)
                _ensure_entry(k, doc)
                stats[k]["rerank_rank"] = rank
                stats[k]["rerank_score"] = score

            # 4. AND検索結果
            for doc in results.get("and", []):
                k = _get_key(doc)
                _ensure_entry(k, doc)
                stats[k]["is_and_hit"] = True

            # --- ソート処理 ---
            # 優先順位: Rerank順 -> BM25順 -> Vector順
            sorted_keys = sorted(
                stats.keys(),
                key=lambda k: (
                    stats[k]["rerank_rank"],
                    stats[k]["bm25_rank"],
                    stats[k]["vector_rank"]
                )
            )

            with open(filepath, "w", encoding="utf-8") as f:
                # --- Header ---
                f.write(f"=== SEARCH DEBUG LOG ===\n")
                f.write(f"Date: {timestamp_str}\n")
                f.write(f"Settings: DeepSearch={deep_search}, SkipRerank={skip_rerank}, RerankCount={rerank_count}\n")
                f.write(f"Input Must Keywords: {must_keywords}\n")
                f.write(f"Input Semantic Query: {semantic_query}\n")
                if optimized_queries:
                    f.write(f"Optimized Queries: {json.dumps(optimized_queries, ensure_ascii=False)}\n")
                f.write("\n")

                # --- Ranking Table ---
                # No. | Final(Rerank) | RerankScore | BM25(Rank/Score) | Vector(Rank/Score) | Source
                header_fmt = "{:<4} | {:<8} | {:<10} | {:<20} | {:<20} | {}"
                row_fmt    = "{:<4} | {:<8} | {:<10} | {:<20} | {:<20} | {}"
                
                f.write(header_fmt.format("No.", "Final", "RrScore", "BM25 (Rank:Score)", "Vector (Rank:Score)", "Source (Page)") + "\n")
                f.write("-" * 130 + "\n")

                for idx, k in enumerate(sorted_keys, 1):
                    stat = stats[k]
                    doc = stat["doc"]
                    
                    # Source Info
                    filename, page_str, _ = k
                    source_info = filename
                    if page_str:
                        source_info += f" (p.{page_str})"
                    
                    # Scores / Ranks
                    final_rank_disp = str(stat["rerank_rank"]) if stat["rerank_rank"] != 9999 else "-"
                    rr_score_disp = f"{stat['rerank_score']:.4f}" if stat["rerank_score"] is not None else "-"
                    
                    bm25_disp = "-"
                    if stat["bm25_rank"] != 9999:
                        bm25_disp = f"{stat['bm25_rank']}:{stat['bm25_score']:.4f}"

                    vec_disp = "-"
                    if stat["vector_rank"] != 9999:
                        vec_disp = f"{stat['vector_rank']}:{stat['vector_score']:.4f}"

                    f.write(row_fmt.format(str(idx), final_rank_disp, rr_score_disp, bm25_disp, vec_disp, source_info) + "\n")

                # --- Content Details ---
                if self.debug_save_content:
                    f.write("\n\n=== CHUNK CONTENTS ===\n")
                    for idx, k in enumerate(sorted_keys, 1):
                        stat = stats[k]
                        doc = stat["doc"]
                        filename, page_str, _ = k
                        pg_info = f" (p.{page_str})" if page_str else ""
                        
                        f.write(f"\n--- No.{idx} | Final:{stat['rerank_rank'] if stat['rerank_rank'] != 9999 else '-'} | {filename}{pg_info} ---\n")
                        f.write(doc.page_content)
                        f.write("\n")

            logger.info(f"デバッグログを保存しました: {filepath}")
        except Exception as e:
            logger.error(f"デバッグログ保存中にエラー: {e}", exc_info=True)

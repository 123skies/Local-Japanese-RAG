# project_root/src/modules/search_engines.py
import streamlit as st
import os
import pickle
import shutil
import re
import time
from sudachipy import Dictionary, Tokenizer as SudachiTokenizer
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, util
import numpy as np
import sudachidict_core
import hashlib
import logging
import lancedb
import pandas as pd
import pyarrow as pa
import unicodedata

logger_se = logging.getLogger('scholarscope_lite')

SIMILARITY_MODEL_NAME = "pkshatech/simcse-ja-bert-base-clcmlp"
CONTEXT_WINDOW_CHARS_BM25 = 200

@st.cache_resource
def get_sudachi_tokenizer_instance(dict_type="core"):
    try:
        logger_se.info(f"SudachiPy Tokenizer ({dict_type}) の初期化を開始します。")
        # ポータブル環境対策: sudachidict_coreのインストール先から辞書パスを動的に特定
        dict_dir = os.path.dirname(sudachidict_core.__file__)
        # 辞書ファイルの場所はバージョンにより直下または resources/ 以下の場合があるため両方探す
        search_dirs = [dict_dir, os.path.join(dict_dir, "resources")]
        possible_names = ["system.dic", "core.dic"]
        dict_path = None
        for d in search_dirs:
            if dict_path: break
            for name in possible_names:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    dict_path = p
                    break
        
        # ファイルが見つかればパス指定、見つからなければデフォルト文字列指定で初期化
        if dict_path:
            # Windowsパスのバックスラッシュ問題を避けるため、正規化して渡す
            import pathlib
            safe_path = str(pathlib.Path(dict_path).resolve())
            tokenizer = Dictionary(dict=safe_path).create()
            logger_se.info(f"辞書ファイルをロードしました: {safe_path}")
        else:
            logger_se.warning(f"辞書ファイルが {dict_dir} 内に見つかりませんでした。デフォルト設定('core')で初期化を試みます。")
            tokenizer = Dictionary(dict="core").create()
            
        logger_se.info(f"SudachiPy Tokenizer ({dict_type}) の初期化成功。")
        return tokenizer
    except Exception as e:
        logger_se.error(f"SudachiPy Tokenizerの初期化に失敗しました: {e}", exc_info=True)
        return None

def tokenize_text_sudachi_util(text, tokenizer_instance, source_info="不明なソース",
                               pos_filter_active=False,
                               target_pos_major=None,
                               use_normalized_form_for_bm25_pos=None):
    if not tokenizer_instance or not text: return []
    if target_pos_major is None: target_pos_major = ["名詞", "動詞", "形容詞", "副詞", "形状詞"]
    if use_normalized_form_for_bm25_pos is None: use_normalized_form_for_bm25_pos = ["動詞", "形容詞"]
    try:
        mode = SudachiTokenizer.SplitMode.C
        tokens = []
        for m in tokenizer_instance.tokenize(text, mode):
            if pos_filter_active:
                part_of_speech_tuple = m.part_of_speech()
                pos_major = part_of_speech_tuple[0]
                if pos_major in target_pos_major:
                    token_to_add = m.normalized_form() if pos_major in use_normalized_form_for_bm25_pos else m.surface()
                    tokens.append(token_to_add)
            else:
                tokens.append(m.surface())
        return [token for token in tokens if token and token.strip()]
    except Exception as e:
        log_message = f"トークン化エラー。ソース: {source_info}, テキスト冒頭: {text[:100]}..., エラー: {e}"
        if "Input is too long" in str(e) or isinstance(e, ValueError):
            logger_se.warning(log_message, exc_info=False)
        else:
            logger_se.warning(log_message, exc_info=True)
        logger_se.warning(f"トークン化エラー（詳細はログ確認）: {e}。ソース: {source_info}")
        return []

def save_bm25_data(bm25_model, document_chunks, bm25_index_path, tokenized_corpus=None):
    data_to_save = { 'model': bm25_model, 'chunks': document_chunks, 'tokens': tokenized_corpus }
    try:
        with open(bm25_index_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        logger_se.info(f"BM25インデックス（+トークンデータ）を '{bm25_index_path}' に保存しました。")
    except Exception as e:
        logger_se.error(f"BM25インデックスの保存中にエラーが発生しました: {e}")

def load_bm25_data(bm25_index_path):
    if not os.path.exists(bm25_index_path):
        return None, None, None
    try:
        with open(bm25_index_path, 'rb') as f:
            loaded_data = pickle.load(f)
        logger_se.info(f"既存のBM25インデックスを '{bm25_index_path}' から読み込みました。")
        return loaded_data.get('model'), loaded_data.get('chunks'), loaded_data.get('tokens')
    except (pickle.UnpicklingError, EOFError, KeyError) as e:
        logger_se.warning(f"BM25インデックスファイルの読み込みに失敗（破損の可能性）。再構築します。エラー: {e}")
        return None, None, None
    except Exception as e:
        logger_se.error(f"BM25インデックスの読み込み中に予期せぬエラー: {e}")
        return None, None, None

def initialize_bm25_engine(documents, bm25_index_path, chunk_size, chunk_overlap, force_rebuild=False, incremental_base_data=None):
    existing_model, existing_chunks, existing_tokens = None, [], []
    need_retokenize_existing = False
    
    if not force_rebuild and not incremental_base_data:
        existing_model, existing_chunks, existing_tokens = load_bm25_data(bm25_index_path)
        if existing_model and existing_chunks:
            return existing_model, existing_chunks

    if incremental_base_data:
        existing_model, existing_chunks, existing_tokens = incremental_base_data
        if existing_tokens is None:
            logger_se.info("既存データ形式が古いため（トークンデータ欠損）、既存チャンクの再トークン化を行います。")
            existing_tokens = []
            need_retokenize_existing = True
        else:
            logger_se.info(f"既存の {len(existing_chunks)} チャンクのトークンデータを再利用します。")

    logger_se.info("BM25インデックスを構築/更新しています...")

    new_chunks = []
    if documents:
        new_chunks = get_document_chunks(documents, chunk_size, chunk_overlap)
        
    if not new_chunks and not existing_chunks:
        logger_se.warning("BM25インデックス構築のための有効なチャンクがありません。")
        return None, []

    final_chunks = (existing_chunks or []) + new_chunks

    try:
        tokenizer_instance = get_sudachi_tokenizer_instance()
        
        if need_retokenize_existing and existing_chunks:
            logger_se.info(f"既存 {len(existing_chunks)} チャンクの再トークン化を実行中...")
            existing_tokens = [tokenize_text_sudachi_util(doc.page_content, tokenizer_instance) for doc in existing_chunks]

        new_tokens = []
        if new_chunks:
            logger_se.info(f"新規 {len(new_chunks)} チャンクのトークン化を実行中...")
            new_tokens = [tokenize_text_sudachi_util(doc.page_content, tokenizer_instance) for doc in new_chunks]
            
        final_tokenized_corpus = (existing_tokens or []) + new_tokens
        
        if len(final_chunks) != len(final_tokenized_corpus):
             logger_se.error("チャンク数とトークンリスト数が一致しません。インデックス構築を中止します。")
             return None, []

        bm25_model = BM25Okapi(final_tokenized_corpus)
        save_bm25_data(bm25_model, final_chunks, bm25_index_path, final_tokenized_corpus)
        
        logger_se.info(f"BM25インデックスの構築完了（総チャンク数: {len(final_chunks)}）。")
        return bm25_model, final_chunks
    except Exception as e:
        logger_se.error(f"BM25モデルの構築中にエラーが発生しました: {e}")
        return None, []

def search_bm25(query, bm25_model, indexed_chunks, tokenizer_instance_for_search, top_n):
    if not query or not bm25_model or not indexed_chunks or not tokenizer_instance_for_search: return []
    logger_se.info(f"BM25検索実行。クエリ: '{query[:50]}...'")
    target_pos_for_bm25_query = ["名詞", "動詞", "形容詞"]
    normalized_form_pos_for_bm25_query = ["動詞", "形容詞"]
    
    query_tokens = tokenize_text_sudachi_util(
        query, tokenizer_instance_for_search, source_info="検索クエリ(BM25)",
        pos_filter_active=True, target_pos_major=target_pos_for_bm25_query,
        use_normalized_form_for_bm25_pos=normalized_form_pos_for_bm25_query
    )
    
    if not query_tokens:
        logger_se.info("BM25検索: クエリトークン化失敗（有効トークンなし）。")
        return []
    
    try:
        chunk_scores = bm25_model.get_scores(query_tokens)
    except Exception as e:
        logger_se.error(f"BM25スコア計算エラー: {e}", exc_info=True)
        return []

    results_with_scores_and_excerpts = []
    sorted_indices_and_scores = sorted(
        enumerate(chunk_scores),
        key=lambda x: x[1], reverse=True
    )
    
    for i, score in sorted_indices_and_scores:
        if score <= 0: continue
        original_chunk_doc = indexed_chunks[i]
               
        results_with_scores_and_excerpts.append((original_chunk_doc, score))
        
        if top_n is not None and len(results_with_scores_and_excerpts) >= top_n: break
        
    logger_se.info(f"BM25検索完了。結果数: {len(results_with_scores_and_excerpts)}")
    return results_with_scores_and_excerpts

def search_and_完全一致(query_text, target_chunks, case_sensitive=False, context_window_chars=150, top_n=100):
    if not query_text or not target_chunks: return []
    logger_se.info(f"AND検索実行。クエリ: '{query_text[:50]}...'")
    query_keywords_raw = [kw.strip() for kw in query_text.split() if kw.strip()]
    if not query_keywords_raw: return []

    matching_excerpts = []
    for chunk in target_chunks:
        chunk_text_to_search = chunk.page_content
        all_keywords_found_in_chunk = True
        first_hit_indices = {}

        for keyword_raw in query_keywords_raw:
            original_keyword_pos_in_chunk = -1
            try:
                # キーワードの各文字間に \s* を挿入して、スペースや改行を無視して検索
                keyword_pattern = r'\s*'.join(map(re.escape, keyword_raw))

                if not case_sensitive:
                    match = re.search(keyword_pattern, chunk.page_content, re.IGNORECASE)
                    if match:
                        original_keyword_pos_in_chunk = match.start()
                else:
                    match = re.search(keyword_pattern, chunk.page_content)
                    if match: original_keyword_pos_in_chunk = match.start()
            except re.error:
                all_keywords_found_in_chunk = False; break
            except Exception:
                all_keywords_found_in_chunk = False; break

            if original_keyword_pos_in_chunk == -1:
                all_keywords_found_in_chunk = False; break

            if keyword_raw not in first_hit_indices:
                first_hit_indices[keyword_raw] = original_keyword_pos_in_chunk

        if not all_keywords_found_in_chunk:
            continue

        if first_hit_indices:
            earliest_keyword_start_index = min(first_hit_indices.values())
            latest_keyword_end_index = 0
            for kw, start_idx in first_hit_indices.items():
                latest_keyword_end_index = max(latest_keyword_end_index, start_idx + len(kw))

            start_extract = max(0, earliest_keyword_start_index - context_window_chars)
            end_extract = min(len(chunk.page_content), latest_keyword_end_index + context_window_chars)

            excerpt_text = chunk.page_content[start_extract:end_extract]
            prefix = "... " if start_extract > 0 else ""
            suffix = " ..." if end_extract < len(chunk.page_content) else ""
            excerpt_text = prefix + excerpt_text + suffix

            excerpt_doc = Document(page_content=excerpt_text, metadata=chunk.metadata.copy())
            matching_excerpts.append(excerpt_doc)
            
            # 件数上限の判定用にもう1件だけ余分に取得してからbreakする
            if top_n is not None and len(matching_excerpts) > top_n:
                break
    logger_se.info(f"AND検索完了。結果数: {len(matching_excerpts)}")
    return matching_excerpts

@st.cache_resource
def get_embedding_model(model_name: str):
    try:
        logger_se.info(f"Ollama 埋め込みモデル ({model_name}) のロードを開始します。")
        model = OllamaEmbeddings(model=model_name)
        logger_se.info(f"Ollama 埋め込みモデル ({model_name}) のロード成功。")
        return model
    except Exception as e:
        logger_se.error(f"Ollama 埋め込みモデル ({model_name}) ロード失敗: {e}", exc_info=True)
        return None

def get_document_chunks(_all_documents_list, chunk_size, chunk_overlap):
    if not _all_documents_list:
        logger_se.info("チャンク化スキップ: 入力ドキュメントリストが空。")
        return []

    # テキストの正規化 (NFKC)
    # 検索精度向上のため、全角英数→半角、半角カナ→全角などに統一します。
    for doc in _all_documents_list:
        if doc.page_content:
            doc.page_content = unicodedata.normalize('NFKC', doc.page_content)

    logger_se.info(f"チャンク化開始。対象ドキュメント数: {len(_all_documents_list)}, チャンクサイズ: {chunk_size}, オーバーラップ: {chunk_overlap}")
    
    # --- 変更: 区切り文字の順序を最適化 ---
    # 段落(\n\n) > 文(。) > 読点(、) > 改行(\n) の優先順位に変更
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "。", "、", "\n", " ", ""]
    )
    
    try:
        chunks = text_splitter.split_documents(_all_documents_list)
        logger_se.info(f"チャンク化成功。生成チャンク数: {len(chunks)}")
        return chunks
    except Exception as e:
        logger_se.error(f"チャンク化エラー: {e}", exc_info=True)
        return []

def initialize_lancedb(document_chunks, embedding_function, persist_directory, collection_name, batch_size=32, force_rebuild=False):
    if not embedding_function:
        logger_se.error("ベクトルストア構築/ロードスキップ: 埋め込み関数が提供されていません。")
        return None

    logger_se.info(f"LanceDB処理開始。テーブル: '{collection_name}', 永続化先: '{persist_directory}', ドキュメント提供: {'あり' if document_chunks else 'なし'}")

    try:
        db = lancedb.connect(persist_directory)
    except Exception as e:
        logger_se.error(f"LanceDBへの接続に失敗 ({persist_directory}): {e}", exc_info=True)
        return None

    if force_rebuild and collection_name in db.table_names():
        try:
            logger_se.info(f"既存テーブル '{collection_name}' の削除を試みます...")
            db.drop_table(collection_name)
            logger_se.info(f"既存テーブル '{collection_name}' を削除しました。")
        except Exception as e_drop:
            logger_se.error(f"既存テーブル '{collection_name}' の削除に失敗: {e_drop}", exc_info=True)
            return None

    if document_chunks:
        logger_se.info(f"ドキュメントからLanceDBテーブルを構築/更新します。提供チャンク数: {len(document_chunks)}")
        table = None
        if collection_name in db.table_names():
            table = db.open_table(collection_name)
        else:
            try:
                # スキーマを定義してテーブルを作成
                # ベクトルの次元数をembedding_functionから取得
                sample_embedding = embedding_function.embed_query("sample")
                embedding_dim = len(sample_embedding)

                schema = pa.schema([
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
                    pa.field("source", pa.string()),
                    pa.field("page", pa.int64())
                ])
                table = db.create_table(collection_name, schema=schema)
                logger_se.info(f"新規テーブル '{collection_name}' を作成しました。")
            except Exception as e_create:
                logger_se.error(f"テーブル '{collection_name}' の作成に失敗: {e_create}", exc_info=True)
                return None

        total_chunks = len(document_chunks)
        for i in range(0, total_chunks, batch_size):
            batch_chunks = document_chunks[i:i + batch_size]
            batch_texts = [chunk.page_content for chunk in batch_chunks]

            logger_se.info(f"チャンク {i+1}-{min(i+batch_size, total_chunks)} / {total_chunks} のベクトル化を開始...")
            try:
                embeddings = embedding_function.embed_documents(batch_texts)
            except Exception as e_embed:
                logger_se.error(f"バッチ {i//batch_size + 1} のベクトル化中にエラー: {e_embed}", exc_info=True)
                continue

            data = []
            for chunk, vector in zip(batch_chunks, embeddings):
                data.append({
                    "text": chunk.page_content,
                    "vector": vector,
                    "source": chunk.metadata.get("source", "N/A"),
                    "page": chunk.metadata.get("page", -1)
                })

            try:
                table.add(pd.DataFrame(data))
            except Exception as e_add:
                logger_se.error(f"DBへのバッチ {i//batch_size + 1} の追加中にエラー: {e_add}", exc_info=True)

        logger_se.info(f"全チャンクのベクトル化とDBへの追加が完了しました。")
        logger_se.info(f"テーブル '{collection_name}' の作成/更新が完了しました。総ドキュメント数: {table.count_rows()}")
        return table
    else:
        logger_se.info(f"既存のLanceDBテーブル ({persist_directory}、テーブル: '{collection_name}') をロードします...")
        try:
            if collection_name in db.table_names():
                table = db.open_table(collection_name)
                count = table.count_rows()
                if count == 0:
                    logger_se.warning(f"LanceDBテーブル '{collection_name}' はロードできましたが空です。")
                else:
                    logger_se.info(f"既存のLanceDBテーブルのロードが完了しました。({count}件のドキュメント)")
                return table
            else:
                 logger_se.warning(f"既存のテーブル '{collection_name}' が見つかりませんでした。")
                 return None
        except Exception as e_load:
            logger_se.error(f"既存のLanceDBテーブルのロード処理中に予期せぬエラー: {e_load}", exc_info=True)
            return None

def search_vector(query, table, embedding_function, top_n, filter_condition=None):
    if not query or not table or not embedding_function: return []
    logger_se.info(f"ベクトル検索実行。クエリ: '{query[:50]}...'")
    try:
        query_vector = embedding_function.embed_query(query)
        search_query = table.search(query_vector)
        
        if filter_condition:
            search_query = search_query.where(filter_condition, prefilter=True)
            
        results_df = search_query.limit(top_n).to_df()

        langchain_results = []
        for _, row in results_df.iterrows():
            metadata = {"source": row["source"], "page": row["page"]}
            doc = Document(page_content=row["text"], metadata=metadata)
            # LanceDBの距離はL2距離なので、スコアとしてそのまま利用
            score = row["_distance"]
            langchain_results.append((doc, score))

        logger_se.info(f"ベクトル検索完了。結果数: {len(langchain_results)}")
        return langchain_results
    except Exception as e:
        logger_se.error(f"ベクトル検索エラー: {e}", exc_info=True)
        return []

_similarity_model_instance = None
@st.cache_resource
def get_similarity_model_instance(model_name=SIMILARITY_MODEL_NAME):
    global _similarity_model_instance
    if _similarity_model_instance is None:
        try:
            logger_se.info(f"類似度計算モデル ({model_name}) のロードを開始します。")
            _similarity_model_instance = SentenceTransformer(model_name)
            logger_se.info(f"類似度計算モデル ({model_name}) のロード成功。")
        except Exception as e:
            logger_se.error(f"類似度計算モデル ({model_name}) のロードに失敗しました: {e}", exc_info=True)
            return None
    return _similarity_model_instance

def calculate_similarity_matrix(texts, model_instance):
    if not texts or not model_instance: return np.array([])
    try:
        embeddings = model_instance.encode(texts, convert_to_tensor=True)
        similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
        return similarity_matrix.cpu().numpy()
    except Exception as e:
        logger_se.warning(f"類似度行列の計算中にエラー: {e}", exc_info=True)
        return np.array([])

def deduplicate_chunks(chunk_docs, similarity_threshold=0.90, model_instance=None):
    if not chunk_docs: return []
    is_tuple_list = all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], Document) for item in chunk_docs)
    actual_docs = [item[0] for item in chunk_docs] if is_tuple_list else chunk_docs
    if not actual_docs: return []
    texts_to_compare = [doc.page_content for doc in actual_docs]
    if len(texts_to_compare) <= 1: return chunk_docs

    # 引数で渡されなかった場合はキャッシュから取得（後方互換性用）
    sim_model = model_instance if model_instance else get_similarity_model_instance()
    if not sim_model:
        logger_se.warning("類似度計算モデル利用不可のため、重複排除をスキップします。")
        return chunk_docs

    similarity_matrix = calculate_similarity_matrix(texts_to_compare, sim_model)
    if similarity_matrix.size == 0: return chunk_docs

    indices_to_keep = []
    indices_to_discard = set()
    for i in range(len(texts_to_compare)):
        if i in indices_to_discard: continue
        indices_to_keep.append(i)
        for j in range(i + 1, len(texts_to_compare)):
            if j in indices_to_discard: continue
            if similarity_matrix[i, j] >= similarity_threshold:
                indices_to_discard.add(j)

    deduplicated_chunk_docs = []
    if is_tuple_list:
        original_scores = [item[1] for item in chunk_docs]
        for i in indices_to_keep: deduplicated_chunk_docs.append((actual_docs[i], original_scores[i]))
    else:
        for i in indices_to_keep: deduplicated_chunk_docs.append(actual_docs[i])

    if len(texts_to_compare) != len(deduplicated_chunk_docs):
        logger_se.info(f"重複排除: {len(texts_to_compare)}チャンクから{len(deduplicated_chunk_docs)}チャンクに削減 (閾値: {similarity_threshold})")

    return deduplicated_chunk_docs

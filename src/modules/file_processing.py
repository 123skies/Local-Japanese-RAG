# project_root/src/modules/file_processing.py
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
import logging
import pandas as pd

# ロガーを取得
logger = logging.getLogger('scholarscope_lite')

def load_document_from_path(file_path_on_server):
    _, ext = os.path.splitext(file_path_on_server)
    ext = ext.lower()
    file_basename = os.path.basename(file_path_on_server)
    docs = []
    logger.debug(f"ファイルロード開始: {file_basename} (タイプ: {ext})")

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path_on_server)
            docs = loader.load()
            logger.debug(f"PDFファイル '{file_basename}' をロードしました。ページ数: {len(docs)}")
        elif ext == ".txt":
            try:
                loader = TextLoader(file_path_on_server, encoding='utf-8')
                docs = loader.load()
                logger.debug(f"TXTファイル '{file_basename}' をUTF-8でロードしました。")
            except UnicodeDecodeError:
                logger.warning(f"ファイル '{file_basename}' はUTF-8でデコードできませんでした。Shift-JISで試行します。")
                try:
                    loader = TextLoader(file_path_on_server, encoding='shift_jis')
                    docs = loader.load()
                    logger.debug(f"TXTファイル '{file_basename}' をShift-JISでロードしました。")
                except Exception as e_enc_sjis:
                    logger.error(f"ファイル '{file_basename}' の読み込みに失敗しました（Shift-JIS試行後）: {e_enc_sjis}", exc_info=True)
                    st.error(f"ファイル '{file_basename}' の読み込みに失敗しました（複数エンコーディング試行後）")
                    return [Document(page_content=f"Error: Could not decode file {file_basename}.", metadata={"source": file_basename, "error": str(e_enc_sjis)})]
            except Exception as e_txt:
                 logger.error(f"TXTファイル '{file_basename}' の読み込みエラー: {e_txt}", exc_info=True)
                 st.error(f"TXTファイル '{file_basename}' の読み込みエラー")
                 return [Document(page_content=f"Error loading TXT file {file_basename}: {e_txt}", metadata={"source": file_basename, "error": str(e_txt)})]
        elif ext == ".md":
            try:
                loader = TextLoader(file_path_on_server, encoding='utf-8')
                docs = loader.load()
                logger.debug(f"Markdownファイル '{file_basename}' をUTF-8でロードしました。")
            except Exception as e_md:
                 logger.error(f"Markdownファイル '{file_basename}' の読み込みエラー: {e_md}", exc_info=True)
                 st.error(f"Markdownファイル '{file_basename}' の読み込みエラー")
                 return [Document(page_content=f"Error loading MD file {file_basename}: {e_md}", metadata={"source": file_basename, "error": str(e_md)})]
        elif ext == ".csv":
            docs = load_csv_as_documents(file_path_on_server)
        else:
            logger.warning(f"サポートされていないファイル形式です: {ext} ({file_basename})")
            return [Document(page_content=f"Error: Unsupported file type {file_basename}.", metadata={"source": file_basename, "error": "Unsupported file type"})]

        # メタデータ処理
        for i, doc_obj in enumerate(docs):
            if "source" not in doc_obj.metadata:
                doc_obj.metadata["source"] = file_basename
            if 'page' not in doc_obj.metadata and len(docs) > 1:
                doc_obj.metadata["page"] = i + 1
            elif 'page' in doc_obj.metadata:
                try:
                    doc_obj.metadata["page"] = int(doc_obj.metadata["page"]) + 1
                except (ValueError, TypeError):
                    logger.warning(f"ファイル '{file_basename}' のページ番号メタデータが不正: {doc_obj.metadata.get('page')}. 連番でフォールバックします。")
                    doc_obj.metadata["page"] = i + 1
        logger.debug(f"ファイル '{file_basename}' のメタデータ処理完了。")
        return docs

    except Exception as e_load_main:
        logger.error(f"ファイル '{file_basename}' のロード中に予期せぬエラー: {e_load_main}", exc_info=True)
        st.error(f"ファイル '{file_basename}' のロード中に予期せぬエラー")
        return [Document(page_content=f"Error loading file {file_basename}: {e_load_main}", metadata={"source": file_basename, "error": str(e_load_main)})]


def get_document_content_as_string(langchain_documents):
    if not langchain_documents:
        return ""
    content_list = [doc.page_content for doc in langchain_documents]
    return "\n\n---\n\n".join(content_list)


def load_csv_as_documents(file_path_on_server):
    """CSVファイルを読み込み、各行をLangChainのDocumentオブジェクトのリストに変換する"""
    file_basename = os.path.basename(file_path_on_server)
    docs = []
    logger.info(f"CSVファイルロード開始: {file_basename}")

    try:
        df = pd.read_csv(file_path_on_server, encoding='utf-8')

        for index, row in df.iterrows():
            # テキスト情報を結合して page_content を作成
            # ここでは全ての列を文字列に変換して結合する汎用的な方法を採用
            page_content = " ".join([f"{col}: {val}" for col, val in row.astype(str).items()])

            metadata = row.to_dict()
            metadata["source"] = file_basename
            metadata["row_number"] = index + 1
            metadata["page"] = -1 # ページ情報がないことを示す

            if page_content.strip():
                docs.append(Document(page_content=page_content, metadata=metadata))

        logger.info(f"CSVファイル '{file_basename}' をロード完了。{len(docs)} 件のドキュメントを生成。")
        return docs

    except Exception as e:
        logger.error(f"CSVファイル '{file_basename}' のロード中にエラー: {e}", exc_info=True)
        return []
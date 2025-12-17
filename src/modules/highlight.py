# modules/highlight.py
import streamlit as st
import re
from modules.search_engines import tokenize_text_sudachi_util # tokenize_text_sudachi_util を直接インポート
import logging # ### 追加 ###

logger = logging.getLogger('scholarscope_lite')

def highlight_text(text_to_search_in, keywords_to_highlight, default_color="#B2EBF2", case_sensitive_highlight=False, search_type="bm25"):
    if not text_to_search_in or not keywords_to_highlight:
        return text_to_search_in

    highlighted_text = text_to_search_in
    valid_keywords = [kw for kw in keywords_to_highlight if kw and kw.strip()]
    if not valid_keywords:
        return text_to_search_in

    unique_sorted_keywords = sorted(list(set(valid_keywords)), key=len, reverse=True)

    for keyword_original_case in unique_sorted_keywords:
        if not keyword_original_case.strip(): continue
        try:
            regex_flags = 0 if case_sensitive_highlight else re.IGNORECASE

            pattern = ""
            if search_type == "and":
                # and検索の場合は、文字間に \s* を挿入してスペースや改行を許容
                pattern = r'\s*'.join(map(re.escape, keyword_original_case))
            else:
                # それ以外は従来通り
                pattern = re.escape(keyword_original_case)

            highlighted_text = re.sub(
                f"({pattern})",
                r"<mark style='background-color: {};'>\1</mark>".format(default_color),
                highlighted_text,
                flags=regex_flags
            )
        except re.error as e:
            # このエラーは特定のキーワードで発生しうるので、ログには残しつつUI警告は頻度を考慮
            logger.warning(f"ハイライト処理中に正規表現エラー (キーワード: '{keyword_original_case}'): {e}")
            # st.warning(f"ハイライト処理中に正規表現エラー (キーワード: '{keyword_original_case}'): {e}") # UI表示はコメントアウト
            pass
        except Exception as ex:
            logger.warning(f"ハイライト処理中に予期せぬエラー (キーワード: '{keyword_original_case}'): {ex}", exc_info=True)
            pass
    return highlighted_text


def get_query_terms_for_highlight(query_text, search_type="bm25", tokenizer_instance=None, source_info="ハイライト用クエリ"):
    if not query_text:
        return []

    query_terms = []
    target_pos_for_bm25 = ["名詞", "動詞", "形容詞"]
    normalized_form_pos_for_bm25 = ["動詞", "形容詞"]

    if search_type == "and":
        raw_keywords = [kw.strip() for kw in query_text.split() if kw.strip()]
        query_terms.extend(raw_keywords)
        if tokenizer_instance:
            from sudachipy import Tokenizer as SudachiTokenizer # これはsearch_enginesからインポート済みなので不要かも
            for kw_raw in raw_keywords:
                try:
                    tokens_from_kw = tokenize_text_sudachi_util(
                        kw_raw, tokenizer_instance,
                        source_info=f"AND検索ハイライト用キーワード '{kw_raw}' ({source_info})",
                        pos_filter_active=False
                    )
                    query_terms.extend([token for token in tokens_from_kw if len(token.strip()) > 0])
                except Exception as e:
                    logger.warning(f"AND検索ハイライト用キーワード '{kw_raw}' のトークン化エラー ({source_info}): {e}", exc_info=True)

    elif search_type == "bm25":
        if tokenizer_instance:
            try:
                query_terms = tokenize_text_sudachi_util(
                    query_text, tokenizer_instance,
                    source_info=f"BM25ハイライト用クエリ ({source_info})",
                    pos_filter_active=True,
                    target_pos_major=target_pos_for_bm25,
                    use_normalized_form_for_bm25_pos=normalized_form_pos_for_bm25
                )
            except Exception as e:
                logger.warning(f"BM25ハイライト用クエリのトークン化エラー ({source_info}): {e}", exc_info=True)
                query_terms = [kw.strip() for kw in query_text.split() if kw.strip()] # フォールバック
        else:
            query_terms = [kw.strip() for kw in query_text.split() if kw.strip()]

    elif search_type == "vector": # Vector検索のハイライトは単純分割
        query_terms = re.split(r'\s+|,|、|。|[\uff0c\uff0e]', query_text)
        query_terms = [term.strip() for term in query_terms if term.strip()]
    else: # 不明なタイプは単純分割
        query_terms = [kw.strip() for kw in query_text.split() if kw.strip()]

    return list(set(query_terms))
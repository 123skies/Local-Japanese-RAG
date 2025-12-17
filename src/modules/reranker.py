# modules/reranker.py
import torch
from sentence_transformers import CrossEncoder
from langchain.schema import Document
import logging

logger = logging.getLogger('scholarscope_lite')

reranker_model = None

def get_reranker_model(model_name='BAAI/bge-reranker-v2-m3'):
    global reranker_model
    if reranker_model is None:
        try:
            logger.info(f"リランキングモデル '{model_name}' のロードを開始...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # fp16有効化により高速化とVRAM節約 (RTX 3060向け)
            automodel_args = {"torch_dtype": torch.float16} if device == 'cuda' else {}
            reranker_model = CrossEncoder(
                model_name, 
                max_length=512, 
                device=device,
                automodel_args=automodel_args
            )
            logger.info(f"リランキングモデルを正常にロードしました (デバイス: {device})。")
        except Exception as e:
            logger.error(f"リランキングモデルのロード中にエラーが発生しました: {e}", exc_info=True)
            reranker_model = None # ロード失敗を明示
    return reranker_model

def rerank_documents(query: str, documents: list[Document], reranker, top_n=50, progress_callback=None, batch_size=1) -> list[tuple[Document, float]]:
    if not documents or not query or not reranker:
        return []
    
    logger.info(f"リランキング処理開始。対象ドキュメント数: {len(documents)}, クエリ: '{query[:50]}...'")
    
    # --- Smart Batching (長さ順ソート) の導入 ---
    # 1. (元のインデックス, (クエリ, 本文)) のリストを作成
    indexed_pairs = []
    for idx, doc in enumerate(documents):
        indexed_pairs.append((idx, (query, doc.page_content)))
    
    # 2. 本文の長さ順にソート (長い順)
    # これにより、バッチ内のパディング量を最小化し、計算効率を最大化する
    indexed_pairs.sort(key=lambda x: len(x[1][1]), reverse=True)
    
    sorted_pairs = [pair for _, pair in indexed_pairs]
    sorted_indices = [idx for idx, _ in indexed_pairs]
    
    scores_in_sorted_order = []
    total = len(documents)
    
    # 3. ソート順でバッチ推論
    for i in range(0, total, batch_size):
        batch = sorted_pairs[i:i + batch_size]
        
        # show_progress_bar=False にして、自前のコールバックを使用
        batch_scores = reranker.predict(
            batch, 
            convert_to_numpy=True, 
            show_progress_bar=False, 
            batch_size=batch_size
        )
        
        # 結果がスカラーの場合とリストの場合を考慮
        if isinstance(batch_scores, float):
            scores_in_sorted_order.append(batch_scores)
        else:
            scores_in_sorted_order.extend(batch_scores.tolist())
        
        if progress_callback:
            progress_callback(min(i + batch_size, total), total)
            
    # 4. スコアを元のドキュメント順序に戻す
    final_scores = [0.0] * total
    for sort_idx, original_idx in enumerate(sorted_indices):
        final_scores[original_idx] = scores_in_sorted_order[sort_idx]

    ranked_results = sorted(zip(documents, final_scores), key=lambda x: x[1], reverse=True)
    logger.info(f"リランキング処理完了。上位{min(top_n, len(ranked_results))}件を返します。")
    
    return ranked_results[:top_n]
# modules/utils.py
import logging

logger = logging.getLogger('scholarscope_lite')

# tiktokenのインポートを試みる（既存コード維持）
try:
    import tiktoken
    tiktoken_encoder = tiktoken.get_encoding("cl100k_base") # gpt-4, gpt-3.5-turbo用
except Exception as e:
    logger.warning(f"tiktokenエンコーダーの初期化に失敗: {e}。トークンカウントは文字数ベースの概算になります。", exc_info=True)
    tiktoken_encoder = None

def count_tokens(text):
    """
    テキストのトークン数をカウントする。
    tiktokenが使えない場合は、日本語環境での安全な概算値（文字数そのもの）を返す。
    """
    if not text:
        return 0

    if tiktoken_encoder:
        try:
            return len(tiktoken_encoder.encode(text))
        except Exception as e:
            logger.warning(f"tiktokenでのカウント中にエラー: {e}。概算値を使用します。")
    
    # --- 修正箇所 ---
    # 旧: return len(text) // 3
    # 新: 日本語を考慮し、文字数とほぼ等倍（係数1.1倍程度）で見積もる
    # ※ Qwenなどのマルチバイト文字に強いモデルでも、安全係数として 1.1〜1.2 を掛けるのが推奨
    return int(len(text) * 1.1)
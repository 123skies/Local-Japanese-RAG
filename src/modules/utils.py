# modules/utils.py
import tiktoken
import logging

logger = logging.getLogger('scholarscope_lite')

try:
    tiktoken_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
except Exception as e:
    logger.warning(f"tiktokenエンコーダーの初期化に失敗: {e}。トークンカウントは文字数ベースの概算になります。", exc_info=True)
    tiktoken_encoder = None

def count_tokens(text):
    if tiktoken_encoder:
        return len(tiktoken_encoder.encode(text))
    else:
        return len(text) // 3 # フォールバック
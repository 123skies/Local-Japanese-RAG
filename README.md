# Local-Japanese-RAG

日本語文書特化の高精度ローカルRAG。入力・根拠・回答を並列表示する調査特化型UI、ハイブリッド検索、リランキング、出典明示機能を備えた分析ツールです。

Local RAG optimized for Japanese documents with a 3-pane interface dedicated to deep research and analysis.

## ✨ 特徴 (Key Features)

- **3パネル構成の調査特化UI / 3-Pane Research Interface**
  - Left: 検索・設定 (Input & Settings)
  - Center: 根拠資料のハイライト表示 (Evidence Viewer)
  - Right: AIによる分析・回答生成 (Analysis & Output)
  チャット形式ではなく、資料を並べて比較・精査する「デスクトップ調査」に最適化されています。

- **日本語特化 / Optimized for Japanese**
  - 和暦の正規化に対応
  - SudachiPyによる高精度な形態素解析

- **高精度検索 / High Precision**
  - Hybrid Search (BM25 + Vector)
  - Cross-Encoder Reranking
  - Strict Citation Mapping (ハルシネーション低減のための出典紐付け)

## 🚀 構成技術 (Stack)
- Streamlit
- Ollama (LLM/Embeddings)
- LanceDB (Vector Store)
- SudachiPy
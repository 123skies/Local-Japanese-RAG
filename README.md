# ScholarScope (Local-Japanese-RAG)

**Japanese-First, High-Precision, Local RAG System.**
完全ローカル環境で動作する、日本語文書に特化した高精度な文献調査・分析システムです。

![UI Preview](https://placehold.co/800x400?text=App+Screenshot+Here)
*(ここにアプリのスクリーンショットを貼るとベストです)*

## 💡 コンセプト (Concept)

本プロジェクトは、単なる「チャットボット」ではなく、研究者や実務家が**「根拠を確認しながら分析できる調査ツール」**を目指して開発しました。

- **完全ローカル志向**: 機密性の高い文献を扱うため、外部APIへのデータ送信を行いません。
- **ミドルレンジGPUへの最適化**: RTX 3060 (VRAM 12GB) 程度の一般的なゲーミングPC環境で、実用的な速度と精度が出るように設計しています。
- **軽量モデルでの高精度化**: 巨大なLLMを使わずに、プロンプトエンジニアリングと検索ロジックの工夫（ハイブリッド検索＋リランキング）によって回答精度を高めています。

## ✨ 主な機能 (Key Features)

### 1. 🇯🇵 日本語への徹底的な最適化
英語圏のRAGツールでは対応しきれない、日本語特有の処理を実装しています。
- **和暦・西暦の正規化**: `date_standardizer.py`
  - 「慶長」から「令和」までの元号に対応。文書内の「昭和○年」などの表記を正規化し、時系列検索の精度を向上させています。
- **表記揺れの吸収 (簡体字/繁体字対策)**: `kanji_converter.py`
  - 軽量LLMが生成しがちな簡体字・繁体字（「学」「學」など）を、日本語の標準的な漢字（新字体）に強制変換するマッピング機能を搭載。
- **形態素解析**: `SudachiPy` を採用し、日本語の文脈を考慮したトークン化を行っています。

### 2. 🔍 調査特化型 3パネルUI
チャット形式ではなく、調査業務に特化した3カラムレイアウトを採用しています。
- **左 (Input)**: 詳細な検索設定（キーワード、意味検索、絞り込み）。
- **中 (Evidence)**: 検索ヒット箇所をハイライト表示。AIがどの部分を参照したか即座に確認できます。
- **右 (Analysis)**: 検索結果に基づいた回答生成。ハルシネーション（嘘）を防ぐため、**「出典番号 [1]」** を厳密に付与するロジックを搭載しています。

### 3. 🛡️ 検索ロジック (Search Architecture)
取りこぼしを防ぐための「守りの検索」ロジックを実装しています。
1. **Hybrid Search**: BM25（キーワード）とVector（意味）の並列検索。
2. **Backfill Logic**: ベクトル検索でスコアの急落や距離の乖離を監視し、関連する可能性のある文書を慎重に拾い上げる独自ロジック。
3. **Reranking**: `Cross-Encoder` を使用し、検索結果を文脈に沿って再順位付け。

## 🛠️ 使用モデル・環境 (Models & Specs)

本システムは、以下の軽量モデルおよび環境で動作確認・チューニングしています。

| Category | Model Name (Example) | Note |
| --- | --- | --- |
| **LLM** | `Qwen3-Instruct` variants (e.g. 4B/7B Quantized) | VRAM容量に合わせて選択。プロンプトはこれらに最適化済み。 |
| **Embedding** | `Qwen3-Embedding-4B` etc. | 日本語対応の高性能埋め込みモデル。 |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | 高精度な再順位付けモデル。 |

- **推奨ハードウェア**: NVIDIA GPU (VRAM 8GB〜12GB推奨)
  - ※開発環境: RTX 3060 (12GB)

## ⚠️ 制限事項・免責 (Limitations)

- **対応フォーマット**: 実装上は `.txt`, `.md`, `.pdf`, `.csv` の読み込みに対応しています（Shift-JIS/UTF-8自動判定あり）。
- **検証と制限**: ただし、開発環境での精度チューニングは主に**手動で整形済みのテキストファイル**を用いて行っています。PDFの高度なレイアウト解析（特に縦書き対応）やOCR機能は搭載しておらず、複雑な形式のファイルについては動作検証が十分ではありません。
- **開発経緯**: 本プロジェクトは、Google Gemini等のAI支援を受けながらコーディングしました。コードベースにはAI由来のパターンが含まれています。

## 🚀 インストールと実行 (Installation)

```bash
# Clone repository
git clone https://github.com/123skies/Local-Japanese-RAG.git
cd Local-Japanese-RAG

# Install dependencies
pip install -r requirements.txt

# Run application
./run_app.bat
# or
streamlit run src/app.py
```

---
*Built with Streamlit, Ollama, LanceDB, and SudachiPy.*

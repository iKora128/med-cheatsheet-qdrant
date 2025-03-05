# Qdrantを用いた医療チートシートRAGシステム

このプロジェクトは、Qdrantベクトルデータベースを使用して医療チートシート（テキストファイル）のための検索・回答生成システム（RAG）を構築します。固定長チャンクによるテキスト分割、HNSWによる高精度な検索、MMRアルゴリズムによる多様な検索結果の選択、検索結果が含まれる元ページの全文をLLMに渡す手法を実装しています。

## ディレクトリ構造

```
project-root/
├── pyproject.toml            # プロジェクト設定と依存関係
├── docker-compose.yml        # Qdrantコンテナ起動用設定
├── qdrant_data/              # Qdrantの永続ストレージ用ディレクトリ
├── cache/                    # ページコンテンツのキャッシュ
├── data/                     # テキストファイルを配置
│   ├── sample.txt            # サンプルデータ（高血圧に関する情報）
│   └── … (他のテキストファイル)
└── src/
    ├── preprocess.py         # データ前処理（チャンク化と埋め込み生成）
    ├── search_retrieve.py    # 検索クエリの処理と結果取得
    └── rag_pipeline.py       # RAG全体の実行（LLMへの入力生成と応答取得）
```

## 特徴

- **固定長チャンキング**: テキストを約250文字の固定長チャンクに分割し、段落の区切りを尊重
- **高精度な検索**: Qdrantの高性能なHNSW (Hierarchical Navigable Small World) アルゴリズムを活用
- **多様な検索結果**: MMR (Maximal Marginal Relevance) アルゴリズムで類似度と多様性のバランスを最適化
- **チャンク追跡**: 各チャンクが元のどのページに属するかを追跡し、検索結果に含まれる元ページ全文をLLMに提供
- **日本語最適化**: 日本語テキスト処理に最適化された埋め込みモデルとLLMを使用

## 前提条件

- Python 3.10以上
- Docker と Docker Compose (Qdrantを実行するため)
- CUDA対応GPUを推奨 (LLM実行のため)

## セットアップ

1. リポジトリをクローン:
```bash
git clone https://github.com/yourusername/med-cheatsheet-qdrant.git
cd med-cheatsheet-qdrant
```

2. uvをインストール:
```bash
pip install uv  # uvがインストールされていない場合
```
もしくは
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. 依存パッケージをuv経由でインストール:
```bash
uv sync
```

4. Qdrantを起動:
```bash
docker compose up -d
```

## 使い方

### 1. データの準備

`data/` ディレクトリにテキストファイル（.txt）を配置します。各ファイルは1ページとして扱われ、**ファイル名（拡張子なし）がページIDとして使用されます**。検索時にはこのページIDを使って元ページ全体を取得します。

例: `data/hypertension.txt` というファイルがある場合、ページIDは `hypertension` となります。

### 2. テキストの前処理とQdrantへの登録

```bash
uv run src/preprocess.py --data_folder data --chunk_size 250
```

オプション:
- `--data_folder`: テキストファイルが格納されているディレクトリ（デフォルト: `data`）
- `--chunk_size`: チャンクの目安文字数（デフォルト: 250）
- `--embedding_model`: 使用する埋め込みモデル（デフォルト: `sbintuitions/sarashina-embedding-v1-1b`）
- `--collection_name`: Qdrantのコレクション名（デフォルト: `cheatsheet_docs`）

### 3. 検索と回答生成

```bash
uv run src/rag_pipeline.py --query "高血圧の治療薬と副作用について教えてください"
```

オプション:
- `--query`: 検索クエリ（必須）
- `--embedding_model`: 使用する埋め込みモデル（デフォルト: `sbintuitions/sarashina-embedding-v1-1b`）
- `--llm_model`: 使用するLLMモデル（デフォルト: `rinna/qwen2.5-bakeneko-32b-instruct`）
- `--no_mmr`: MMRによる再ランク付けを行わない場合は指定
- `--mmr_top_n`: MMRで選択する上位結果数（デフォルト: 5）
- `--lambda_param`: MMRのバランスパラメータ、1に近いほど関連性重視（デフォルト: 0.7）
- `--max_new_tokens`: 生成する最大トークン数（デフォルト: 1024）
- `--temperature`: 生成の温度パラメータ（デフォルト: 0.7）
- `--output`: 結果を保存するJSONファイルのパス

## 検索のみを実行する場合

検索フェーズのみを実行したい場合は、以下のコマンドを使用します：

```bash
uv run src/search_retrieve.py --query "高血圧の症状は何ですか？" --output search_results.json
```

## テキスト処理の仕組み

### チャンク化と原文参照の仕組み

1. 前処理時にテキストファイルを読み込み、ファイル名からページIDを生成
   ```python
   page_id = os.path.splitext(os.path.basename(file_path))[0]
   ```

2. 各ファイルの全テキストを`page_contents`辞書に保存（キー：ページID、値：全テキスト）
   ```python
   page_contents[page_id] = page_text
   ```

3. 各テキストを約250文字の固定長チャンクに分割し、チャンク辞書に元のページIDを紐付け
   ```python
   chunks.append({"page": page_id, "text": current_chunk.strip()})
   ```

4. チャンクとページコンテンツをそれぞれQdrantとキャッシュに保存
   ```python
   # チャンクをQdrantに保存
   client.upsert(collection_name=collection_name, points=points)
   
   # ページコンテンツをキャッシュに保存
   with open(os.path.join(cache_dir, "page_contents.pkl"), "wb") as f:
       pickle.dump(page_contents, f)
   ```

5. 検索時、チャンクに紐づいたページIDを使って元ページの全テキストをキャッシュから取得
   ```python
   pages_used = sorted({res.payload["page"] for res in selected_results})
   for page_id in pages_used:
       page_text = page_contents.get(page_id, "")
       context_pages_text.append(f"【ページ {page_id}】\n{page_text}")
   ```

### 元データファイルの取り扱い

埋め込み処理後は、検索プロセス自体では元のテキストファイル（.txt）は必要ありません。これは以下の理由によります：

1. **検索に必要なデータが既に保存されているため**：
   - Qdrantデータベース：チャンクのベクトル埋め込みとメタデータ
   - キャッシュファイル（`./cache/page_contents.pkl`）：ページID→全文のマッピング

2. **検索プロセスの流れ**：
   - ユーザークエリをベクトル化
   - Qdrantで類似チャンクを検索
   - チャンクから関連ページIDを抽出
   - `page_contents.pkl`からページ全文を取得
   - LLMに入力として提供

ただし、以下の理由から元のテキストファイルも保持しておくことを強く推奨します：

- **キャッシュ喪失の可能性**：
  サーバー再起動やキャッシュの削除により`page_contents.pkl`が失われた場合、元データから再構築できます

- **パラメータ変更の必要性**：
  チャンクサイズの調整や埋め込みモデルの変更など、再前処理が必要になった場合に元データが必要です

- **データ更新の容易さ**：
  医療情報の更新や新しいコンテンツの追加が必要な場合、元ファイルを編集して再前処理するのが簡単です

したがって、システム運用では、Qdrantデータベースとキャッシュファイルが検索処理に必須ですが、バックアップと将来の拡張性のために元のテキストファイルも保存しておくべきです。


## 依存関係の管理

本プロジェクトはuvを使用した依存関係管理を推奨しています。必要なパッケージは全て`pyproject.toml`の`dependencies`セクションに記載されています。

```toml
dependencies = [
    "qdrant-client>=1.13.2",
    "sentence-transformers>=2.2.2",
    "transformers>=4.33.1",
    "torch>=2.5.1",
    "numpy>=1.24.0",
    "sentencepiece>=0.2.0",
]
```

## 注意点

- LLMの実行には十分なGPUメモリが必要です。32Bパラメータモデルを実行するには、少なくとも24GB以上のGPUメモリを推奨します。
- Qdrantはデフォルトでローカルホストの6333ポートで動作します。
- キャッシュディレクトリ (`./cache/`) には前処理時に生成されたページコンテンツが保存されます。
- テキストファイルの名前は意味のある識別子にすることをおすすめします（例: `hypertension.txt`, `diabetes.txt`など）。

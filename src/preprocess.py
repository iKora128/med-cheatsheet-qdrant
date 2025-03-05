import os
import glob
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import argparse

def main():
    parser = argparse.ArgumentParser(description='医療チートシートのテキストをチャンク化してQdrantに登録します')
    parser.add_argument('--data_folder', type=str, default='data', help='テキストデータが格納されているフォルダパス')
    parser.add_argument('--chunk_size', type=int, default=250, help='チャンクの目安文字数')
    parser.add_argument('--embedding_model', type=str, default='sbintuitions/sarashina-embedding-v1-1b', help='使用する埋め込みモデル')
    parser.add_argument('--batch_size', type=int, default=24, help='バッチサイズ')
    parser.add_argument('--qdrant_host', type=str, default='localhost', help='Qdrantサーバーのホスト名')
    parser.add_argument('--qdrant_port', type=int, default=6333, help='QdrantサーバーのRESTポート番号')
    parser.add_argument('--collection_name', type=str, default='cheatsheet_docs', help='Qdrantのコレクション名')
    args = parser.parse_args()

    # 1. 複数TXTファイルの読み込み（data/ フォルダ内の全TXT）
    page_contents = {}  # {page_id: 全文}
    pages = []         # [(page_id, page_text), ...]
    txt_files = sorted(glob.glob(os.path.join(args.data_folder, "*.txt")))
    print(f"{len(txt_files)}個のTXTファイルを検索しました: {txt_files}")
    
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            page_text = f.read().strip()
        # ファイル名（拡張子除く）をページIDとする
        page_id = os.path.splitext(os.path.basename(file_path))[0]
        pages.append((page_id, page_text))
        page_contents[page_id] = page_text

    print(f"{args.data_folder}フォルダから{len(pages)}ページを読み込みました。")
    if len(pages) == 0:
        print(f"警告: {args.data_folder}フォルダにTXTファイルが見つかりませんでした。")
        return

    # 2. 各ページを固定長チャンクに分割
    chunks = []        # 各チャンク: {"page": page_id, "text": チャンクテキスト}
    for page_id, page_text in pages:
        # 段落単位で分割しながら、固定長チャンクを作成
        paragraphs = page_text.split("\n\n")
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < args.chunk_size:
                current_chunk += para + "\n\n"
            else:
                chunks.append({"page": page_id, "text": current_chunk.strip()})
                current_chunk = para + "\n\n"
        if current_chunk:
            chunks.append({"page": page_id, "text": current_chunk.strip()})

    print(f"{len(pages)}ページから{len(chunks)}個のチャンクを作成しました。")

    # 3. 埋め込みの生成（Sarashinaモデルの利用）
    print(f"埋め込みモデル '{args.embedding_model}' を読み込んでいます...")
    embed_model = SentenceTransformer(args.embedding_model)
    texts = [chunk["text"] for chunk in chunks]
    print(f"{len(texts)}個のチャンクテキストをベクトル化します...")
    embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=args.batch_size)
    print(f"埋め込みの形状: {embeddings.shape}")

    # 4. Qdrantへのデータ投入
    print(f"Qdrantに接続しています: {args.qdrant_host}:{args.qdrant_port}")
    client = QdrantClient(
        host=args.qdrant_host,
        port=args.qdrant_port,
        timeout=60  # タイムアウトを60秒に設定
    )
    vector_dim = embeddings.shape[1]  # Sarashinaモデルに合わせる（例: 1792）

    if not client.collection_exists(args.collection_name):
        print(f"コレクション '{args.collection_name}' を作成しています...")
        client.create_collection(
            collection_name=args.collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
    else:
        print(f"コレクション '{args.collection_name}' は既に存在します。データを追加します。")

    # バッチサイズを設定（メモリと処理時間のバランスを取る）
    batch_size = 100
    total_points = len(chunks)
    
    for start_idx in range(0, total_points, batch_size):
        end_idx = min(start_idx + batch_size, total_points)
        batch_points = []
        
        for idx in range(start_idx, end_idx):
            point = PointStruct(
                id=idx,
                vector=embeddings[idx].tolist(),
                payload={"page": chunks[idx]["page"], "text": chunks[idx]["text"]}
            )
            batch_points.append(point)
        
        print(f"バッチ {start_idx//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}: "
              f"{len(batch_points)}個のベクトルを登録中...")
        client.upsert(collection_name=args.collection_name, points=batch_points)

    print(f"処理完了: 合計{total_points}個のベクトルがQdrantに登録されました。")
    
    # ページコンテンツを別途保存（検索時の参照用）
    cache_dir = os.path.join(".", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    import pickle
    with open(os.path.join(cache_dir, "page_contents.pkl"), "wb") as f:
        pickle.dump(page_contents, f)
    print(f"ページコンテンツをキャッシュに保存しました: {os.path.join(cache_dir, 'page_contents.pkl')}")

if __name__ == "__main__":
    main() 
import os
import numpy as np
import argparse
import pickle
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import json

def load_page_contents(cache_path="./cache/page_contents.pkl"):
    """キャッシュからページコンテンツを読み込む"""
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"警告: ページコンテンツのキャッシュファイル {cache_path} が見つかりません。")
        return {}

def mmr_rerank(query_vec, results, top_n=5, lambda_param=0.7):
    """
    MMR (Maximal Marginal Relevance) アルゴリズムで検索結果を再ランク付け
    
    Args:
        query_vec: クエリベクトル
        results: Qdrantからの検索結果リスト
        top_n: 選択する上位結果数
        lambda_param: 関連性と多様性のバランスを調整するパラメータ (0-1)
                     1に近いほど関連性重視、0に近いほど多様性重視
    
    Returns:
        選択された結果のリスト
    """
    selected = []
    selected_ids = []
    
    # 各結果のベクトルを取得
    vecs = []
    for res in results:
        if hasattr(res, 'vector') and res.vector is not None:
            vecs.append(np.array(res.vector, dtype=float))
        else:
            # ベクトルが含まれていない場合は空リストを追加
            print(f"警告: ID {res.id} のベクトルがありません")
            vecs.append(np.zeros(len(query_vec), dtype=float))
    
    query_vec_np = np.array(query_vec, dtype=float)
    
    for _ in range(min(top_n, len(results))):
        best_idx = -1
        best_score = -1e9
        
        for i, vec in enumerate(vecs):
            if i in selected_ids:
                continue
            
            # クエリとの類似度計算 (コサイン類似度)
            norm_q = np.linalg.norm(query_vec_np)
            norm_vec = np.linalg.norm(vec)
            
            if norm_q == 0 or norm_vec == 0:
                sim_q = 0
            else:
                sim_q = float(np.dot(query_vec_np, vec) / (norm_q * norm_vec))
            
            # 既に選択された結果との類似度
            sim_to_selected = 0.0
            if selected_ids:
                max_sim = 0.0
                for j in selected_ids:
                    norm_j = np.linalg.norm(vecs[j])
                    if norm_vec == 0 or norm_j == 0:
                        continue
                    sim = float(np.dot(vec, vecs[j]) / (norm_vec * norm_j))
                    max_sim = max(max_sim, sim)
                sim_to_selected = max_sim
            
            # MMRスコア計算
            mmr_score = lambda_param * sim_q - (1 - lambda_param) * sim_to_selected
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected_ids.append(best_idx)
        selected.append(results[best_idx])
    
    return selected

def search_and_retrieve(query, embedding_model="sbintuitions/sarashina-embedding-v1-1b", 
                        qdrant_host="localhost", qdrant_port=6333, 
                        collection_name="cheatsheet_docs", limit=10, 
                        use_mmr=True, mmr_top_n=5, lambda_param=0.7):
    """
    クエリを受け取り、Qdrantで検索して関連するページを取得
    
    Args:
        query: 検索クエリ文字列
        embedding_model: 埋め込みモデルのパス
        qdrant_host: Qdrantホスト
        qdrant_port: Qdrantポート
        collection_name: 検索対象のコレクション名
        limit: 検索する上位結果数
        use_mmr: MMRでリランキングするかどうか
        mmr_top_n: MMRで選択する上位結果数
        lambda_param: MMRのバランスパラメータ
    
    Returns:
        検索結果とコンテキストテキスト
    """
    print(f"クエリ: {query}")
    
    # 1. クエリをベクトル化
    print(f"埋め込みモデル '{embedding_model}' を読み込み中...")
    embed_model = SentenceTransformer(embedding_model)
    query_vec = embed_model.encode(query)
    print(f"クエリをベクトル化しました。ベクトル次元: {len(query_vec)}")
    
    # 2. Qdrantで検索
    print(f"Qdrantに接続中: {qdrant_host}:{qdrant_port}")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    if not client.collection_exists(collection_name):
        raise ValueError(f"コレクション '{collection_name}' が存在しません。先に前処理を実行してください。")
    
    print(f"コレクション '{collection_name}' の上位 {limit} 件のドキュメントを検索中...")
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vec.tolist(),
        limit=limit,
        with_vectors=True  # MMRのためにベクトルも取得
    )
    
    print(f"検索結果: {len(results)} 件")
    
    # 3. 必要に応じてMMRで再ランク付け
    if use_mmr and len(results) > 1:
        print(f"MMRアルゴリズムを使用して結果の多様性を確保します (λ={lambda_param})...")
        selected_results = mmr_rerank(query_vec, results, top_n=mmr_top_n, lambda_param=lambda_param)
        print(f"MMR後の結果数: {len(selected_results)} 件")
    else:
        selected_results = results[:mmr_top_n]
    
    # 4. ページコンテンツをキャッシュから読み込み
    page_contents = load_page_contents()
    if not page_contents:
        print("警告: ページコンテンツが見つかりません。前処理を再実行してください。")
    
    # 5. 選択された結果が含まれるページテキストを取得
    pages_used = sorted({res.payload["page"] for res in selected_results})
    context_pages_text = []
    
    print(f"関連ページ: {pages_used}")
    
    for page_id in pages_used:
        page_text = page_contents.get(page_id, "")
        if page_text:
            context_pages_text.append(f"【ページ {page_id}】\n{page_text}")
        else:
            print(f"警告: ページID '{page_id}' のコンテンツが見つかりません。")
    
    context_text = "\n\n".join(context_pages_text)
    
    # 結果を表示
    print("\n検索結果の詳細:")
    for i, hit in enumerate(selected_results):
        print(f"[{i+1}] スコア: {hit.score:.4f}, ページ: {hit.payload['page']}")
        print(f"   テキスト: {hit.payload['text'][:100]}...")
    
    # 結果の戻り値を構築
    result_dict = {
        "query": query,
        "hits": [
            {
                "score": hit.score,
                "page": hit.payload["page"],
                "text": hit.payload["text"]
            } for hit in selected_results
        ],
        "pages_used": pages_used,
        "context_text": context_text
    }
    
    return result_dict

def main():
    parser = argparse.ArgumentParser(description='Qdrantを使用してテキストを検索し、関連ページを取得します')
    parser.add_argument('--query', type=str, required=True, help='検索クエリ')
    parser.add_argument('--embedding_model', type=str, default='sbintuitions/sarashina-embedding-v1-1b', help='使用する埋め込みモデル')
    parser.add_argument('--qdrant_host', type=str, default='localhost', help='Qdrantサーバーのホスト名')
    parser.add_argument('--qdrant_port', type=int, default=6333, help='QdrantサーバーのRESTポート番号')
    parser.add_argument('--collection_name', type=str, default='cheatsheet_docs', help='Qdrantのコレクション名')
    parser.add_argument('--limit', type=int, default=10, help='検索する上位結果数')
    parser.add_argument('--no_mmr', action='store_true', help='MMRによる再ランク付けを行わない')
    parser.add_argument('--mmr_top_n', type=int, default=5, help='MMRで選択する上位結果数')
    parser.add_argument('--lambda_param', type=float, default=0.7, help='MMRのバランスパラメータ')
    parser.add_argument('--output', type=str, help='検索結果を保存するJSONファイルのパス')
    args = parser.parse_args()
    
    result = search_and_retrieve(
        query=args.query,
        embedding_model=args.embedding_model,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection_name,
        limit=args.limit,
        use_mmr=not args.no_mmr,
        mmr_top_n=args.mmr_top_n,
        lambda_param=args.lambda_param
    )
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"検索結果を {args.output} に保存しました")
    
    print("\nコンテキストテキストのサンプル:")
    print(result["context_text"][:500] + "...")

if __name__ == "__main__":
    main() 
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from search_retrieve import search_and_retrieve
import json
import time

def generate_answer(query, context_text, model_id="rinna/qwen2.5-bakeneko-32b-instruct",
                   max_new_tokens=1024, temperature=0.7, top_p=0.8, top_k=20,
                   repetition_penalty=1.1, do_sample=True):
    """
    LLMを使用して、コンテキストと質問から回答を生成する

    Args:
        query: ユーザーからの質問
        context_text: 検索結果から取得したコンテキスト文
        model_id: 使用するモデルID
        max_new_tokens: 生成する最大トークン数
        temperature: 生成の温度パラメータ
        top_p: 生成のtop-pパラメータ 
        top_k: 生成のtop-kパラメータ
        repetition_penalty: 繰り返しペナルティ
        do_sample: サンプリングによる生成を行うかどうか
    
    Returns:
        生成された回答
    """
    start_time = time.time()
    print(f"\n--- LLMによる回答生成を開始します ({model_id}) ---")
    
    # デバイスの確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # トークナイザーのロード
    print("トークナイザーをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # モデルのロード
    print("モデルをロード中...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",        # 利用可能なGPUに自動割当
        torch_dtype=torch.bfloat16  # bfloat16精度でロード
    )
    
    print(f"モデルとトークナイザーのロード完了: {time.time() - start_time:.2f}秒")
    
    # プロンプトの構築
    system_message = "あなたは誠実で優秀な日本人のアシスタントです。与えられた情報をもとに質問に答えてください。"
    user_message = f"以下の情報があります。\n{context_text}\nこの情報に基づいて、次の質問に答えてください。\n質問: {query}"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    # チャット形式のプロンプトに変換
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # トークン化
    print("入力をトークン化中...")
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    input_token_count = input_ids.shape[1]
    print(f"入力トークン数: {input_token_count}")
    
    # 回答生成
    print(f"回答を生成中... (max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k})")
    generation_start = time.time()
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
    
    generation_time = time.time() - generation_start
    print(f"回答生成完了: {generation_time:.2f}秒")
    
    # デコード
    print("回答をデコード中...")
    response = tokenizer.decode(output_ids[0][input_token_count:], skip_special_tokens=True)
    
    total_time = time.time() - start_time
    print(f"合計処理時間: {total_time:.2f}秒")
    
    return response

def run_rag_pipeline(query, embedding_model="sbintuitions/sarashina-embedding-v1-1b",
                    llm_model="rinna/qwen2.5-bakeneko-32b-instruct",
                    qdrant_host="localhost", qdrant_port=6333,
                    collection_name="cheatsheet_docs", search_limit=10,
                    use_mmr=True, mmr_top_n=5, lambda_param=0.7,
                    max_new_tokens=1024, temperature=0.7):
    """
    RAGパイプライン全体を実行する関数
    
    Args:
        query: ユーザーからの質問
        embedding_model: 埋め込みモデルのパス
        llm_model: 使用するLLMモデルのパス
        その他のパラメータは各コンポーネントに渡される
    
    Returns:
        検索結果と生成された回答を含む辞書
    """
    print(f"\n=== RAGパイプラインを開始します ===")
    print(f"質問: {query}")
    
    start_time = time.time()
    
    # 1. 検索フェーズ
    print(f"\n--- 検索フェーズ ---")
    search_start = time.time()
    search_result = search_and_retrieve(
        query=query,
        embedding_model=embedding_model,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=collection_name,
        limit=search_limit,
        use_mmr=use_mmr,
        mmr_top_n=mmr_top_n,
        lambda_param=lambda_param
    )
    search_time = time.time() - search_start
    print(f"検索フェーズ完了: {search_time:.2f}秒")
    
    context_text = search_result["context_text"]
    
    # 2. 生成フェーズ
    print(f"\n--- 生成フェーズ ---")
    generation_start = time.time()
    response = generate_answer(
        query=query,
        context_text=context_text,
        model_id=llm_model,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    generation_time = time.time() - generation_start
    print(f"生成フェーズ完了: {generation_time:.2f}秒")
    
    # 3. 結果の整形
    result = {
        "query": query,
        "search_result": search_result,
        "response": response,
        "metrics": {
            "search_time": search_time,
            "generation_time": generation_time,
            "total_time": time.time() - start_time
        }
    }
    
    print(f"\n=== RAGパイプライン完了 (合計時間: {result['metrics']['total_time']:.2f}秒) ===")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Qdrantを用いたRAGシステムを実行')
    parser.add_argument('--query', type=str, required=True, help='ユーザーからの質問')
    parser.add_argument('--embedding_model', type=str, default='sbintuitions/sarashina-embedding-v1-1b', help='使用する埋め込みモデル')
    parser.add_argument('--llm_model', type=str, default='rinna/qwen2.5-bakeneko-32b-instruct', help='使用するLLMモデル')
    parser.add_argument('--qdrant_host', type=str, default='localhost', help='Qdrantサーバーのホスト名')
    parser.add_argument('--qdrant_port', type=int, default=6333, help='QdrantサーバーのRESTポート番号')
    parser.add_argument('--collection_name', type=str, default='cheatsheet_docs', help='Qdrantのコレクション名')
    parser.add_argument('--search_limit', type=int, default=10, help='検索する上位結果数')
    parser.add_argument('--no_mmr', action='store_true', help='MMRによる再ランク付けを行わない')
    parser.add_argument('--mmr_top_n', type=int, default=5, help='MMRで選択する上位結果数')
    parser.add_argument('--lambda_param', type=float, default=0.7, help='MMRのバランスパラメータ')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='生成する最大トークン数')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成の温度パラメータ')
    parser.add_argument('--output', type=str, help='結果を保存するJSONファイルのパス')
    args = parser.parse_args()
    
    result = run_rag_pipeline(
        query=args.query,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection_name,
        search_limit=args.search_limit,
        use_mmr=not args.no_mmr,
        mmr_top_n=args.mmr_top_n,
        lambda_param=args.lambda_param,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # 結果の表示
    print("\n" + "=" * 50)
    print("回答:")
    print(result["response"])
    print("=" * 50)
    
    # 結果の保存
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"結果を {args.output} に保存しました")

if __name__ == "__main__":
    main() 
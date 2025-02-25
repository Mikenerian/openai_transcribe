import os
from dotenv import load_dotenv
from openai import OpenAI

# 設定
OUTPUT_DIR = "output"  # 生成されたテキストファイルを格納

def read_text_file(file_path):
    """指定されたテキストファイルを読み込む"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def write_text_file(file_path, content):
    """指定されたテキストファイルに内容を書き込む"""
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

def summarize_text(client, text):
    """文字起こししたテキストを要約する"""
    prompt = f"""
    以下の音声の内容を基に、雑誌に掲載する紹介記事を2000字を目安に作成してください。
    
    --- 文字起こし内容 ---
    {text}
    
    --- 要約のルール ---
    - 重要なポイントを簡潔にまとめる
    - 冗長な表現を削ぎ落とす
    - 読みたくなるようなキャッチーなタイトルを冒頭につける
    """
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "あなたはプロのライターです。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content



if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(api_key=os.getenv('API_KEY'))

    tgt_text = 'hoge.txt'  # 要約するテキストファイルの指定
    input_path = os.path.join(OUTPUT_DIR, tgt_text)
    output_path = os.path.join(OUTPUT_DIR, f"summary_{tgt_text}")
    original_text = read_text_file(input_path)

    print(f"{tgt_text}の要約を開始します...")
    summarized_text = summarize_text(client, original_text)
    print(f"{tgt_text}の要約が完了しました！")

    write_text_file(output_path, summarized_text)

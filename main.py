import os
import re
import time
from typing import List, Tuple
from glob import glob
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI

# 設定
INPUT_DIR = 'input'  # 音声ファイルを格納
CONV_DIR = 'converted'  # 中間生成ファイルを格納
OUTPUT_DIR = "output"  # 生成されたテキストファイルを格納
SPLIT_TIME = 20 * 60 * 1000  # 1ファイルあたりの最大長（ミリ秒）
OVERLAP = 20 * 1000  # ファイル分割時の重複区間（ミリ秒）
INTERVAL = 30  # API 制限回避のためのインターバル（秒）

def setup_directories() -> None:
    """必要なディレクトリを作成する"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(CONV_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_audio_files(input_dir):
    """指定されたディレクトリから音声ファイルを読み込み"""
    extensions = ['*.mp3', '*.m4a', '*.wav']  # 対応している音声ファイルの形式
    audio_files = []

    for ext in extensions:
        files = glob(os.path.join(input_dir, ext)) + glob(os.path.join(input_dir, ext.upper()))
        for file_path in files:
            file_name = os.path.basename(file_path)
            _, extension = os.path.splitext(file_name)
            extension = extension.lower()

            try:
                if extension == '.mp3':
                    audio = AudioSegment.from_mp3(file_path)
                elif extension == '.m4a':
                    audio = AudioSegment.from_file(file_path)
                elif extension == '.wav':
                    audio = AudioSegment.from_wav(file_path)
                else:
                    raise ValueError(f"{file_name}は無効な形式のファイルです")

                audio_files.append((file_name, audio))  # ファイル名と音声ファイルをタプルとして格納

            except ValueError as e:
                print(e)  # エラーメッセージを表示

    return audio_files

def split_audio(file_name: str, audio: AudioSegment, split_time: int, overlap: int) -> None:
    """音声ファイルを分割して保存する"""
    base_name, _ = os.path.splitext(file_name)
    for i, x in enumerate(range(0, len(audio), split_time - overlap)):
        output_name = f"{base_name}_{i:04d}.mp3"
        segment_audio = audio[x:x + split_time]
        segment_audio.export(os.path.join(CONV_DIR, output_name), format="mp3")

def list_segmented_audio(converted_dir):
    """分割した音声ファイルの一覧取得"""
    mp3_files = glob(os.path.join(converted_dir, "*.mp3"))
    mp3_file_names = [os.path.basename(file_path) for file_path in mp3_files]
    mp3_file_names = sorted(mp3_file_names)
    return mp3_file_names

def transcribe_audio(converted_dir, file_name, client):
    """音声をテキスト化する"""
    file_path = os.path.join(converted_dir, file_name)  

    try:
        with open(file_path, "rb") as mp3_audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=mp3_audio
            )
            return transcript.text

    except Exception as e:
        print(f"Error transcribing {file_name}: {e}")
        return None

def save_transcript(converted_dir, file_name, txt):
    """テキストをファイルに保存する"""
    base_name, _ = os.path.splitext(file_name)
    txt_name = f"{base_name}.txt"
    with open(os.path.join(converted_dir, txt_name), "w") as f:
        f.write(txt)

def combine_text_files(converted_dir, output_dir):
    """分割されたテキストファイルを結合する"""
    txt_files = glob(os.path.join(converted_dir, "*.txt"))
    file_groups = {}
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        match = re.match(r"(.+)_(\d+)\.txt", file_name)
        if match:
            base_name = match.group(1)
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(file_path)

    for base_name, file_paths in file_groups.items():
        # ファイルパスを連番順にソート
        file_paths.sort(key=lambda x: int(re.match(r".*_(\d+)\.txt", os.path.basename(x)).group(1)))

        combined_text = ""
        for file_path in file_paths:
            with open(file_path, "r") as f:
                combined_text += f.read()
                combined_text += "\n"  # ファイル間に改行を入れる（必要に応じて）

        # 結合したテキストをoutputディレクトリに保存
        output_file_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(output_file_path, "w") as f:
            f.write(combined_text)



if __name__ == "__main__":
    # 変数の読み込み
    load_dotenv()
    setup_directories()
    client = OpenAI(api_key=os.getenv('API_KEY'))

    # 音声ファイルの取得
    audio_files = load_audio_files(INPUT_DIR)
    print(f"{len(audio_files)}個の音声ファイルが見つかりました")
    for file_name, _ in audio_files:
        print(file_name)

    # 音声ファイルの分割
    print("音声ファイルを分割します")
    for file_name, audio in audio_files:   
        split_audio(file_name, audio, SPLIT_TIME, OVERLAP)
    print("音声ファイルの分割が完了しました")
    
    # 分割した音声ファイルの読み込み
    segmented_audios = list_segmented_audio(CONV_DIR)
    for i, audio_name in enumerate(segmented_audios):
        print(f"{audio_name}: 音声をテキスト化しています...")
        txt = transcribe_audio(CONV_DIR, audio_name, client)
        save_transcript(CONV_DIR, audio_name, txt)
        print(f"{audio_name}: テキスト化した音声を保存しました")

        # 5ファイルごとにインターバルを設ける
        if (i + 1) % 5 == 0:
            print(f"{INTERVAL}秒間スリープします...")
            time.sleep(INTERVAL)
            print("スリープ終了")
    
    # テキストを結合して音声ファイルごとにまとめる
    print("テキストファイルをまとめます")
    combine_text_files(CONV_DIR, OUTPUT_DIR)
    print("すべての処理が完了しました")

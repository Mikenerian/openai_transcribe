
# 必要なモジュールのインストール
import os
import time
import fnmatch
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI

# 環境変数の呼び出し
# 事前にAPI_KEY='your api key' の形式で.envファイルに記述しておく
load_dotenv()
client = OpenAI(
    api_key=os.getenv('API_KEY')
)

# 音声ファイルを格納するフォルダの作成（未作成の場合）
input_dir = 'input'
os.makedirs(input_dir, exist_ok=True)

# 音声ファイルを変換・分割した結果を格納するフォルダの作成（未作成の場合）
conv_dir ='converted'
os.makedirs(conv_dir, exist_ok=True)

# テキスト化結果を格納するフォルダの作成（未作成の場合）
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)



# 指定フォルダから音声ファイルを取得
# NOTE: 音声ファイルとして想定する拡張子をextensionsに定義
extensions = ['*.mp3', '*.MP3', '*.m4a', '*.M4A', '*.wav', '*.WAV']
audio_files = []
for root, dirnames, filenames in os.walk(input_dir):
    for ext in extensions:
        for filename in fnmatch.filter(filenames, ext):
            audio_files.append(filename)

# 1ファイルあたりの最大長（ミリ秒）
split_time = 20 * 60 * 1000
overlap = 20 * 1000

print(f"{len(audio_files)}個の音声ファイルが見つかりました")
print(audio_files)

for num, fname in enumerate(audio_files):
    print(fname + " を音声テキスト化します")
    input_fname = os.path.join(input_dir, fname)
    output_fname = os.path.join(output_dir, fname)
    base_name, extension = os.path.splitext(fname)

    # ファイル情報から音声化
    extension = extension.lower()  # 拡張子を小文字に変換
    if extension == '.mp3':
        audio = AudioSegment.from_mp3(input_fname)
    elif extension == '.m4a':
        audio = AudioSegment.from_file(input_fname, 'm4a')
    elif extension == '.wav':
        audio = AudioSegment.from_wav(input_fname)
    else:
        print("音声ファイルの形式が無効です")

    # 音声ファイルの分割
    segments = []
    for i in range(0, len(audio), split_time - overlap):
        segments.append(audio[i:i + split_time])
    
    print(f"音声ファイルは{len(segments)}個に分割されました")

    # 分割ファイルごとに処理
    for i, segment in enumerate(segments):
        # 分割ファイルをmp3形式で保存
        output_mp3 = f"{base_name}_{i:04d}.mp3"
        segment.export(os.path.join(conv_dir, output_mp3), format="mp3")

        # mp3の読み込み
        mp3_audio = open(os.path.join(conv_dir, output_mp3), "rb")

        # 音声テキスト化
        # transcript = openai.Audio.transcribe("whisper-1", mp3_audio)
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=mp3_audio
        )
        txt = transcript.text

        # テキスト形式で保存
        output_txt = f"{base_name}_{i:04d}.txt"
        f = open(os.path.join(output_dir, output_txt), "w")
        f.write(txt)
        f.close()

        print("テキストを次の名前で保存しました： " + output_txt)
    
    # API制限を回避するため、念のためファイル単位でインターバルを設ける
    if num < len(audio_files) - 1:
        print("30秒のインターバルに入ります")
        time.sleep(30)
        print("インターバル終了。次のファイルを処理します")
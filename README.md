# OpenAI Transcribe

**「安価に文字起こし（音声テキスト化）を実現する方法」**

note: [【ライター必見！】ChatGPTを生み出したOpenAIは音声文字起こしも超高性能だったので使い方をまとめる - Whisper APIについて｜おがくずにゃんこ](https://note.com/mikenerian/n/n3a43ada3a792)

※以降の説明にはClaudeを活用しています。

音声ファイルを自動的にテキスト化し、要約まで作成する統合ツールです。OpenAI の Whisper API と GPT-4 を活用して、長時間の音声ファイルを効率的に処理します。

## 機能

- 📝 **並列文字起こし**: OpenAI Whisper API を使用した高精度な音声認識
- 📋 **自動要約**: GPT-4 による読みやすい要約の生成
- ⚡ **メモリ効率**: 大容量ファイルにも対応した省メモリ設計
- 🔄 **リトライ機能**: API エラー時の自動再試行

## 対応形式

- **音声**: MP3, M4A, WAV
- **出力**: UTF-8 テキストファイル

## 前提条件

### 必須

1. **OpenAI API キー**
   - [OpenAI Platform](https://platform.openai.com/api-keys) でアカウント作成・API キー取得
  ※事前に残高を確認ください

2. **ffmpeg**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # Windows
   # https://ffmpeg.org/download.html からダウンロード
   ```

3. **Python 3.8+**

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd openai_transcribe
```

### 2. 仮想環境の作成・アクティベート

```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

`.env` ファイルを作成し、OpenAI API キーを設定：

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## 使用方法

### 基本的なワークフロー

1. **音声ファイルを配置** → `input/` フォルダ
2. **文字起こし実行** → `main.py`
3. **要約作成** → `summarize.py`

### 1. 音声ファイルの文字起こし

```bash
# 音声ファイルを input/ フォルダに配置
cp your_audio.mp3 input/

# 文字起こし実行
python main.py
```

**処理フロー:**
- `input/*.{mp3,m4a,wav}` → 分割 → `converted/`
- `converted/*.mp3` → 文字起こし → `converted/*.txt`
- `converted/*.txt` → 結合 → `output/*.txt`

### 2. テキストの要約

```bash
# output/ フォルダの全テキストファイルを要約
python summarize.py

# 特定ファイルのみ要約
python summarize.py your_file.txt
```

**処理フロー:**
- `output/*.txt` → 要約 → `summarized/*_summary.txt`

## ディレクトリ構造

```
openai_transcribe/
├── input/           # 元の音声ファイル
├── converted/       # 分割された音声・テキスト（中間ファイル）
├── output/          # 結合されたテキストファイル
├── summarized/      # 要約されたテキストファイル
├── main.py          # 音声文字起こしメイン処理
├── summarize.py     # テキスト要約処理
├── requirements.txt # 依存関係
└── .env            # 環境変数（自分で作成）
```

## 設定のカスタマイズ

### 音声分割設定（main.py）

```python
@dataclass
class Config:
    split_time: int = 20 * 60 * 1000  # 分割時間（ミリ秒）
    overlap: int = 20 * 1000          # 重複時間（ミリ秒）
    max_workers: int = 3              # 並列処理数
    max_retries: int = 3              # リトライ回数
```

### 要約設定（summarize.py）

```python
@dataclass 
class SummaryConfig:
    model: str = 'gpt-4-turbo'        # 使用モデル
    target_length: int = 2000         # 目標文字数
    max_workers: int = 2              # 並列処理数
```

## トラブルシューティング

### よくあるエラー

**APIキー関連**
```
ValueError: OpenAI APIキーが設定されていません
```
→ `.env` ファイルに正しいAPIキーを設定してください

**ffmpeg関連**
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```
→ ffmpegをインストールしてください

**レート制限**
```
RateLimitError: Rate limit exceeded
```
→ 自動的にリトライされます。APIプランの確認をしてください

### ログの確認

- 文字起こし: `transcribe.log`
- 要約処理: `summarize.log`

## ライセンス

MIT License

## 貢献

pull request、issueは歓迎しますが、肝心の管理者の実力はClaude Code頼みです。

## 注意事項

- 本リポジトリが提供する機能によるアウトプットは個人利用の範囲で使用してください
- API使用量に注意し、定期的に利用状況を確認してください
- 機密情報を含む音声の処理時は、OpenAIのデータ利用ポリシーを確認してください
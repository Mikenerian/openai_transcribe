import os
import re
import time
import logging
import sys
from typing import List, Optional, Tuple
from dataclasses import dataclass
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError

@dataclass
class Config:
    """アプリケーション設定クラス"""
    input_dir: str = 'input'
    conv_dir: str = 'converted' 
    output_dir: str = 'output'
    split_time: int = 20 * 60 * 1000  # ミリ秒
    overlap: int = 20 * 1000  # ミリ秒
    interval: int = 30  # 秒
    max_retries: int = 3
    max_workers: int = 3
    api_key_env: str = 'OPENAI_API_KEY'
    supported_extensions: List[str] = None
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['*.mp3', '*.m4a', '*.wav']

# グローバル設定インスタンス
config = Config()


# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcribe.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_directories() -> bool:
    """必要なディレクトリを作成する"""
    try:
        os.makedirs(config.input_dir, exist_ok=True)
        os.makedirs(config.conv_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"ディレクトリを作成しました: {config.input_dir}, {config.conv_dir}, {config.output_dir}")
        return True
    except OSError as e:
        logger.error(f"ディレクトリの作成に失敗しました: {e}")
        return False

def get_audio_file_paths(input_dir: str = None) -> List[str]:
    """指定されたディレクトリから音声ファイルパスを取得（メモリ節約版）
    
    Args:
        input_dir: 音声ファイルディレクトリ（Noneの場合はconfig.input_dirを使用）
    """
    if input_dir is None:
        input_dir = config.input_dir
        
    audio_paths = []

    try:
        for ext in config.supported_extensions:
            files = glob(os.path.join(input_dir, ext)) + glob(os.path.join(input_dir, ext.upper()))
            for file_path in files:
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    audio_paths.append(file_path)
                    logger.info(f"音声ファイルを登録: {os.path.basename(file_path)}")
                else:
                    logger.warning(f"空または不正なファイルをスキップ: {file_path}")

    except Exception as e:
        logger.error(f"ディレクトリの読み込みに失敗: {input_dir} - {e}")
        return []

    return audio_paths

def load_and_split_audio(file_path: str, split_time: int = None, overlap: int = None) -> bool:
    """音声ファイルを読み込み、即座に分割してメモリを解放
    
    Args:
        file_path: 音声ファイルのパス
        split_time: 分割時間（Noneの場合はconfig.split_timeを使用）
        overlap: 重複時間（Noneの場合はconfig.overlapを使用）
    """
    if split_time is None:
        split_time = config.split_time
    if overlap is None:
        overlap = config.overlap
        
    file_name = os.path.basename(file_path)
    _, extension = os.path.splitext(file_name)
    extension = extension.lower()
    
    try:
        logger.info(f"音声ファイルを読み込み中: {file_name}")
        
        # メモリ効率を考慮して一時的に読み込み
        if extension == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
        elif extension == '.m4a':
            audio = AudioSegment.from_file(file_path)
        elif extension == '.wav':
            audio = AudioSegment.from_wav(file_path)
        else:
            logger.error(f"サポートされていない形式: {file_name}")
            return False
            
        # 即座に分割処理を実行
        result = split_audio_from_memory(file_name, audio, split_time, overlap)
        
        # メモリを明示的に解放
        del audio
        
        return result
        
    except Exception as e:
        logger.error(f"音声ファイルの処理に失敗: {file_name} - {e}")
        return False

def split_audio_from_memory(file_name: str, audio: AudioSegment, split_time: int, overlap: int) -> bool:
    """メモリ上の音声データを分割して保存（メモリ効率版）"""
    try:
        base_name, _ = os.path.splitext(file_name)
        segments_created = 0
        total_segments = (len(audio) + split_time - overlap - 1) // (split_time - overlap)
        
        logger.info(f"音声分割開始: {file_name} (予定セグメント数: {total_segments})")
        
        for i, x in enumerate(range(0, len(audio), split_time - overlap)):
            output_name = f"{base_name}_{i:04d}.mp3"
            output_path = os.path.join(config.conv_dir, output_name)
            
            # 既存ファイルをスキップ
            if os.path.exists(output_path):
                continue
                
            try:
                # セグメントを作成し、即座にエクスポートしてメモリ解放
                segment_audio = audio[x:x + split_time]
                segment_audio.export(output_path, format="mp3")
                del segment_audio  # メモリ解放
                segments_created += 1
                
                # 進捗ログ
                if (i + 1) % 10 == 0:
                    logger.info(f"分割進捗: {i + 1}/{total_segments} セグメント完了")
                    
            except Exception as e:
                logger.error(f"セグメント分割に失敗: {output_name} - {e}")
                return False
                
        logger.info(f"音声分割完了: {file_name} ({segments_created}個の新セグメント作成)")
        return True
        
    except Exception as e:
        logger.error(f"音声分割処理に失敗: {file_name} - {e}")
        return False

def list_segmented_audio(converted_dir: str) -> List[str]:
    """分割した音声ファイルの一覧を取得する
    
    Args:
        converted_dir: 分割済み音声ファイルが格納されているディレクトリ
        
    Returns:
        ソート済みの音声ファイル名のリスト
    """
    mp3_files = glob(os.path.join(converted_dir, "*.mp3"))
    mp3_file_names = [os.path.basename(file_path) for file_path in mp3_files]
    mp3_file_names = sorted(mp3_file_names)
    return mp3_file_names

def transcribe_audio_with_retry(converted_dir: str, file_name: str, client: OpenAI) -> Optional[str]:
    """音声をテキスト化する（リトライ機能付き）"""
    file_path = os.path.join(converted_dir, file_name)
    
    # ファイル存在チェック
    if not os.path.exists(file_path):
        logger.error(f"ファイルが見つかりません: {file_path}")
        return None
    
    for attempt in range(config.max_retries):
        try:
            with open(file_path, "rb") as mp3_audio:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=mp3_audio
                )
                logger.info(f"テキスト化成功: {file_name}")
                return transcript.text
                
        except RateLimitError as e:
            wait_time = (2 ** attempt) * 60  # 指数バックオフ
            logger.warning(f"レート制限エラー (試行 {attempt + 1}/{config.max_retries}): {file_name} - {wait_time}秒待機")
            if attempt < config.max_retries - 1:
                time.sleep(wait_time)
            else:
                logger.error(f"レート制限エラーでリトライ上限に達しました: {file_name}")
                
        except APIConnectionError as e:
            wait_time = (2 ** attempt) * 10
            logger.warning(f"API接続エラー (試行 {attempt + 1}/{config.max_retries}): {file_name} - {wait_time}秒待機")
            if attempt < config.max_retries - 1:
                time.sleep(wait_time)
            else:
                logger.error(f"API接続エラーでリトライ上限に達しました: {file_name}")
                
        except APIError as e:
            logger.error(f"API エラー: {file_name} - {e}")
            break
            
        except Exception as e:
            logger.error(f"予期しないエラー (試行 {attempt + 1}/{config.max_retries}): {file_name} - {e}")
            if attempt == config.max_retries - 1:
                break
            time.sleep(5)
    
    return None

def save_transcript(converted_dir: str, file_name: str, txt: str) -> bool:
    """テキストをファイルに保存する"""
    try:
        base_name, _ = os.path.splitext(file_name)
        txt_name = f"{base_name}.txt"
        txt_path = os.path.join(converted_dir, txt_name)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt)
        
        logger.info(f"テキスト保存完了: {txt_name}")
        return True
        
    except Exception as e:
        logger.error(f"テキスト保存に失敗: {file_name} - {e}")
        return False

def combine_text_files_efficient(converted_dir: str = None, output_dir: str = None) -> bool:
    """分割されたテキストファイルをメモリ効率よく結合する
    
    Args:
        converted_dir: 中間ファイルディレクトリ（Noneの場合はconfig.conv_dirを使用）
        output_dir: 出力ディレクトリ（Noneの場合はconfig.output_dirを使用）
    """
    if converted_dir is None:
        converted_dir = config.conv_dir
    if output_dir is None:
        output_dir = config.output_dir
        
    try:
        txt_files = glob(os.path.join(converted_dir, "*.txt"))
        if not txt_files:
            logger.warning("結合するテキストファイルが見つかりません")
            return False
            
        file_groups = {}
        for file_path in txt_files:
            file_name = os.path.basename(file_path)
            match = re.match(r"(.+)_(\d+)\.txt", file_name)
            if match:
                base_name = match.group(1)
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(file_path)

        success_count = 0
        for base_name, file_paths in file_groups.items():
            try:
                # ファイルパスを連番順にソート
                file_paths.sort(key=lambda x: int(re.match(r".*_(\d+)\.txt", os.path.basename(x)).group(1)))

                output_file_path = os.path.join(output_dir, f"{base_name}.txt")
                
                # メモリ効率を考慮してストリーミング処理
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    files_processed = 0
                    for file_path in file_paths:
                        try:
                            with open(file_path, "r", encoding="utf-8") as input_file:
                                # チャンクごとに読み込んでメモリ使用量を抑制
                                chunk_size = 8192  # 8KBチャンク
                                while True:
                                    chunk = input_file.read(chunk_size)
                                    if not chunk:
                                        break
                                    output_file.write(chunk)
                                output_file.write("\n")  # ファイル間の区切り
                                files_processed += 1
                                
                        except Exception as e:
                            logger.error(f"ファイル読み込みエラー: {file_path} - {e}")
                            continue
                    
                    logger.info(f"テキスト結合完了: {base_name}.txt ({files_processed}ファイルを結合)")
                    success_count += 1
                
            except Exception as e:
                logger.error(f"テキスト結合に失敗: {base_name} - {e}")
                continue
                
        logger.info(f"テキスト結合処理完了: {success_count}個のファイルを処理")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"テキスト結合処理でエラー: {e}")
        return False



def validate_api_key(api_key: Optional[str]) -> str:
    """APIキーの検証を行う
    
    Args:
        api_key: 検証するAPIキー
        
    Returns:
        検証済みのAPIキー
        
    Raises:
        ValueError: APIキーが無効な場合
    """
    if not api_key:
        raise ValueError("OpenAI APIキーが設定されていません")
    
    if not api_key.startswith('sk-'):
        raise ValueError("OpenAI APIキーの形式が正しくありません")
        
    if len(api_key) < 20:
        raise ValueError("OpenAI APIキーが短すぎます")
        
    return api_key

def initialize_environment() -> OpenAI:
    """環境の初期化とOpenAIクライアントの作成
    
    Returns:
        設定済みのOpenAIクライアント
        
    Raises:
        ValueError: 設定エラーの場合
        RuntimeError: ディレクトリ作成エラーの場合
    """
    load_dotenv()
    
    # OpenAI APIキーの確認と検証
    api_key = os.getenv(config.api_key_env) or os.getenv('API_KEY')  # 後方互換性
    validated_key = validate_api_key(api_key)
        
    if not setup_directories():
        raise RuntimeError("ディレクトリの作成に失敗しました")
        
    return OpenAI(api_key=validated_key)

def validate_audio_files() -> List[str]:
    """音声ファイルの検証とパス一覧の取得"""
    audio_file_paths = get_audio_file_paths()
    if not audio_file_paths:
        raise RuntimeError("処理する音声ファイルが見つかりません")
        
    logger.info(f"{len(audio_file_paths)}個の音声ファイルが見つかりました")
    for file_path in audio_file_paths:
        logger.info(f"音声ファイル: {os.path.basename(file_path)}")
        
    return audio_file_paths

def process_audio_splitting(audio_file_paths: List[str]) -> bool:
    """音声ファイルの分割処理"""
    logger.info("音声ファイルの分割を開始します（メモリ効率モード）")
    split_success = True
    
    for i, file_path in enumerate(audio_file_paths, 1):
        logger.info(f"処理中 ({i}/{len(audio_file_paths)}): {os.path.basename(file_path)}")
        
        if not load_and_split_audio(file_path):
            split_success = False
            logger.error(f"分割失敗: {os.path.basename(file_path)}")
        else:
            logger.info(f"分割完了: {os.path.basename(file_path)}")
            
    if not split_success:
        logger.error("一部の音声ファイル分割に失敗しました")
    else:
        logger.info("すべての音声ファイル分割が完了しました")
        
    return split_success

@dataclass
class TranscriptionResult:
    """テキスト化処理の結果"""
    success_count: int
    failed_count: int
    total_count: int
    
    @property
    def success_rate(self) -> float:
        """成功率を計算"""
        return self.success_count / self.total_count if self.total_count > 0 else 0.0

def process_transcription(client: OpenAI) -> TranscriptionResult:
    """並列テキスト化処理を実行する
    
    Args:
        client: OpenAIクライアント
        
    Returns:
        テキスト化処理の結果
        
    Raises:
        RuntimeError: 分割音声ファイルが見つからない場合
    """
    segmented_audios = list_segmented_audio(config.conv_dir)
    if not segmented_audios:
        raise RuntimeError("分割された音声ファイルが見つかりません")
        
    logger.info(f"{len(segmented_audios)}個の分割音声ファイルを処理します")
    
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # タスクを作成
        tasks = [(config.conv_dir, audio_name, client) for audio_name in segmented_audios]
        
        # 並列実行
        future_to_audio = {executor.submit(process_single_audio, task): task[1] for task in tasks}
        
        for future in as_completed(future_to_audio):
            audio_name = future_to_audio[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"並列処理でエラー: {audio_name} - {e}")
                failed_count += 1
    
    result = TranscriptionResult(success_count, failed_count, len(segmented_audios))
    logger.info(f"テキスト化完了: 成功 {result.success_count}件, 失敗 {result.failed_count}件 (成功率: {result.success_rate:.1%})")
    return result

def process_text_combination() -> bool:
    """テキストファイルの結合処理"""
    logger.info("テキストファイルの結合を開始します（メモリ効率モード）")
    
    if combine_text_files_efficient():
        logger.info("テキスト結合が正常に完了しました")
        return True
    else:
        logger.error("テキストファイルの結合に失敗しました")
        return False

def process_single_audio(args: Tuple[str, str, OpenAI]) -> bool:
    """単一音声ファイルのテキスト化処理（並列処理用）
    
    Args:
        args: (converted_dir, audio_name, client)のタプル
        
    Returns:
        処理が成功した場合True、失敗した場合False
    """
    converted_dir, audio_name, client = args
    txt_path = os.path.join(converted_dir, f"{os.path.splitext(audio_name)[0]}.txt")
    
    # 既存テキストファイルをスキップ
    if os.path.exists(txt_path):
        logger.info(f"既存テキストファイルをスキップ: {audio_name}")
        return True
    
    logger.info(f"テキスト化開始: {audio_name}")
    txt = transcribe_audio_with_retry(converted_dir, audio_name, client)
    
    if txt:
        return save_transcript(converted_dir, audio_name, txt)
    else:
        logger.error(f"テキスト化に失敗: {audio_name}")
        return False

def main() -> None:
    """メイン処理関数
    
    Raises:
        ValueError: 設定エラーの場合
        RuntimeError: 処理エラーの場合
    """
    logger.info("音声テキスト化処理を開始します")
    
    try:
        # 1. 環境初期化
        client = initialize_environment()
        
        # 2. 音声ファイル検証
        audio_file_paths = validate_audio_files()
        
        # 3. 音声分割処理
        process_audio_splitting(audio_file_paths)
        
        # 4. テキスト化処理
        result = process_transcription(client)
        
        # 5. テキスト結合処理
        if not process_text_combination():
            raise RuntimeError("テキストファイルの結合に失敗しました")
        
        # 6. 処理結果のサマリ表示
        logger.info("■ 処理結果サマリ ■")
        logger.info(f"音声ファイル数: {len(audio_file_paths)}件")
        logger.info(f"テキスト化成功: {result.success_count}件")
        logger.info(f"テキスト化失敗: {result.failed_count}件")
        logger.info(f"成功率: {result.success_rate:.1%}")
        logger.info("すべての処理が正常に完了しました")
        
    except (ValueError, RuntimeError) as e:
        logger.error(f"処理エラー: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("処理が中断されました")
        sys.exit(130)  # SIGINT標準終了コード
    except (ValueError, RuntimeError) as e:
        logger.error(f"設定または実行時エラー: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        logger.exception("詳細なエラー情報:")
        sys.exit(2)

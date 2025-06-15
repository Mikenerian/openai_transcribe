import os
import sys
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError

@dataclass
class SummaryConfig:
    """要約機能の設定クラス"""
    input_dir: str = 'output'  # 元のテキストファイルがあるディレクトリ
    output_dir: str = 'summarized'  # 要約ファイルを保存するディレクトリ
    api_key_env: str = 'OPENAI_API_KEY'
    model: str = 'gpt-4-turbo'
    max_retries: int = 3
    max_workers: int = 2
    chunk_size: int = 8192
    target_length: int = 2000
    supported_extensions: List[str] = None
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['*.txt']

# グローバル設定インスタンス
config = SummaryConfig()

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('summarize.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SummaryResult:
    """要約処理の結果"""
    success_count: int
    failed_count: int
    total_count: int
    
    @property
    def success_rate(self) -> float:
        """成功率を計算"""
        return self.success_count / self.total_count if self.total_count > 0 else 0.0

def setup_directories() -> bool:
    """必要なディレクトリを作成する"""
    try:
        # input_dir (元テキストファイル) の存在確認
        if not os.path.exists(config.input_dir):
            logger.error(f"入力ディレクトリが存在しません: {config.input_dir}")
            return False
            
        # output_dir (要約ファイル保存先) を作成
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"ディレクトリを確認/作成しました: {config.input_dir} -> {config.output_dir}")
        return True
    except OSError as e:
        logger.error(f"ディレクトリの作成に失敗しました: {e}")
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

def read_text_file_safe(file_path: str) -> Optional[str]:
    """指定されたテキストファイルを安全に読み込む
    
    Args:
        file_path: 読み込むファイルのパス
        
    Returns:
        ファイルの内容、エラー時はNone
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"ファイルが見つかりません: {file_path}")
            return None
            
        if os.path.getsize(file_path) == 0:
            logger.warning(f"空のファイルです: {file_path}")
            return None
            
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if not content:
                logger.warning(f"ファイルの内容が空です: {file_path}")
                return None
            return content
            
    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {file_path} - {e}")
        return None

def write_text_file_safe(file_path: str, content: str) -> bool:
    """指定されたテキストファイルに安全に書き込む
    
    Args:
        file_path: 書き込み先ファイルのパス
        content: 書き込む内容
        
    Returns:
        成功時True、失敗時False
    """
    try:
        # ディレクトリが存在しない場合は作成
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
            
        logger.info(f"ファイル書き込み完了: {os.path.basename(file_path)}")
        return True
        
    except Exception as e:
        logger.error(f"ファイル書き込みエラー: {file_path} - {e}")
        return False

def create_summary_prompt(text: str, target_length: int = None) -> str:
    """要約用のプロンプトを作成する
    
    Args:
        text: 要約対象のテキスト
        target_length: 目標文字数
        
    Returns:
        要約用プロンプト
    """
    if target_length is None:
        target_length = config.target_length
        
    return f"""
以下の音声の内容を基に、雑誌に掲載する紹介記事を{target_length}字を目安に作成してください。

--- 文字起こし内容 ---
{text}

--- 要約のルール ---
- 重要なポイントを簡潔にまとめる
- 冗長な表現を削ぎ落とす
- 読みたくなるようなキャッチーなタイトルを冒頭につける
- 構造化された読みやすい形式で出力する
"""

def summarize_text_with_retry(client: OpenAI, text: str, file_name: str = "不明") -> Optional[str]:
    """テキストを要約する（リトライ機能付き）
    
    Args:
        client: OpenAIクライアント
        text: 要約対象のテキスト
        file_name: ファイル名（ログ用）
        
    Returns:
        要約されたテキスト、失敗時はNone
    """
    if not text or not text.strip():
        logger.error(f"要約対象のテキストが空です: {file_name}")
        return None
        
    prompt = create_summary_prompt(text)
    
    for attempt in range(config.max_retries):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": "あなたはプロのライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            summary = response.choices[0].message.content
            if summary and summary.strip():
                logger.info(f"要約成功: {file_name}")
                return summary.strip()
            else:
                logger.warning(f"空の要約が返されました: {file_name}")
                return None
                
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

def get_text_file_paths(input_dir: str = None) -> List[str]:
    """指定されたディレクトリからテキストファイルパスを取得
    
    Args:
        input_dir: テキストファイルディレクトリ（Noneの場合はconfig.input_dirを使用）
        
    Returns:
        テキストファイルパスのリスト
    """
    if input_dir is None:
        input_dir = config.input_dir
        
    text_paths = []
    
    try:
        for ext in config.supported_extensions:
            files = glob(os.path.join(input_dir, ext))
            for file_path in files:
                # summary_ から始まるファイルはスキップ（既に要約済み）
                if os.path.basename(file_path).startswith('summary_'):
                    continue
                    
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    text_paths.append(file_path)
                    logger.info(f"テキストファイルを登録: {os.path.basename(file_path)}")
                else:
                    logger.warning(f"空または不正なファイルをスキップ: {file_path}")
                    
    except Exception as e:
        logger.error(f"ディレクトリの読み込みに失敗: {input_dir} - {e}")
        return []
        
    return text_paths

def process_single_file(args: tuple) -> bool:
    """単一ファイルの要約処理（並列処理用）
    
    Args:
        args: (file_path, client, output_dir)のタプル
        
    Returns:
        処理が成功した場合True、失敗した場合False
    """
    file_path, client, output_dir = args
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    output_path = os.path.join(output_dir, f"{base_name}_summary{ext}")
    
    # 既存要約ファイルをスキップ
    if os.path.exists(output_path):
        logger.info(f"既存要約ファイルをスキップ: {file_name} -> {os.path.basename(output_path)}")
        return True
    
    logger.info(f"要約開始: {file_name}")
    
    # テキスト読み込み
    original_text = read_text_file_safe(file_path)
    if not original_text:
        logger.error(f"テキスト読み込みに失敗: {file_name}")
        return False
    
    # 要約処理
    summarized_text = summarize_text_with_retry(client, original_text, file_name)
    if not summarized_text:
        logger.error(f"要約処理に失敗: {file_name}")
        return False
    
    # 要約ファイル保存
    if write_text_file_safe(output_path, summarized_text):
        logger.info(f"要約完了: {file_name} -> {os.path.basename(output_path)}")
        return True
    else:
        logger.error(f"要約ファイル保存に失敗: {file_name}")
        return False

def process_multiple_files(client: OpenAI, file_paths: List[str]) -> SummaryResult:
    """複数ファイルの並列要約処理
    
    Args:
        client: OpenAIクライアント
        file_paths: 処理対象ファイルパスのリスト
        
    Returns:
        要約処理の結果
    """
    if not file_paths:
        logger.warning("処理対象のファイルがありません")
        return SummaryResult(0, 0, 0)
    
    logger.info(f"{len(file_paths)}個のテキストファイルを要約処理します")
    
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # タスクを作成
        tasks = [(file_path, client, config.output_dir) for file_path in file_paths]
        
        # 並列実行
        future_to_file = {executor.submit(process_single_file, task): task[0] for task in tasks}
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            file_name = os.path.basename(file_path)
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"並列処理でエラー: {file_name} - {e}")
                failed_count += 1
    
    result = SummaryResult(success_count, failed_count, len(file_paths))
    logger.info(f"要約処理完了: 成功 {result.success_count}件, 失敗 {result.failed_count}件 (成功率: {result.success_rate:.1%})")
    return result

def main(target_file: Optional[str] = None) -> None:
    """メイン処理関数
    
    Args:
        target_file: 処理対象ファイル名（Noneの場合は全ファイル処理）
        
    Raises:
        ValueError: 設定エラーの場合
        RuntimeError: 処理エラーの場合
    """
    logger.info("テキスト要約処理を開始します")
    
    try:
        # 1. 環境初期化
        client = initialize_environment()
        
        # 2. 処理対象ファイルの特定
        if target_file:
            # 単一ファイル処理 (input_dirから検索)
            file_path = os.path.join(config.input_dir, target_file)
            if not os.path.exists(file_path):
                raise RuntimeError(f"指定されたファイルが{config.input_dir}フォルダに見つかりません: {target_file}")
            file_paths = [file_path]
            logger.info(f"単一ファイル処理モード: {target_file}")
        else:
            # 全ファイル処理 (input_dirから検索)
            file_paths = get_text_file_paths()
            if not file_paths:
                raise RuntimeError(f"{config.input_dir}フォルダに処理するテキストファイルが見つかりません")
            logger.info(f"全ファイル処理モード: {len(file_paths)}件 (from {config.input_dir})")
        
        # 3. 要約処理実行
        result = process_multiple_files(client, file_paths)
        
        # 4. 処理結果のサマリ表示
        logger.info("■ 処理結果サマリ ■")
        logger.info(f"処理対象ファイル数: {result.total_count}件 (from {config.input_dir})")
        logger.info(f"要約成功: {result.success_count}件 (to {config.output_dir})")
        logger.info(f"要約失敗: {result.failed_count}件")
        logger.info(f"成功率: {result.success_rate:.1%}")
        
        if result.success_count > 0:
            logger.info(f"要約処理が正常に完了しました - 要約ファイルは{config.output_dir}フォルダに保存されました")
        else:
            logger.warning("要約処理で成功したファイルがありません")
        
    except (ValueError, RuntimeError) as e:
        logger.error(f"処理エラー: {e}")
        raise

if __name__ == "__main__":
    try:
        # コマンドライン引数から対象ファイルを取得（オプション）
        target_file = sys.argv[1] if len(sys.argv) > 1 else None
        main(target_file)
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

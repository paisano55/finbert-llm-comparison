# FinBERT vs OpenAI 금융 감성분석 벤치마크

이 프로젝트는 **FinBERT(ProsusAI/finbert)** 와 **OpenAI 생성형 LLM**의 금융 문장 감성분석 성능을 비교합니다.

## 구현된 스펙

- Python 기반 구현
- OpenAI API 사용
- `uv` 패키지 매니저 사용
- GPU(CUDA) 환경 자동 감지 후 FinBERT 추론에 사용
- 데이터셋:
- 기본: FinBERT 저장소에 명시된 설정을 재현 (`Financial PhraseBank`의 `sentences_50agree`, 72/8/20 split 중 test)
  - 대체: `PhraseBank`에서 1000개 샘플 추출 모드 제공
- OpenAI API 배치 추론 (한 번에 `n`개 문장 처리)
- FinBERT/OpenAI 모두 실행 시간 측정
- OpenAI 토큰 사용량 측정

> 참고: FinBERT 기준 데이터는 **동일 분할 방법(method-equivalent)** 을 재현합니다.
> (20% test, `random_state=0`)

## 사전 준비

1. OpenAI API Key 설정

```bash
set OPENAI_API_KEY=your_api_key_here
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

2. CUDA 환경에서 `torch`가 GPU 빌드로 설치되어 있는지 확인

## 설치

```bash
uv sync --extra dev
```

## 실행

기본 실행 (FinBERT reference eval split 사용):

```bash
uv run fin-sentiment-benchmark --openai-model gpt-4o-mini --allow-openai-data-transfer
```

1000개 샘플 fallback 모드:

```bash
uv run fin-sentiment-benchmark --dataset-mode phrasebank_1000 --openai-model gpt-4o-mini --allow-openai-data-transfer
```

주요 옵션:

- `--openai-batch-size`: OpenAI 한 요청당 문장 수
- `--finbert-batch-size`: FinBERT 추론 배치 크기
- `--output-path`: 결과 JSON 저장 경로
- `--allow-openai-data-transfer`: PhraseBank 문장을 OpenAI API로 전송하는 것에 대한 명시적 동의

예시:

```bash
uv run fin-sentiment-benchmark \
  --openai-model gpt-4o-mini \
  --openai-batch-size 16 \
  --finbert-batch-size 64 \
  --allow-openai-data-transfer \
  --output-path outputs/benchmark_result.json
```

## 출력

실행 후 JSON 리포트에 아래 정보가 포함됩니다.

- 데이터셋 정보 (source/config/split/샘플 수)
- FinBERT 성능 (accuracy, macro_f1, elapsed_seconds)
- OpenAI 성능 (accuracy, macro_f1, elapsed_seconds)
- OpenAI 토큰 사용량 (`prompt_tokens`, `completion_tokens`, `total_tokens`)

## 참고

- FinBERT repo의 데이터 전처리 스크립트(`scripts/datasets.py`)는 `sentences_50agree`에서
  `train_test_split(... test_size=0.2, random_state=0)`을 사용합니다.
- 본 프로젝트 기본 모드는 동일한 **분할 방법(method-equivalent) 기반 평가 split(test)** 을 재현합니다.
- OpenAI 비교를 실행하면 문장 텍스트가 OpenAI API로 전송됩니다.

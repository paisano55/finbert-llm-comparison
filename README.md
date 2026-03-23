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
- PhraseBank는 Hugging Face의 원본 zip 파일을 직접 받아 파싱

> 참고: FinBERT 기준 데이터는 **동일 분할 방법(method-equivalent)** 을 재현합니다.
> (20% test, `random_state=0`)

## 사전 준비

1. OpenAI API Key 설정 (`--run-target` 이 `both` 또는 `openai`일 때만 필요)

프로젝트 루트의 `.env` 파일을 사용합니다.

```bash
OPEN_API_KEY=your_api_key_here
```

`OPENAI_API_KEY` 이름으로 적어도 동작합니다.

다른 파일명을 쓰고 싶으면 실행 시 `--openai-env-file` 로 지정할 수 있습니다.

예시:

```bash
uv run fin-sentiment-benchmark \
  --openai-env-file .secrets/openai.env \
  --openai-model gpt-4o-mini \
  --allow-openai-data-transfer
```

2. CUDA 환경에서 `torch`가 GPU 빌드로 설치되어 있는지 확인

3. 첫 실행 시 FinBERT 모델과 PhraseBank zip 파일을 Hugging Face에서 자동 다운로드

`HF_TOKEN`은 필수는 아니지만, 설정하면 rate limit과 다운로드 속도 면에서 유리합니다.

## 설치

```bash
uv sync --extra dev
```

Windows에서 CUDA 12.8 GPU 추론을 쓰려면 `uv`가 `torch`를 PyTorch의 CUDA 12.8 인덱스에서 받도록 설정되어 있습니다.
기존 CPU 빌드가 이미 설치돼 있었다면 아래처럼 한 번 다시 동기화하는 편이 안전합니다.

```powershell
uv sync --extra dev --reinstall-package torch
```

## 실행

기본 실행 (FinBERT reference eval split 사용):

```bash
uv run fin-sentiment-benchmark --openai-model gpt-4o-mini --allow-openai-data-transfer
```

FinBERT만 실행:

```bash
uv run fin-sentiment-benchmark --run-target finbert
```

OpenAI만 실행:

```bash
uv run fin-sentiment-benchmark --run-target openai --allow-openai-data-transfer
```

1000개 샘플 fallback 모드:

```bash
uv run fin-sentiment-benchmark --dataset-mode phrasebank_1000 --openai-model gpt-4o-mini --allow-openai-data-transfer
```

주요 옵션:

- `--openai-batch-size`: OpenAI 한 요청당 문장 수
- `--finbert-batch-size`: FinBERT 추론 배치 크기
- `--run-target`: `both`, `finbert`, `openai` 중 실행 대상 선택
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

`--run-target finbert` 또는 `--run-target openai`로 실행하면 선택하지 않은 쪽 결과는 JSON에서 `null`로 기록됩니다.

## 참고

- FinBERT repo의 데이터 전처리 스크립트(`scripts/datasets.py`)는 `sentences_50agree`에서
  `train_test_split(... test_size=0.2, random_state=0)`을 사용합니다.
- 본 프로젝트 기본 모드는 동일한 **분할 방법(method-equivalent) 기반 평가 split(test)** 을 재현합니다.
- OpenAI 비교를 실행하면 문장 텍스트가 OpenAI API로 전송됩니다.

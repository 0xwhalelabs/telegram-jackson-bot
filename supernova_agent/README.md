# Supernova Yield-Farming Agent 🌟

**Supernova DEX** (https://supernova.xyz) 에서 자동으로 최적의 유동성 풀을 찾아 이동하며 수익을 극대화하는 에이전트입니다.

## 핵심 기능

- **실시간 풀 모니터링**: Goldsky 서브그래프 + 온체인 데이터로 Basic/Concentrated 풀 스캔
- **APR 기반 스코어링**: Fee APR + Emissions APR + TVL 안정성을 종합 평가
- **자동 마이그레이션**: 더 높은 수익률 풀 발견 시 자동으로 유동성 이동
  - Gauge에서 언스테이크 → 유동성 제거 → 토큰 스왑 → 새 풀에 유동성 추가 → Gauge 스테이크
- **가스비 최적화**: 가스비 대비 APR 개선 효과를 계산하여 손해 나는 이동 방지
- **Telegram 알림/제어**: 실시간 알림 + 명령어로 에이전트 제어

## 아키텍처

```
supernova_agent/
├── main.py          # 진입점 (에이전트 + 텔레그램 봇 실행)
├── agent.py         # 메인 에이전트 루프 (스캔 → 평가 → 마이그레이션)
├── config.py        # 환경변수 설정
├── abis.py          # 스마트 컨트랙트 ABI 정의
├── subgraph.py      # Goldsky 서브그래프 데이터 페칭
├── chain.py         # 온체인 상호작용 (Web3, 트랜잭션)
├── strategy.py      # 풀 스코어링 & 마이그레이션 전략
├── notifier.py      # Telegram 알림 & 명령어 핸들러
├── .env.example     # 환경변수 템플릿
└── requirements.txt # Python 의존성
```

## 작동 흐름

```
1. 서브그래프에서 Basic + Concentrated 풀 데이터 수집
2. 각 풀의 Gauge에서 emissions 보상률 조회
3. Fee APR + Emissions APR + TVL 기반 종합 스코어 계산
4. 현재 포지션 대비 최고 풀의 APR 차이 확인
5. APR 개선 > 최소 기준(2%) && 가스비 7일 내 회수 가능 → 마이그레이션 실행
6. 5분마다 반복
```

## 설치 & 실행

### 1. 의존성 설치
```bash
cd supernova_agent
pip install -r requirements.txt
```

### 2. 환경변수 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 값 입력
```

**필수 설정:**
- `ETH_RPC_URL` – Ethereum RPC 엔드포인트 (Alchemy, Infura 등)
- `PRIVATE_KEY` – 지갑 프라이빗 키 ⚠️ 절대 공유 금지
- `WALLET_ADDRESS` – 지갑 주소

**선택 설정:**
- `SN_TELEGRAM_BOT_TOKEN` – Telegram 봇 토큰
- `SN_TELEGRAM_CHAT_ID` – 알림 받을 채팅 ID

### 3. 실행

**Dry Run 모드 (시뮬레이션만, 기본값):**
```bash
python main.py
```

**실제 트랜잭션 실행:**
```bash
# .env에서 DRY_RUN=false 로 변경 후
python main.py
```

## Telegram 명령어

| 명령어 | 설명 |
|--------|------|
| `/sn_status` | 현재 에이전트 상태 확인 |
| `/sn_pools` | 상위 10개 풀 목록 |
| `/sn_pause` | 에이전트 일시정지 |
| `/sn_resume` | 에이전트 재개 |
| `/sn_migrate` | 수동 마이그레이션 체크 |

## 마이그레이션 조건

에이전트는 다음 조건을 **모두** 만족할 때만 풀을 이동합니다:

1. **APR 개선**: 새 풀의 APR이 현재 풀보다 `MIN_APR_IMPROVEMENT`% 이상 높음
2. **가스비 회수**: 마이그레이션 가스비를 7일 이내에 APR 차이로 회수 가능
3. **가스 가격**: 현재 가스 가격이 `MAX_GAS_PRICE_GWEI` 이하
4. **최소 TVL**: 대상 풀의 TVL이 `MIN_POOL_TVL_USD` 이상

## 풀 타입

### Basic Pools (Uniswap V2 스타일)
- **Volatile**: `x * y = k` (변동성 높은 토큰 쌍)
- **Stable**: `x³y + y³x ≥ k` (스테이블코인 쌍, 낮은 슬리피지)

### Concentrated Pools (Algebra 기반)
- 틱 기반 범위 유동성
- 자본 효율성 극대화
- 동적 수수료 모델

## ⚠️ 주의사항

- **DRY_RUN=true** 상태에서 충분히 테스트 후 실제 모드로 전환하세요
- 프라이빗 키는 절대 공유하지 마세요
- 이 에이전트는 스마트 컨트랙트 리스크, 비영구적 손실(IL) 등의 DeFi 리스크를 제거하지 않습니다
- 소액으로 먼저 테스트하는 것을 강력히 권장합니다
- Ethereum 메인넷의 가스비가 높을 수 있으므로 `MAX_GAS_PRICE_GWEI` 설정을 적절히 조정하세요

## 컨트랙트 주소 (Mainnet)

| 컨트랙트 | 주소 |
|----------|------|
| RouterV2 | `0xbFAe8E87053309fDe07ab3cA5f4B5345f8e3058f` |
| PairFactory | `0x5aef44edfc5a7edd30826c724ea12d7be15bdc30` |
| GaugeManager | `0x19a410046afc4203aece5fbfc7a6ac1a4f517ae2` |
| NonfungiblePositionManager | `0x00d5bbd0fe275efee371a2b34d0a4b95b0c8aaaa` |
| Algebra Factory | `0x44b7fbd4d87149efa5347c451e74b9fd18e89c55` |
| VoterV3 | `0x1c7bf2532dfa34eeea02c3759e0ca8d87b1d8171` |

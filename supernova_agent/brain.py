"""
Supernova Yield-Farming Agent – Gemini AI Brain
모든 전략적 판단을 Gemini가 수행합니다:
- 어떤 풀에 진입할지
- 언제 이동할지
- 리스크 평가
- 시장 상황 분석
"""
import json
import logging
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_POOL_FEE

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini 클라이언트 초기화 완료: %s", GEMINI_MODEL)
    return _client


# ── 시스템 프롬프트 (에이전트 자아) ──────────────────────────────

SYSTEM_PROMPT = """당신은 Supernova DEX에서 유동성 풀을 관리하는 전문 DeFi 에이전트입니다.

## 당신의 역할
- 실시간 풀 데이터를 분석하여 최적의 수익 전략을 결정합니다
- 모든 판단은 수익 극대화 + 리스크 최소화를 기준으로 합니다
- 한국어로 판단 근거를 설명합니다

## 절대 규칙 (반드시 지켜야 함)
1. **fee가 0.01 (1%) 이상인 풀에는 절대 진입하지 않습니다** - 수수료가 너무 높아 수익을 갉아먹음
2. TVL이 $10,000 미만인 풀은 유동성 리스크가 높으므로 피합니다
3. 가스비 대비 수익이 7일 내 회수 불가능하면 이동하지 않습니다

## 판단 기준
- **APR**: 높을수록 좋지만, 비정상적으로 높은 APR(>500%)은 의심해야 함
- **TVL**: 높을수록 안정적. $100K 이상 선호
- **거래량**: 24시간 거래량이 TVL 대비 높으면 수수료 수익이 좋음
- **풀 타입**: stable 풀은 비영구적 손실(IL) 낮음, volatile 풀은 IL 높지만 수익도 높을 수 있음
- **fee**: 낮을수록 좋음. 0.01 미만만 허용

## 응답 형식
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.

```json
{
  "decision": "migrate" | "hold" | "exit",
  "target_pool": "풀 주소 (migrate일 때만)",
  "confidence": 0.0~1.0,
  "reasoning": "한국어로 판단 근거 설명",
  "risk_assessment": "low" | "medium" | "high",
  "risk_factors": ["리스크 요인 목록"],
  "market_insight": "현재 시장 상황에 대한 분석"
}
```
"""


# ── 풀 데이터를 Gemini에게 보낼 형식으로 변환 ─────────────────────

def _format_pool_for_ai(pool_data: Dict[str, Any]) -> Dict[str, Any]:
    """풀 데이터를 AI가 이해하기 쉬운 형태로 변환"""
    return {
        "address": pool_data.get("address", ""),
        "symbol": pool_data.get("symbol", ""),
        "pool_type": pool_data.get("pool_type", ""),
        "tvl_usd": round(pool_data.get("tvl_usd", 0), 2),
        "fee_apr": round(pool_data.get("fee_apr", 0), 2),
        "emissions_apr": round(pool_data.get("emissions_apr", 0), 2),
        "total_apr": round(pool_data.get("total_apr", 0), 2),
        "fee_rate": pool_data.get("fee_rate", 0),
        "volume_24h": round(pool_data.get("volume_24h", 0), 2),
        "score": round(pool_data.get("score", 0), 2),
    }


def _pool_score_to_dict(pool) -> Dict[str, Any]:
    """PoolScore 객체를 dict로 변환"""
    return {
        "address": pool.address,
        "symbol": pool.symbol,
        "pool_type": pool.pool_type,
        "tvl_usd": pool.tvl_usd,
        "fee_apr": pool.fee_apr,
        "emissions_apr": pool.emissions_apr,
        "total_apr": pool.total_apr,
        "fee_rate": getattr(pool, "fee_bps", 0) / 10000 if getattr(pool, "fee_bps", 0) else 0,
        "volume_24h": 0,
        "score": pool.score,
    }


# ── AI 판단 결과 파싱 ────────────────────────────────────────────

def _parse_ai_response(text: str) -> Dict[str, Any]:
    """Gemini 응답에서 JSON을 추출하여 파싱"""
    text = text.strip()

    # ```json ... ``` 블록 추출
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.error("Gemini 응답 JSON 파싱 실패: %s", text[:300])
        return {
            "decision": "hold",
            "confidence": 0.0,
            "reasoning": "AI 응답 파싱 실패 – 안전을 위해 현재 포지션 유지",
            "risk_assessment": "high",
            "risk_factors": ["AI 응답 파싱 오류"],
            "market_insight": "",
        }

    # 필수 필드 검증
    valid_decisions = {"migrate", "hold", "exit"}
    if result.get("decision") not in valid_decisions:
        result["decision"] = "hold"

    # fee 규칙 강제 적용: AI가 fee 높은 풀을 추천해도 차단
    if result.get("decision") == "migrate" and result.get("target_pool"):
        # 이 검증은 호출하는 쪽에서 수행
        pass

    return result


# ── 핵심 AI 판단 함수들 ──────────────────────────────────────────

async def analyze_pools_and_decide(
    ranked_pools: list,
    current_pool: Optional[Any] = None,
    gas_price_gwei: float = 20.0,
    eth_balance: float = 0.0,
) -> Dict[str, Any]:
    """
    Gemini에게 전체 풀 데이터를 보내고 전략적 판단을 받습니다.
    모든 결정(진입, 이동, 유지, 퇴장)을 AI가 수행합니다.
    """
    # 풀 데이터 준비 (상위 20개)
    pools_for_ai = []
    for p in ranked_pools[:20]:
        pools_for_ai.append(_pool_score_to_dict(p))

    # 현재 포지션 정보
    current_info = None
    if current_pool:
        current_info = _pool_score_to_dict(current_pool)

    prompt = f"""## 현재 상황

### 내 포지션
{json.dumps(current_info, indent=2, ensure_ascii=False) if current_info else "없음 (아직 어떤 풀에도 진입하지 않음)"}

### 가스 가격
현재 가스: {gas_price_gwei:.1f} gwei
ETH 잔고: {eth_balance:.4f} ETH

### 상위 풀 목록 (APR 스코어 순)
{json.dumps(pools_for_ai, indent=2, ensure_ascii=False)}

### 규칙 리마인더
- fee_rate >= 0.01 (1%) 인 풀은 절대 진입 금지
- TVL $10,000 미만 풀 진입 금지
- 가스비 7일 내 회수 불가능하면 이동 금지

## 요청
위 데이터를 분석하여 지금 어떤 행동을 해야 하는지 판단해주세요.
migrate를 추천할 경우 반드시 target_pool에 풀 주소를 포함하세요.
"""

    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
            ),
        )
        result = _parse_ai_response(response.text)
        logger.info("Gemini 판단: %s (신뢰도: %.0f%%) – %s",
                     result.get("decision"), result.get("confidence", 0) * 100,
                     result.get("reasoning", "")[:100])
        return result
    except Exception as e:
        logger.error("Gemini API 호출 실패: %s", e)
        return {
            "decision": "hold",
            "confidence": 0.0,
            "reasoning": f"Gemini API 오류: {e} – 안전을 위해 현재 포지션 유지",
            "risk_assessment": "high",
            "risk_factors": ["API 오류"],
            "market_insight": "",
        }


async def evaluate_migration_safety(
    from_pool: Optional[Any],
    to_pool: Any,
    gas_cost_usd: float,
) -> Dict[str, Any]:
    """
    특정 마이그레이션의 안전성을 Gemini가 최종 검토합니다.
    실제 트랜잭션 실행 직전에 호출됩니다.
    """
    from_info = _pool_score_to_dict(from_pool) if from_pool else None
    to_info = _pool_score_to_dict(to_pool)

    prompt = f"""## 마이그레이션 최종 검토

### 현재 풀
{json.dumps(from_info, indent=2, ensure_ascii=False) if from_info else "없음"}

### 이동 대상 풀
{json.dumps(to_info, indent=2, ensure_ascii=False)}

### 예상 가스비
${gas_cost_usd:.2f}

### 규칙 리마인더
- fee_rate >= 0.01 (1%) 인 풀은 절대 진입 금지
- 가스비 7일 내 회수 불가능하면 이동 금지

## 요청
이 마이그레이션을 실행해도 안전한지 최종 판단해주세요.
decision은 "migrate" (실행) 또는 "hold" (취소) 중 하나로 응답하세요.
"""

    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        result = _parse_ai_response(response.text)
        logger.info("Gemini 최종 검토: %s – %s",
                     result.get("decision"), result.get("reasoning", "")[:100])
        return result
    except Exception as e:
        logger.error("Gemini 최종 검토 실패: %s", e)
        return {
            "decision": "hold",
            "confidence": 0.0,
            "reasoning": f"최종 검토 API 오류: {e} – 안전을 위해 이동 취소",
            "risk_assessment": "high",
        }


def enforce_fee_rule(pool) -> bool:
    """
    fee 규칙 강제 적용. AI 판단과 무관하게 fee >= MAX_POOL_FEE 이면 차단.
    Returns True if pool is allowed, False if blocked.
    """
    fee_rate = getattr(pool, "fee_bps", 0)
    if fee_rate:
        # Algebra fee: hundredths of a bip (1e-6)
        # fee=500 → 0.0005 (0.05%), fee=3000 → 0.003 (0.3%)
        fee_decimal = fee_rate / 1_000_000
    else:
        fee_decimal = getattr(pool, "fee_percent", 0)
        if fee_decimal > 1:
            fee_decimal = fee_decimal / 100

    if fee_decimal >= MAX_POOL_FEE:
        logger.warning("🚫 풀 %s (fee=%.4f) 진입 차단: fee >= %.4f 규칙 위반",
                        getattr(pool, "symbol", "?"), fee_decimal, MAX_POOL_FEE)
        return False
    return True

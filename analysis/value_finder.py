"""
Анализатор ценных ставок (Value Bets)
Сравнивает прогнозы модели с коэффициентами букмекеров
Выявляет расхождения и рассчитывает процент проходимости
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from models.prediction_engine import MatchPrediction, PoissonModel
from utils.helpers import (
    odds_to_probability,
    probability_to_odds,
    calculate_value,
    kelly_criterion,
    format_probability,
    format_odds,
)
from config.settings import MODEL_PARAMS

logger = logging.getLogger(__name__)


class BetRecommendation(Enum):
    """Рекомендация по ставке"""
    STRONG_BET = "Сильная ставка"
    GOOD_BET = "Хорошая ставка"
    NEUTRAL = "Нейтрально"
    AVOID = "Избегать"


@dataclass
class ValueBet:
    """Информация о ценной ставке"""

    match: str
    bookmaker: str
    market: str
    bet_type: str
    line: Optional[float]
    bookmaker_odds: float
    model_probability: float
    bookmaker_implied_prob: float
    fair_odds: float
    value: float  # > 0 = ценная ставка
    kelly_fraction: float
    recommendation: BetRecommendation

    # ROI симуляция
    expected_roi: float = 0.0
    win_probability: float = 0.0
    loss_probability: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "match": self.match,
            "bookmaker": self.bookmaker,
            "market": self.market,
            "bet_type": self.bet_type,
            "line": self.line,
            "bookmaker_odds": round(self.bookmaker_odds, 2),
            "model_probability": round(self.model_probability, 4),
            "model_probability_pct": f"{self.model_probability * 100:.1f}%",
            "bookmaker_implied_prob": round(self.bookmaker_implied_prob, 4),
            "bookmaker_implied_pct": f"{self.bookmaker_implied_prob * 100:.1f}%",
            "fair_odds": round(self.fair_odds, 2),
            "value": round(self.value, 4),
            "value_pct": f"{self.value * 100:.1f}%",
            "kelly_fraction": round(self.kelly_fraction, 4),
            "recommendation": self.recommendation.value,
            "expected_roi": f"{self.expected_roi * 100:.1f}%",
        }


class ValueBetFinder:
    """
    Находит ценные ставки путем сравнения прогнозов модели
    с коэффициентами букмекеров
    """

    # Минимальный порог ценности для рекомендации
    MIN_VALUE_THRESHOLD = 0.05  # 5%
    STRONG_VALUE_THRESHOLD = 0.15  # 15%

    def analyze_match(
        self,
        prediction: MatchPrediction,
        bookmaker_odds: Dict,
    ) -> List[ValueBet]:
        """
        Анализ одного матча на наличие ценных ставок

        Args:
            prediction: прогноз модели
            bookmaker_odds: коэффициенты букмекеров
            {market: {bookmaker, odds, line}}
        """
        value_bets = []
        match_name = f"{prediction.home_team} - {prediction.away_team}"

        # Анализ каждого рынка
        markets_data = bookmaker_odds.get("markets", {})

        # Желтые карточки
        yc_bets = self._analyze_yellow_cards(
            prediction, markets_data, match_name
        )
        value_bets.extend(yc_bets)

        # Угловые
        corner_bets = self._analyze_corners(
            prediction, markets_data, match_name
        )
        value_bets.extend(corner_bets)

        # Пенальти
        penalty_bets = self._analyze_penalties(
            prediction, markets_data, match_name
        )
        value_bets.extend(penalty_bets)

        # Медицина
        medical_bets = self._analyze_medical(
            prediction, markets_data, match_name
        )
        value_bets.extend(medical_bets)

        return value_bets

    def _analyze_yellow_cards(
        self,
        prediction: MatchPrediction,
        markets_data: Dict,
        match_name: str,
    ) -> List[ValueBet]:
        """Анализ рынка желтых карточек"""
        bets = []
        lambda_ = prediction.yellow_cards_expected

        for bet_key, market_info in markets_data.items():
            if "yellow_cards" not in bet_key:
                continue

            line = market_info.get("line")
            odds = market_info.get("odds", 0)
            bookmaker = market_info.get("bookmaker", "unknown")

            if not odds or odds <= 1.0 or line is None:
                continue

            if "over" in bet_key:
                model_prob = PoissonModel.probability_over(lambda_, line)
                bet_type = f"ЖК Больше {line}"
            else:
                model_prob = PoissonModel.probability_under(lambda_, line)
                bet_type = f"ЖК Меньше {line}"

            vbet = self._create_value_bet(
                match_name, bookmaker, "Желтые карточки",
                bet_type, line, odds, model_prob
            )
            if vbet:
                bets.append(vbet)

        return bets

    def _analyze_corners(
        self,
        prediction: MatchPrediction,
        markets_data: Dict,
        match_name: str,
    ) -> List[ValueBet]:
        """Анализ рынка угловых"""
        bets = []
        lambda_ = prediction.corners_expected

        for bet_key, market_info in markets_data.items():
            if "corners" not in bet_key:
                continue

            line = market_info.get("line")
            odds = market_info.get("odds", 0)
            bookmaker = market_info.get("bookmaker", "unknown")

            if not odds or odds <= 1.0 or line is None:
                continue

            if "over" in bet_key:
                model_prob = PoissonModel.probability_over(lambda_, line)
                bet_type = f"Угловые Больше {line}"
            else:
                model_prob = PoissonModel.probability_under(lambda_, line)
                bet_type = f"Угловые Меньше {line}"

            vbet = self._create_value_bet(
                match_name, bookmaker, "Угловые",
                bet_type, line, odds, model_prob
            )
            if vbet:
                bets.append(vbet)

        return bets

    def _analyze_penalties(
        self,
        prediction: MatchPrediction,
        markets_data: Dict,
        match_name: str,
    ) -> List[ValueBet]:
        """Анализ рынка пенальти"""
        bets = []
        penalty_prob = prediction.penalty_probability

        for bet_key, market_info in markets_data.items():
            if "penalty" not in bet_key:
                continue

            odds = market_info.get("odds", 0)
            bookmaker = market_info.get("bookmaker", "unknown")

            if not odds or odds <= 1.0:
                continue

            if "yes" in bet_key:
                model_prob = penalty_prob
                bet_type = "Пенальти Да"
            else:
                model_prob = 1.0 - penalty_prob
                bet_type = "Пенальти Нет"

            vbet = self._create_value_bet(
                match_name, bookmaker, "Пенальти",
                bet_type, None, odds, model_prob
            )
            if vbet:
                bets.append(vbet)

        return bets

    def _analyze_medical(
        self,
        prediction: MatchPrediction,
        markets_data: Dict,
        match_name: str,
    ) -> List[ValueBet]:
        """Анализ рынка выхода медбригады"""
        bets = []

        # Если нет прямых данных от букмекеров, создаем из предсказания
        medical_markets = {
            k: v for k, v in markets_data.items() if "medical" in k
        }

        if not medical_markets:
            # Генерируем гипотетические данные для анализа
            lambda_ = prediction.medical_expected
            for line in [0.5, 1.5, 2.5]:
                over_prob = PoissonModel.probability_over(lambda_, line)
                fair_odds_over = 1.0 / max(over_prob, 0.001)

                # Типичная маржа букмекера ~5%
                implied_bk_odds_over = fair_odds_over * 0.95

                logger.debug(
                    f"Медицина линия {line}: "
                    f"ожидание {lambda_:.2f}, "
                    f"P(больше) = {over_prob:.3f}, "
                    f"справедливые коэф: {fair_odds_over:.2f}"
                )
        else:
            for bet_key, market_info in medical_markets.items():
                line = market_info.get("line")
                odds = market_info.get("odds", 0)
                bookmaker = market_info.get("bookmaker", "unknown")

                if not odds or odds <= 1.0 or line is None:
                    continue

                lambda_ = prediction.medical_expected
                if "over" in bet_key:
                    model_prob = PoissonModel.probability_over(lambda_, line)
                    bet_type = f"Медбригада Больше {line}"
                else:
                    model_prob = PoissonModel.probability_under(lambda_, line)
                    bet_type = f"Медбригада Меньше {line}"

                vbet = self._create_value_bet(
                    match_name, bookmaker, "Медицинская бригада",
                    bet_type, line, odds, model_prob
                )
                if vbet:
                    bets.append(vbet)

        return bets

    def _create_value_bet(
        self,
        match_name: str,
        bookmaker: str,
        market: str,
        bet_type: str,
        line: Optional[float],
        bookmaker_odds: float,
        model_probability: float,
    ) -> Optional[ValueBet]:
        """Создать объект ValueBet если есть ценность"""
        if bookmaker_odds <= 1.0 or model_probability <= 0:
            return None

        bk_prob = odds_to_probability(bookmaker_odds)
        fair_odds = probability_to_odds(model_probability)
        value = calculate_value(model_probability, bookmaker_odds)

        kelly = kelly_criterion(
            model_probability,
            bookmaker_odds,
            fraction=MODEL_PARAMS["kelly_fraction"],
        )

        # Определить рекомендацию
        if value >= self.STRONG_VALUE_THRESHOLD:
            recommendation = BetRecommendation.STRONG_BET
        elif value >= self.MIN_VALUE_THRESHOLD:
            recommendation = BetRecommendation.GOOD_BET
        elif value >= -0.02:
            recommendation = BetRecommendation.NEUTRAL
        else:
            recommendation = BetRecommendation.AVOID

        # Расчет ожидаемого ROI
        expected_roi = (model_probability * bookmaker_odds) - 1.0

        return ValueBet(
            match=match_name,
            bookmaker=bookmaker,
            market=market,
            bet_type=bet_type,
            line=line,
            bookmaker_odds=bookmaker_odds,
            model_probability=model_probability,
            bookmaker_implied_prob=bk_prob,
            fair_odds=fair_odds,
            value=value,
            kelly_fraction=kelly,
            recommendation=recommendation,
            expected_roi=expected_roi,
            win_probability=model_probability,
            loss_probability=1 - model_probability,
        )

    def analyze_all_matches(
        self,
        predictions: List[MatchPrediction],
        all_bookmaker_odds: Dict,
    ) -> pd.DataFrame:
        """
        Анализ всех матчей и сводная таблица ценных ставок
        """
        all_value_bets = []

        for prediction in predictions:
            match_key = f"{prediction.home_team} - {prediction.away_team}"
            bk_odds = all_bookmaker_odds.get(match_key, {})

            bets = self.analyze_match(prediction, bk_odds)
            all_value_bets.extend(bets)

        if not all_value_bets:
            return pd.DataFrame()

        df = pd.DataFrame([b.to_dict() for b in all_value_bets])

        # Фильтр только ценных ставок
        good_bets = df[
            df["recommendation"].isin([
                BetRecommendation.STRONG_BET.value,
                BetRecommendation.GOOD_BET.value,
            ])
        ].sort_values("value", ascending=False)

        return good_bets

    def generate_report(
        self,
        predictions: List[MatchPrediction],
        all_bookmaker_odds: Dict,
    ) -> Dict:
        """
        Генерация детального отчета по всем матчам
        """
        report = {
            "matches": [],
            "value_bets_summary": {},
            "market_analysis": {},
        }

        for prediction in predictions:
            match_key = f"{prediction.home_team} - {prediction.away_team}"
            bk_odds = all_bookmaker_odds.get(match_key, {})
            markets_data = bk_odds.get("markets", {})

            match_analysis = {
                "match": match_key,
                "referee": prediction.referee,
                "data_quality": prediction.data_quality,
                "confidence": f"{prediction.confidence * 100:.0f}%",
                "predictions": {
                    "yellow_cards": {
                        "expected": round(prediction.yellow_cards_expected, 2),
                        "std": round(prediction.yellow_cards_std, 2),
                        "lines": self._format_line_analysis(
                            prediction.yellow_cards_probs,
                            markets_data,
                            "yellow_cards",
                        ),
                    },
                    "corners": {
                        "expected": round(prediction.corners_expected, 2),
                        "std": round(prediction.corners_std, 2),
                        "lines": self._format_line_analysis(
                            prediction.corners_probs,
                            markets_data,
                            "corners",
                        ),
                    },
                    "penalty": {
                        "probability": f"{prediction.penalty_probability * 100:.1f}%",
                        "fair_odds_yes": round(
                            1 / max(prediction.penalty_probability, 0.001), 2
                        ),
                        "fair_odds_no": round(
                            1 / max(1 - prediction.penalty_probability, 0.001), 2
                        ),
                        "bookmaker_yes": markets_data.get(
                            "penalty_yes", {}
                        ).get("odds", "—"),
                        "bookmaker_no": markets_data.get(
                            "penalty_no", {}
                        ).get("odds", "—"),
                        "value_yes": self._calc_value_str(
                            prediction.penalty_probability,
                            markets_data.get("penalty_yes", {}).get("odds", 0),
                        ),
                        "value_no": self._calc_value_str(
                            1 - prediction.penalty_probability,
                            markets_data.get("penalty_no", {}).get("odds", 0),
                        ),
                    },
                    "medical": {
                        "expected": round(prediction.medical_expected, 2),
                        "lines": self._format_line_analysis(
                            prediction.medical_probs,
                            markets_data,
                            "medical",
                        ),
                    },
                },
                "factors_used": prediction.factors_used,
            }
            report["matches"].append(match_analysis)

        return report

    def _format_line_analysis(
        self,
        model_probs: Dict,
        markets_data: Dict,
        market_prefix: str,
    ) -> List[Dict]:
        """Форматирование анализа линий"""
        lines_analysis = []

        for line, probs in model_probs.items():
            over_key = f"{market_prefix}_over"
            under_key = f"{market_prefix}_under"

            bk_over_odds = None
            bk_under_odds = None

            for mk, md in markets_data.items():
                if over_key in mk and md.get("line") == line:
                    bk_over_odds = md.get("odds")
                elif under_key in mk and md.get("line") == line:
                    bk_under_odds = md.get("odds")

            over_value = None
            under_value = None

            if bk_over_odds and bk_over_odds > 1.0:
                over_value = calculate_value(probs["over"], bk_over_odds)
            if bk_under_odds and bk_under_odds > 1.0:
                under_value = calculate_value(probs["under"], bk_under_odds)

            lines_analysis.append({
                "line": line,
                "model_over_prob": f"{probs['over'] * 100:.1f}%",
                "model_under_prob": f"{probs['under'] * 100:.1f}%",
                "fair_over_odds": probs["over_odds"],
                "fair_under_odds": probs["under_odds"],
                "bk_over_odds": bk_over_odds or "—",
                "bk_under_odds": bk_under_odds or "—",
                "over_value": f"{over_value * 100:.1f}%" if over_value else "—",
                "under_value": f"{under_value * 100:.1f}%" if under_value else "—",
                "over_recommendation": self._value_to_recommendation(over_value),
                "under_recommendation": self._value_to_recommendation(under_value),
            })

        return lines_analysis

    def _calc_value_str(self, model_prob: float, bk_odds: float) -> str:
        if not bk_odds or bk_odds <= 1.0:
            return "—"
        value = calculate_value(model_prob, bk_odds)
        return f"{value * 100:+.1f}%"

    def _value_to_recommendation(self, value: Optional[float]) -> str:
        if value is None:
            return "—"
        if value >= self.STRONG_VALUE_THRESHOLD:
            return "СИЛЬНАЯ СТАВКА"
        elif value >= self.MIN_VALUE_THRESHOLD:
            return "Хорошая ставка"
        elif value >= -0.02:
            return "Нейтрально"
        else:
            return "Избегать"

"""
Движок предсказаний для ставок на РПЛ
Использует статистические модели для 4 рынков:
- Желтые карточки
- Угловые
- Выход медицинской бригады
- Пенальти
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import poisson
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from config.settings import MODEL_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class MatchPrediction:
    """Результат предсказания для одного матча"""

    home_team: str
    away_team: str
    referee: str = ""

    # Желтые карточки
    yellow_cards_expected: float = 0.0
    yellow_cards_std: float = 0.0
    yellow_cards_probs: Dict[str, float] = field(default_factory=dict)
    # {line: {"over": p, "under": p}}

    # Угловые
    corners_expected: float = 0.0
    corners_std: float = 0.0
    corners_probs: Dict[str, float] = field(default_factory=dict)

    # Пенальти
    penalty_probability: float = 0.0

    # Медицина
    medical_expected: float = 0.0
    medical_probs: Dict[str, float] = field(default_factory=dict)

    # Метаданные
    confidence: float = 0.0
    data_quality: str = "medium"
    factors_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "referee": self.referee,
            "yellow_cards": {
                "expected": round(self.yellow_cards_expected, 2),
                "std": round(self.yellow_cards_std, 2),
                "probabilities": self.yellow_cards_probs,
            },
            "corners": {
                "expected": round(self.corners_expected, 2),
                "std": round(self.corners_std, 2),
                "probabilities": self.corners_probs,
            },
            "penalty": {
                "probability": round(self.penalty_probability, 4),
                "odds": round(1 / max(self.penalty_probability, 0.001), 2),
            },
            "medical": {
                "expected": round(self.medical_expected, 2),
                "probabilities": self.medical_probs,
            },
            "confidence": round(self.confidence, 3),
            "data_quality": self.data_quality,
            "factors_used": self.factors_used,
        }


class PoissonModel:
    """Модель Пуассона для подсчетных событий"""

    @staticmethod
    def probability_over(lambda_: float, line: float) -> float:
        """P(X > line) для распределения Пуассона"""
        k = int(np.floor(line))
        return 1 - poisson.cdf(k, lambda_)

    @staticmethod
    def probability_under(lambda_: float, line: float) -> float:
        """P(X < line) для распределения Пуассона"""
        k = int(np.ceil(line)) - 1
        if k < 0:
            return 0.0
        return poisson.cdf(k, lambda_)

    @staticmethod
    def probability_exact(lambda_: float, k: int) -> float:
        """P(X = k) для распределения Пуассона"""
        return poisson.pmf(k, lambda_)

    @staticmethod
    def calculate_line_probs(
        lambda_: float, lines: List[float]
    ) -> Dict[float, Dict[str, float]]:
        """Вычислить вероятности для нескольких линий"""
        result = {}
        for line in lines:
            over_p = PoissonModel.probability_over(lambda_, line)
            under_p = PoissonModel.probability_under(lambda_, line)
            result[line] = {
                "over": round(over_p, 4),
                "under": round(under_p, 4),
                "over_odds": round(1 / max(over_p, 0.001), 3),
                "under_odds": round(1 / max(under_p, 0.001), 3),
            }
        return result


class YellowCardsModel:
    """
    Модель предсказания желтых карточек

    Факторы:
    1. Средние показатели команд (агрессивность)
    2. Статистика судьи (строгость)
    3. Очные встречи (H2H)
    4. Домашнее/гостевое преимущество
    5. Важность матча (позиция в таблице)
    """

    # Средние значения РПЛ (база)
    RPL_AVG_YELLOW = 3.4
    RPL_HOME_YELLOW = 1.62
    RPL_AWAY_YELLOW = 1.78

    def predict(
        self,
        home_team_stats: Dict,
        away_team_stats: Dict,
        referee_stats: Optional[Dict] = None,
        h2h_stats: Optional[Dict] = None,
        weights: Optional[Dict] = None,
    ) -> Tuple[float, float, Dict]:
        """
        Предсказать количество желтых карточек

        Returns: (expected_total, std, probabilities_dict)
        """
        if weights is None:
            weights = MODEL_PARAMS["weights"]

        factors_contributions = []

        # 1. Базовые показатели команд
        home_yellow_avg = home_team_stats.get(
            "avg_yellow_cards_home", self.RPL_HOME_YELLOW
        )
        away_yellow_avg = away_team_stats.get(
            "avg_yellow_cards_away", self.RPL_AWAY_YELLOW
        )
        team_expected = home_yellow_avg + away_yellow_avg
        factors_contributions.append(("team_form", team_expected))

        # 2. Корректировка на судью
        referee_multiplier = 1.0
        if referee_stats:
            referee_multiplier = referee_stats.get("strictness_index", 1.0)
            # Сглаживание - не доверяем слишком экстремальным значениям
            referee_multiplier = 0.7 + 0.3 * referee_multiplier

        # 3. H2H
        h2h_expected = team_expected
        if h2h_stats and h2h_stats.get("matches_count", 0) >= 3:
            h2h_yellow = h2h_stats.get("avg_total_yellow", team_expected)
            # Взвешенное среднее: 80% текущая форма, 20% H2H
            h2h_expected = 0.8 * team_expected + 0.2 * h2h_yellow

        # Финальное ожидание
        expected = h2h_expected * referee_multiplier

        # Стандартное отклонение (эмпирически для РПЛ ~1.5)
        std = 1.5 + 0.1 * expected

        # Вычислить вероятности для стандартных линий
        standard_lines = [2.5, 3.5, 4.5, 5.5]
        probs = PoissonModel.calculate_line_probs(expected, standard_lines)

        logger.debug(
            f"Желтые карточки: {home_team_stats.get('team')} vs "
            f"{away_team_stats.get('team')}: "
            f"ожидается {expected:.2f} (судья x{referee_multiplier:.2f})"
        )

        return expected, std, probs


class CornersModel:
    """
    Модель предсказания угловых

    Факторы:
    1. Стиль игры команд (атакующий/оборонительный)
    2. Домашнее преимущество (хозяева бьют больше угловых)
    3. Качество атаки vs обороны соперника
    4. Исторические данные матча
    """

    RPL_AVG_CORNERS = 9.8
    RPL_HOME_CORNERS = 5.1
    RPL_AWAY_CORNERS = 4.7

    def predict(
        self,
        home_team_stats: Dict,
        away_team_stats: Dict,
        h2h_stats: Optional[Dict] = None,
    ) -> Tuple[float, float, Dict]:
        """
        Предсказать количество угловых

        Returns: (expected_total, std, probabilities_dict)
        """
        # Атакующая сила команды влияет на угловые
        home_corners_avg = home_team_stats.get(
            "avg_corners_home", self.RPL_HOME_CORNERS
        )
        away_corners_avg = away_team_stats.get(
            "avg_corners_away", self.RPL_AWAY_CORNERS
        )

        # Корректировка на оборону соперника
        # Слабая оборона -> больше угловых у атакующей команды
        home_attack_adj = home_team_stats.get("avg_goals", 1.5) / 1.5
        away_defense_adj = away_team_stats.get("avg_goals_conceded", 1.1) / 1.1

        home_corners_pred = home_corners_avg * (0.7 + 0.3 * home_attack_adj)

        away_attack_adj = away_team_stats.get("avg_goals", 1.1) / 1.1
        home_defense_adj = home_team_stats.get("avg_goals_conceded", 1.1) / 1.1

        away_corners_pred = away_corners_avg * (0.7 + 0.3 * away_attack_adj)

        expected = home_corners_pred + away_corners_pred

        # H2H корректировка
        if h2h_stats and h2h_stats.get("matches_count", 0) >= 3:
            h2h_corners = h2h_stats.get("avg_total_corners", expected)
            expected = 0.75 * expected + 0.25 * h2h_corners

        # Стандартное отклонение (эмпирически ~3.0)
        std = 3.0

        # Вероятности для стандартных линий
        standard_lines = [7.5, 8.5, 9.5, 10.5, 11.5]
        probs = PoissonModel.calculate_line_probs(expected, standard_lines)

        logger.debug(
            f"Угловые: ожидается {expected:.2f} "
            f"(хозяева: {home_corners_pred:.1f}, гости: {away_corners_pred:.1f})"
        )

        return expected, std, probs


class PenaltyModel:
    """
    Модель предсказания пенальти

    Факторы:
    1. Частота пенальти в матчах каждой команды
    2. Склонность судьи назначать пенальти
    3. Атакующий стиль (больше атак -> больше нарушений в штрафной)
    4. Важность матча
    """

    # Базовая вероятность пенальти в РПЛ
    RPL_PENALTY_RATE = 0.28

    def predict(
        self,
        home_team_stats: Dict,
        away_team_stats: Dict,
        referee_stats: Optional[Dict] = None,
        h2h_stats: Optional[Dict] = None,
    ) -> float:
        """
        Предсказать вероятность пенальти

        Returns: probability (0-1)
        """
        # Базовая вероятность из статистики команд
        home_penalty_rate = home_team_stats.get(
            "penalty_rate", self.RPL_PENALTY_RATE
        )
        away_penalty_rate = away_team_stats.get(
            "penalty_rate", self.RPL_PENALTY_RATE
        )

        # Объединенная вероятность (хотя бы один получит пенальти)
        team_prob = 1 - (1 - home_penalty_rate) * (1 - away_penalty_rate)

        # Корректировка на судью
        if referee_stats:
            penalty_index = referee_stats.get("penalty_index", 1.0)
            # Сглаживание
            penalty_index = 0.6 + 0.4 * penalty_index
            team_prob *= penalty_index

        # H2H корректировка
        if h2h_stats and h2h_stats.get("matches_count", 0) >= 5:
            h2h_penalty_rate = h2h_stats.get("penalty_rate", team_prob)
            team_prob = 0.7 * team_prob + 0.3 * h2h_penalty_rate

        # Ограничение в разумных пределах
        probability = min(max(team_prob, 0.05), 0.70)

        logger.debug(f"Пенальти: вероятность {probability:.3f} ({probability*100:.1f}%)")

        return probability


class MedicalExitsModel:
    """
    Модель предсказания выхода медицинской бригады
    Выход медбригады = замена из-за травмы или серьезного нарушения

    Факторы:
    1. Физический стиль игры команд
    2. Количество нарушений (грубость)
    3. Усталость команд (плотный календарь)
    4. Качество покрытия поля
    5. Статистика судьи
    """

    # Базовое среднее для РПЛ
    RPL_AVG_MEDICAL = 1.8

    def predict(
        self,
        home_team_stats: Dict,
        away_team_stats: Dict,
        referee_stats: Optional[Dict] = None,
    ) -> Tuple[float, Dict]:
        """
        Предсказать количество выходов медбригады

        Returns: (expected, probabilities_dict)
        """
        # Базовые показатели команд
        home_medical = home_team_stats.get("avg_medical", self.RPL_AVG_MEDICAL / 2)
        away_medical = away_team_stats.get("avg_medical", self.RPL_AVG_MEDICAL / 2)

        # Влияние нарушений
        home_foul_factor = home_team_stats.get("avg_fouls", 12.5) / 12.5
        away_foul_factor = away_team_stats.get("avg_fouls", 13.2) / 13.2

        # Корректировка на строгость судьи
        # Строгий судья -> больше нарушений -> больше травм
        referee_factor = 1.0
        if referee_stats:
            strictness = referee_stats.get("strictness_index", 1.0)
            referee_factor = 0.85 + 0.15 * strictness

        expected = (
            (home_medical * (0.8 + 0.2 * home_foul_factor))
            + (away_medical * (0.8 + 0.2 * away_foul_factor))
        ) * referee_factor

        # Если нет данных - использовать базовое значение
        if expected <= 0:
            expected = self.RPL_AVG_MEDICAL

        # Стандартные линии
        standard_lines = [0.5, 1.5, 2.5, 3.5]
        probs = PoissonModel.calculate_line_probs(expected, standard_lines)

        logger.debug(f"Медицина: ожидается {expected:.2f} выхода(ов)")

        return expected, probs


class PredictionEngine:
    """
    Главный движок предсказаний
    Интегрирует все модели и возвращает комплексный прогноз
    """

    def __init__(self):
        self.yellow_model = YellowCardsModel()
        self.corners_model = CornersModel()
        self.penalty_model = PenaltyModel()
        self.medical_model = MedicalExitsModel()

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        team_stats_df: pd.DataFrame,
        referee_stats_df: pd.DataFrame,
        matches_df: pd.DataFrame,
        referee: str = "",
    ) -> MatchPrediction:
        """
        Создать полный прогноз для матча

        Args:
            home_team: название команды хозяев
            away_team: название команды гостей
            team_stats_df: статистика команд
            referee_stats_df: статистика судей
            matches_df: исторические данные матчей
            referee: имя судьи
        """
        prediction = MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            referee=referee,
        )

        factors_used = []

        # Получить статистику команд
        home_stats = self._get_team_stats(home_team, team_stats_df)
        away_stats = self._get_team_stats(away_team, team_stats_df)

        if home_stats:
            factors_used.append("home_team_stats")
        if away_stats:
            factors_used.append("away_team_stats")

        # Статистика судьи
        referee_stats = self._get_referee_stats(referee, referee_stats_df)
        if referee_stats:
            factors_used.append("referee_stats")

        # H2H статистика
        h2h_stats = self._get_h2h_stats(home_team, away_team, matches_df)
        if h2h_stats.get("matches_count", 0) > 0:
            factors_used.append("h2h_history")

        # Прогноз желтых карточек
        yc_exp, yc_std, yc_probs = self.yellow_model.predict(
            home_stats, away_stats, referee_stats, h2h_stats
        )
        prediction.yellow_cards_expected = yc_exp
        prediction.yellow_cards_std = yc_std
        prediction.yellow_cards_probs = yc_probs

        # Прогноз угловых
        c_exp, c_std, c_probs = self.corners_model.predict(
            home_stats, away_stats, h2h_stats
        )
        prediction.corners_expected = c_exp
        prediction.corners_std = c_std
        prediction.corners_probs = c_probs

        # Прогноз пенальти
        penalty_prob = self.penalty_model.predict(
            home_stats, away_stats, referee_stats, h2h_stats
        )
        prediction.penalty_probability = penalty_prob

        # Прогноз медицины
        med_exp, med_probs = self.medical_model.predict(
            home_stats, away_stats, referee_stats
        )
        prediction.medical_expected = med_exp
        prediction.medical_probs = med_probs

        # Оценка уверенности
        n_factors = len(factors_used)
        home_matches = home_stats.get("matches_analyzed", 0)
        away_matches = away_stats.get("matches_analyzed", 0)
        min_matches = min(home_matches, away_matches)

        confidence = min(
            0.9,
            0.4 + (n_factors * 0.1) + (min_matches / 100),
        )
        prediction.confidence = confidence

        if n_factors >= 4 and min_matches >= 15:
            prediction.data_quality = "high"
        elif n_factors >= 2 and min_matches >= 8:
            prediction.data_quality = "medium"
        else:
            prediction.data_quality = "low"

        prediction.factors_used = factors_used

        return prediction

    def _get_team_stats(self, team: str, stats_df: pd.DataFrame) -> Dict:
        """Получить статистику команды"""
        if stats_df.empty or "team" not in stats_df.columns:
            return self._default_team_stats(team)

        row = stats_df[stats_df["team"] == team]
        if row.empty:
            # Поиск частичного совпадения
            for _, r in stats_df.iterrows():
                if team.lower() in str(r.get("team", "")).lower():
                    row = pd.DataFrame([r])
                    break

        if row.empty:
            return self._default_team_stats(team)

        return row.iloc[0].to_dict()

    def _default_team_stats(self, team: str) -> Dict:
        """Дефолтные значения для команды без данных"""
        return {
            "team": team,
            "matches_analyzed": 0,
            "avg_yellow_cards": 1.7,
            "avg_yellow_cards_home": 1.62,
            "avg_yellow_cards_away": 1.78,
            "avg_corners": 4.9,
            "avg_corners_home": 5.1,
            "avg_corners_away": 4.7,
            "avg_fouls": 12.8,
            "penalty_rate": 0.28,
            "avg_medical": 0.9,
            "avg_goals": 1.3,
            "avg_goals_conceded": 1.1,
        }

    def _get_referee_stats(
        self, referee: str, referee_stats_df: pd.DataFrame
    ) -> Dict:
        """Получить статистику судьи"""
        if not referee or referee_stats_df.empty:
            return {}

        row = referee_stats_df[referee_stats_df["referee"] == referee]
        if row.empty:
            return {}

        return row.iloc[0].to_dict()

    def _get_h2h_stats(
        self, home_team: str, away_team: str, matches_df: pd.DataFrame
    ) -> Dict:
        """Получить статистику H2H"""
        if matches_df.empty:
            return {}

        h2h = matches_df[
            (
                (matches_df["home_team"] == home_team)
                & (matches_df["away_team"] == away_team)
            )
            | (
                (matches_df["home_team"] == away_team)
                & (matches_df["away_team"] == home_team)
            )
        ]

        if h2h.empty:
            return {}

        result = {
            "matches_count": len(h2h),
        }

        if "total_yellow_cards" in h2h.columns:
            result["avg_total_yellow"] = h2h["total_yellow_cards"].mean()
        if "total_corners" in h2h.columns:
            result["avg_total_corners"] = h2h["total_corners"].mean()
        if "has_penalty" in h2h.columns:
            result["penalty_rate"] = h2h["has_penalty"].mean()
        if "medical_exits" in h2h.columns:
            result["avg_medical"] = h2h["medical_exits"].mean()

        return result

    def batch_predict(
        self,
        upcoming_matches: List[Dict],
        team_stats_df: pd.DataFrame,
        referee_stats_df: pd.DataFrame,
        matches_df: pd.DataFrame,
    ) -> List[MatchPrediction]:
        """Пакетное предсказание для нескольких матчей"""
        predictions = []

        for match in upcoming_matches:
            home = match.get("home_team", "")
            away = match.get("away_team", "")
            referee = match.get("referee", "")

            if not home or not away:
                continue

            prediction = self.predict_match(
                home_team=home,
                away_team=away,
                team_stats_df=team_stats_df,
                referee_stats_df=referee_stats_df,
                matches_df=matches_df,
                referee=referee,
            )
            predictions.append(prediction)

        return predictions

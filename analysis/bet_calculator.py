"""
Калькулятор ставок
Расчет оптимального размера ставки, ROI, вероятности прохода
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from utils.helpers import kelly_criterion, odds_to_probability


@dataclass
class BetCalculation:
    """Результат расчета ставки"""

    # Входные данные
    bet_amount: float
    odds: float
    model_probability: float
    bankroll: float

    # Результаты
    potential_win: float
    potential_profit: float
    expected_value: float
    roi_if_win: float
    roi_if_loss: float
    expected_roi: float

    # Оптимальный размер
    kelly_bet: float
    quarter_kelly_bet: float
    max_recommended_bet: float

    # Вероятности
    pass_probability: float  # = model_probability
    fail_probability: float

    # Анализ риска
    risk_level: str
    breakeven_probability: float  # минимальная вероятность для безубыточности

    def to_dict(self) -> Dict:
        return {
            "Сумма ставки": f"{self.bet_amount:.2f} ₽",
            "Коэффициент": f"{self.odds:.2f}",
            "Потенциальный выигрыш": f"{self.potential_win:.2f} ₽",
            "Чистая прибыль": f"{self.potential_profit:.2f} ₽",
            "Ожидаемое значение": f"{self.expected_value:.2f} ₽",
            "ROI при победе": f"{self.roi_if_win:.1f}%",
            "ROI при проигрыше": f"{self.roi_if_loss:.1f}%",
            "Ожидаемый ROI": f"{self.expected_roi:.1f}%",
            "Вероятность прохода": f"{self.pass_probability * 100:.1f}%",
            "Вероятность провала": f"{self.fail_probability * 100:.1f}%",
            "Ставка Келли (полная)": f"{self.kelly_bet:.2f} ₽",
            "Ставка Келли (1/4)": f"{self.quarter_kelly_bet:.2f} ₽",
            "Макс. рекомендованная ставка": f"{self.max_recommended_bet:.2f} ₽",
            "Точка безубыточности": f"{self.breakeven_probability * 100:.1f}%",
            "Уровень риска": self.risk_level,
        }


class BetCalculator:
    """Калькулятор параметров ставки"""

    MAX_BET_FRACTION = 0.05  # максимум 5% от банкролла

    def calculate(
        self,
        bet_amount: float,
        odds: float,
        model_probability: float,
        bankroll: float = 10000.0,
    ) -> BetCalculation:
        """
        Расчет всех параметров ставки

        Args:
            bet_amount: сумма ставки
            odds: коэффициент букмекера
            model_probability: вероятность победы по модели
            bankroll: общий банкролл
        """
        # Базовые расчеты
        potential_win = bet_amount * odds
        potential_profit = potential_win - bet_amount
        expected_value = (model_probability * potential_win) - bet_amount
        roi_if_win = (potential_profit / bet_amount) * 100
        roi_if_loss = -100.0
        expected_roi = (expected_value / bet_amount) * 100

        # Критерий Келли
        kelly_full = kelly_criterion(model_probability, odds, fraction=1.0) * bankroll
        kelly_quarter = kelly_criterion(model_probability, odds, fraction=0.25) * bankroll
        max_recommended = bankroll * self.MAX_BET_FRACTION

        # Точка безубыточности
        breakeven_prob = 1.0 / odds if odds > 1.0 else 1.0

        # Уровень риска
        bet_fraction = bet_amount / bankroll if bankroll > 0 else 0
        if bet_fraction <= 0.01:
            risk_level = "Очень низкий"
        elif bet_fraction <= 0.02:
            risk_level = "Низкий"
        elif bet_fraction <= 0.05:
            risk_level = "Средний"
        elif bet_fraction <= 0.10:
            risk_level = "Высокий"
        else:
            risk_level = "Очень высокий"

        return BetCalculation(
            bet_amount=bet_amount,
            odds=odds,
            model_probability=model_probability,
            bankroll=bankroll,
            potential_win=potential_win,
            potential_profit=potential_profit,
            expected_value=expected_value,
            roi_if_win=roi_if_win,
            roi_if_loss=roi_if_loss,
            expected_roi=expected_roi,
            kelly_bet=kelly_full,
            quarter_kelly_bet=kelly_quarter,
            max_recommended_bet=min(kelly_quarter, max_recommended),
            pass_probability=model_probability,
            fail_probability=1.0 - model_probability,
            risk_level=risk_level,
            breakeven_probability=breakeven_prob,
        )

    def simulate_bankroll(
        self,
        bet_amount: float,
        odds: float,
        model_probability: float,
        n_bets: int = 100,
        bankroll: float = 10000.0,
        n_simulations: int = 1000,
    ) -> Dict:
        """
        Симуляция Монте-Карло для оценки роста банкролла

        Args:
            n_bets: количество ставок
            n_simulations: количество симуляций
        """
        np.random.seed(42)
        final_bankrolls = []

        for _ in range(n_simulations):
            current_bankroll = bankroll
            history = [bankroll]

            for _ in range(n_bets):
                if current_bankroll <= 0:
                    break

                actual_bet = min(bet_amount, current_bankroll)
                win = np.random.random() < model_probability

                if win:
                    current_bankroll += actual_bet * (odds - 1)
                else:
                    current_bankroll -= actual_bet

                history.append(current_bankroll)

            final_bankrolls.append(current_bankroll)

        final_bankrolls = np.array(final_bankrolls)

        return {
            "initial_bankroll": bankroll,
            "mean_final": float(np.mean(final_bankrolls)),
            "median_final": float(np.median(final_bankrolls)),
            "std_final": float(np.std(final_bankrolls)),
            "p10_final": float(np.percentile(final_bankrolls, 10)),
            "p90_final": float(np.percentile(final_bankrolls, 90)),
            "prob_profit": float(np.mean(final_bankrolls > bankroll)),
            "prob_ruin": float(np.mean(final_bankrolls <= 0)),
            "mean_roi": float(
                (np.mean(final_bankrolls) - bankroll) / bankroll * 100
            ),
            "n_bets": n_bets,
            "n_simulations": n_simulations,
        }

    def calculate_series(
        self,
        bets: List[Dict],
        bankroll: float = 10000.0,
    ) -> pd.DataFrame:
        """
        Расчет для серии ставок
        Each bet: {odds, probability, amount}
        """
        results = []
        current_bankroll = bankroll

        for i, bet in enumerate(bets):
            odds = bet.get("odds", 1.0)
            prob = bet.get("probability", 0.5)
            amount = bet.get("amount", 100.0)

            calc = self.calculate(amount, odds, prob, current_bankroll)
            results.append({
                "bet_number": i + 1,
                **calc.to_dict(),
                "bankroll_before": current_bankroll,
                "bankroll_after_win": current_bankroll + calc.potential_profit,
                "bankroll_after_loss": current_bankroll - amount,
            })

        return pd.DataFrame(results)

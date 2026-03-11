"""
Главный модуль - предиктивная модель ставок РПЛ
Запуск: python main.py [--update] [--report] [--dashboard]
"""
import argparse
import json
import logging
import sys
import schedule
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import PROCESSED_DIR
from models.prediction_engine import PredictionEngine
from analysis.value_finder import ValueBetFinder
from analysis.bet_calculator import BetCalculator
from scrapers.statistics import (
    RPLStatisticsCollector,
    RefereeStatsCollector,
    TeamStatsCollector,
)
from scrapers.bookmakers import OddsAggregator
from utils.helpers import setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


class RPLBettingModel:
    """
    Главный класс предиктивной модели ставок РПЛ
    """

    def __init__(self):
        self.engine = PredictionEngine()
        self.value_finder = ValueBetFinder()
        self.calculator = BetCalculator()
        self.stats_collector = RPLStatisticsCollector()
        self.referee_collector = RefereeStatsCollector()
        self.team_collector = TeamStatsCollector()
        self.odds_aggregator = OddsAggregator()

        self.matches_df = pd.DataFrame()
        self.referee_stats_df = pd.DataFrame()
        self.team_stats_df = pd.DataFrame()

    def update_data(self) -> None:
        """Обновить все данные"""
        logger.info("=== Обновление данных ===")

        # 1. Исторические данные
        logger.info("Загрузка исторических данных...")
        self.matches_df = self.stats_collector.collect_historical_data()
        logger.info(f"Матчей в базе: {len(self.matches_df)}")

        # 2. Статистика команд
        logger.info("Расчет статистики команд...")
        self.team_stats_df = self.team_collector.get_team_stats(self.matches_df)
        logger.info(f"Команд в базе: {len(self.team_stats_df)}")

        # 3. Статистика судей
        logger.info("Расчет статистики судей...")
        self.referee_stats_df = self.referee_collector.get_referee_stats(
            self.matches_df
        )
        logger.info(f"Судей в базе: {len(self.referee_stats_df)}")

    def analyze_match(
        self,
        home_team: str,
        away_team: str,
        referee: str = "",
        bankroll: float = 10000.0,
        bet_amount: float = 500.0,
    ) -> dict:
        """
        Полный анализ матча

        Args:
            home_team: команда хозяев
            away_team: команда гостей
            referee: судья
            bankroll: банкролл
            bet_amount: сумма ставки

        Returns:
            Словарь с прогнозами и рекомендациями
        """
        if self.matches_df.empty:
            self.update_data()

        logger.info(f"Анализ: {home_team} vs {away_team} (судья: {referee or 'не указан'})")

        # Прогноз
        prediction = self.engine.predict_match(
            home_team=home_team,
            away_team=away_team,
            team_stats_df=self.team_stats_df,
            referee_stats_df=self.referee_stats_df,
            matches_df=self.matches_df,
            referee=referee,
        )

        # Получить коэффициенты
        all_odds = self.odds_aggregator.get_all_odds()
        best_odds = self.odds_aggregator.get_best_odds(all_odds)
        match_key = f"{home_team} - {away_team}"
        bk_odds = best_odds.get(match_key, {})

        # Найти ценные ставки
        value_bets = self.value_finder.analyze_match(prediction, bk_odds)

        # Результат
        result = {
            "match": {
                "home_team": home_team,
                "away_team": away_team,
                "referee": referee,
                "analyzed_at": datetime.now().isoformat(),
            },
            "predictions": prediction.to_dict(),
            "value_bets": [b.to_dict() for b in value_bets],
            "recommended_bets": [
                b.to_dict() for b in value_bets
                if b.value >= 0.05  # только если Value > 5%
            ],
        }

        # Добавить расчет ставки для рекомендованных
        if result["recommended_bets"]:
            for bet in result["recommended_bets"]:
                odds = bet.get("bookmaker_odds", 0)
                prob = bet.get("model_probability", 0)
                if odds > 1.0 and prob > 0:
                    calc = self.calculator.calculate(
                        bet_amount, odds, prob, bankroll
                    )
                    bet["bet_calculation"] = calc.to_dict()

        return result

    def generate_daily_report(
        self, upcoming_matches: list, output_path: Path = None
    ) -> dict:
        """
        Генерация ежедневного отчета по предстоящим матчам
        """
        if self.matches_df.empty:
            self.update_data()

        logger.info(f"Генерация отчета для {len(upcoming_matches)} матчей")

        # Прогнозы
        predictions = self.engine.batch_predict(
            upcoming_matches,
            self.team_stats_df,
            self.referee_stats_df,
            self.matches_df,
        )

        # Коэффициенты
        all_odds = self.odds_aggregator.get_all_odds()
        best_odds = self.odds_aggregator.get_best_odds(all_odds)

        # Полный отчет
        report = self.value_finder.generate_report(predictions, best_odds)
        report["generated_at"] = datetime.now().isoformat()
        report["matches_count"] = len(upcoming_matches)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"Отчет сохранен: {output_path}")

        return report

    def run_auto_update(self, interval_minutes: int = 30) -> None:
        """Запуск автообновления"""
        logger.info(f"Запуск автообновления (каждые {interval_minutes} мин)")
        self.update_data()

        schedule.every(interval_minutes).minutes.do(self._scheduled_update)

        while True:
            schedule.run_pending()
            time.sleep(60)

    def _scheduled_update(self) -> None:
        """Плановое обновление данных"""
        logger.info("Плановое обновление данных...")
        try:
            self.update_data()
            logger.info("Обновление завершено")
        except Exception as e:
            logger.error(f"Ошибка при обновлении: {e}")


def print_report(report: dict) -> None:
    """Красивый вывод отчета в консоль"""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    console.print(
        Panel(
            "[bold blue]⚽ Предиктивная модель ставок РПЛ[/bold blue]\n"
            f"Время: {report.get('generated_at', 'N/A')}",
            expand=False,
        )
    )

    for match in report.get("matches", []):
        preds = match.get("predictions", {})

        console.print(f"\n[bold]🏟️ {match['match']}[/bold]")
        if match.get("referee"):
            console.print(f"Судья: {match['referee']}")
        console.print(
            f"Уверенность: {match.get('confidence', 'N/A')} | "
            f"Качество данных: {match.get('data_quality', 'N/A')}"
        )

        # Таблица прогнозов
        table = Table(title="Прогнозы")
        table.add_column("Рынок")
        table.add_column("Ожидание")
        table.add_column("Строки")

        yc = preds.get("yellow_cards", {})
        table.add_row(
            "Желтые карточки",
            f"{yc.get('expected', 0):.2f} ± {yc.get('std', 0):.2f}",
            " | ".join([
                f"{'Б' if p['over_recommendation'] != 'Избегать' else ''}3.5: "
                f"{p['fair_over_odds']:.2f}"
                for p in yc.get("lines", [])[:2]
            ]),
        )

        corners = preds.get("corners", {})
        table.add_row(
            "Угловые",
            f"{corners.get('expected', 0):.2f}",
            " | ".join([
                f"9.5: {p['fair_over_odds']:.2f}/{p['fair_under_odds']:.2f}"
                for p in corners.get("lines", [])[:1]
            ]),
        )

        penalty = preds.get("penalty", {})
        table.add_row(
            "Пенальти",
            f"Да: {penalty.get('probability', 'N/A')}",
            f"Да: {penalty.get('fair_odds_yes', 0):.2f} | "
            f"Нет: {penalty.get('fair_odds_no', 0):.2f}",
        )

        console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Предиктивная модель ставок РПЛ"
    )
    parser.add_argument("--update", action="store_true", help="Обновить данные")
    parser.add_argument("--report", action="store_true", help="Генерировать отчет")
    parser.add_argument("--dashboard", action="store_true", help="Запустить дашборд")
    parser.add_argument("--auto-update", action="store_true", help="Автообновление")
    parser.add_argument("--match", nargs="+", help="Анализ конкретного матча")
    parser.add_argument(
        "--bankroll", type=float, default=10000.0, help="Размер банкролла"
    )
    parser.add_argument(
        "--bet", type=float, default=500.0, help="Размер ставки"
    )

    args = parser.parse_args()

    if args.dashboard:
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "dashboard/app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
        ])
        return

    model = RPLBettingModel()

    if args.update or not (PROCESSED_DIR / "rpl_historical.csv").exists():
        model.update_data()

    if args.match:
        if len(args.match) >= 2:
            home, away = args.match[0], args.match[1]
            referee = args.match[2] if len(args.match) > 2 else ""
            result = model.analyze_match(
                home, away, referee, args.bankroll, args.bet
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("Укажите: --match 'Команда1' 'Команда2' ['Судья']")
        return

    if args.report:
        upcoming = [
            {"home_team": "Зенит", "away_team": "ЦСКА", "referee": "Казарцев С."},
            {"home_team": "Спартак", "away_team": "Краснодар", "referee": "Лапочкин С."},
            {"home_team": "Локомотив", "away_team": "Динамо", "referee": "Вилков В."},
        ]
        report_path = PROCESSED_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        report = model.generate_daily_report(upcoming, report_path)
        print_report(report)
        return

    if args.auto_update:
        model.run_auto_update(interval_minutes=30)
        return

    # По умолчанию - интерактивный режим
    print("\n=== Предиктивная модель ставок РПЛ ===")
    print("Команды: --match 'Зенит' 'ЦСКА'")
    print("Отчет:   --report")
    print("Дашборд: --dashboard")
    print("Помощь:  --help")

    # Демо запуск
    print("\n--- Демо анализ ---")
    model.update_data()
    result = model.analyze_match(
        "Зенит", "ЦСКА", "Казарцев С.", args.bankroll, args.bet
    )

    print(f"\nМатч: {result['match']['home_team']} vs {result['match']['away_team']}")
    preds = result["predictions"]
    print(f"Желтые карточки: {preds['yellow_cards']['expected']:.2f} (ожидается)")
    print(f"Угловые: {preds['corners']['expected']:.2f} (ожидается)")
    print(f"Пенальти: {preds['penalty']['probability']} вероятность")
    print(f"Медбригада: {preds['medical']['expected']:.2f} выхода(ов)")

    if result["recommended_bets"]:
        print(f"\nРекомендованных ставок: {len(result['recommended_bets'])}")
        for bet in result["recommended_bets"]:
            print(
                f"  - {bet['bet_type']}: коэф {bet['bookmaker_odds']:.2f}, "
                f"Value {bet['value_pct']}, {bet['recommendation']}"
            )
    else:
        print("\nЦенных ставок не найдено (нет данных от букмекеров)")


if __name__ == "__main__":
    main()

"""
Сборщик статистических данных для РПЛ
Источники: FBref, Understat, Football-Data.co.uk, Transfermarkt, Smart Tables
"""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup

from utils.helpers import safe_request, cache_data, load_cache, get_random_headers
from config.settings import CACHE_DIR, PROCESSED_DIR, SEASONS, REFEREES_2025_26

logger = logging.getLogger(__name__)


class RPLStatisticsCollector:
    """Сборщик статистики РПЛ"""

    # Football-Data.co.uk - бесплатные CSV данные
    FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/R1.csv"

    # FBref URLs
    FBREF_BASE = "https://fbref.com"
    FBREF_RPL = "https://fbref.com/en/comps/30/Russian-Premier-League-Stats"

    def __init__(self):
        self.seasons_data: Dict[str, pd.DataFrame] = {}

    def collect_historical_data(
        self, seasons: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Собрать исторические данные матчей РПЛ
        Включает: желтые карточки, угловые, пенальти, судьи
        Источники: Football-Data.co.uk, Smart Tables (smart-tables.ru)
        """
        if seasons is None:
            seasons = SEASONS["available"]

        all_data = []
        for season in seasons:
            df = self._get_season_data(season)
            if df is not None and not df.empty:
                all_data.append(df)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            # Сохранить обработанные данные
            output_path = PROCESSED_DIR / "rpl_historical.csv"
            combined.to_csv(output_path, index=False, encoding="utf-8")
            logger.info(f"Исторические данные сохранены: {len(combined)} матчей")
            return combined

        # Если данные недоступны - вернуть синтетические для демо
        return self._generate_synthetic_data()

    def _get_season_data(self, season: str) -> Optional[pd.DataFrame]:
        """Получить данные одного сезона"""
        cache_path = CACHE_DIR / f"rpl_season_{season.replace('/', '_')}.csv"

        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path)
                if not df.empty:
                    logger.info(f"Сезон {season}: загружен из кеша ({len(df)} матчей)")
                    return df
            except Exception:
                pass

        # Football-Data.co.uk формат: 2021/22 -> 2122
        season_code = season.replace("/", "")[-4:]
        url = self.FOOTBALL_DATA_URL.format(season=season_code)

        resp = safe_request(url)
        if resp and resp.status_code == 200:
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                df = self._normalize_football_data(df, season)
                df.to_csv(cache_path, index=False)
                logger.info(f"Сезон {season}: загружен ({len(df)} матчей)")
                return df
            except Exception as e:
                logger.error(f"Ошибка парсинга сезона {season}: {e}")

        return None

    def _normalize_football_data(self, df: pd.DataFrame, season: str) -> pd.DataFrame:
        """
        Нормализация данных Football-Data.co.uk
        Колонки: Date, HomeTeam, AwayTeam, FTHG, FTAG, HY, AY, HC, AC, HP, AP, Referee
        HY - желтые карточки хозяев
        AY - желтые карточки гостей
        HC - угловые хозяев
        AC - угловые гостей
        HP - пенальти хозяев (если есть)
        AP - пенальти гостей (если есть)
        """
        column_map = {
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "HY": "home_yellow_cards",
            "AY": "away_yellow_cards",
            "HR": "home_red_cards",
            "AR": "away_red_cards",
            "HC": "home_corners",
            "AC": "away_corners",
            "Referee": "referee",
            "HS": "home_shots",
            "AS": "away_shots",
            "HST": "home_shots_target",
            "AST": "away_shots_target",
            "HF": "home_fouls",
            "AF": "away_fouls",
        }

        existing_cols = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=existing_cols)

        # Добавить производные поля
        df["season"] = season
        df["total_yellow_cards"] = (
            df.get("home_yellow_cards", 0).fillna(0)
            + df.get("away_yellow_cards", 0).fillna(0)
        )
        df["total_corners"] = (
            df.get("home_corners", 0).fillna(0)
            + df.get("away_corners", 0).fillna(0)
        )
        df["total_fouls"] = (
            df.get("home_fouls", 0).fillna(0)
            + df.get("away_fouls", 0).fillna(0)
        )

        # Пенальти - нет прямых данных в Football-Data, считаем через голы и xG
        # Примерно: 1 пенальти на каждые ~12 матчей в РПЛ
        if "home_goals" in df.columns:
            # Приближение: пенальти было, если было много нарушений
            df["has_penalty"] = 0  # будет заполнено из fbref

        return df

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Генерация синтетических исторических данных на основе
        реальной статистики РПЛ 2020-2024
        """
        import numpy as np

        logger.warning("Генерация синтетических данных на основе статистики РПЛ")

        teams = [
            "Зенит", "ЦСКА", "Спартак", "Локомотив", "Краснодар",
            "Динамо", "Рубин", "Ростов", "Факел", "Оренбург",
            "Химки", "Ахмат", "Сочи", "Пари НН", "Урал", "Крылья Советов",
        ]

        # Все судьи для исторических сезонов (2023-24, 2024-25)
        referees_historical = [
            "Казарцев С.", "Вилков В.", "Матюнин А.", "Левников С.",
            "Безбородов А.", "Панин А.", "Карасев С.", "Иванов И.",
            "Москалев А.", "Чистяков Е.", "Еськов А.", "Лапочкин С.",
        ]

        # Судьи сезона 2025/2026 (только активные в этом сезоне)
        referees_2025_26 = REFEREES_2025_26

        # Статистика судей (строгость)
        referee_strictness = {
            "Казарцев С.": 1.2,
            "Вилков В.": 0.9,
            "Матюнин А.": 1.1,
            "Левников С.": 1.3,
            "Безбородов А.": 0.8,
            "Панин А.": 1.0,
            "Карасев С.": 1.15,
            "Иванов И.": 0.95,
            "Москалев А.": 1.05,
            "Чистяков Е.": 1.1,
            "Еськов А.": 0.85,
            "Лапочкин С.": 1.0,
        }

        # Средние показатели РПЛ по сезонам (реальные данные):
        # Желтые карточки: среднее 3.4 за матч
        # Угловые: среднее 9.8 за матч
        # Пенальти: ~28% матчей содержат пенальти
        # Выход медбригады: ~1.8 раза за матч

        np.random.seed(42)
        n_matches = 240  # примерно 2 исторических сезона

        data = []
        match_id = 1

        for i in range(n_matches):
            home = np.random.choice(teams)
            away = np.random.choice([t for t in teams if t != home])
            referee = np.random.choice(referees_historical)
            strictness = referee_strictness.get(referee, 1.0)

            # Желтые карточки (Пуассон, среднее 3.4, модифицированное строгостью)
            base_yellow = 3.4 * strictness
            home_yellow = np.random.poisson(base_yellow * 0.48)
            away_yellow = np.random.poisson(base_yellow * 0.52)

            # Угловые (зависит от команд)
            home_attack = np.random.uniform(0.8, 1.2)
            away_attack = np.random.uniform(0.8, 1.2)
            home_corners = np.random.poisson(5.0 * home_attack)
            away_corners = np.random.poisson(4.8 * away_attack)

            # Пенальти (Бернулли, ~28%)
            has_penalty = int(np.random.random() < 0.28)
            penalty_count = np.random.poisson(0.35) if has_penalty else 0

            # Выход медбригады (травмы: среднее 1.8 за матч)
            medical_exits = np.random.poisson(1.8)

            # Голы
            home_goals = np.random.poisson(1.5)
            away_goals = np.random.poisson(1.1)

            # Нарушения
            home_fouls = np.random.poisson(12.5)
            away_fouls = np.random.poisson(13.2)

            season = "2023-24" if i < 120 else "2024-25"

            data.append({
                "match_id": match_id,
                "date": f"2024-{(i % 10) + 1:02d}-{(i % 28) + 1:02d}",
                "season": season,
                "home_team": home,
                "away_team": away,
                "referee": referee,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_yellow_cards": home_yellow,
                "away_yellow_cards": away_yellow,
                "total_yellow_cards": home_yellow + away_yellow,
                "home_red_cards": int(np.random.random() < 0.05),
                "away_red_cards": int(np.random.random() < 0.05),
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": home_corners + away_corners,
                "home_fouls": home_fouls,
                "away_fouls": away_fouls,
                "total_fouls": home_fouls + away_fouls,
                "has_penalty": has_penalty,
                "penalty_count": penalty_count,
                "medical_exits": medical_exits,
                "referee_strictness": strictness,
            })
            match_id += 1

        # Добавить матчи сезона 2025/2026 (только активные судьи этого сезона)
        np.random.seed(2025)
        # РПЛ 2025/2026: 16 команд, 30 туров = 240 матчей в полном сезоне.
        # Генерируем ~120 матчей (первая половина сезона, по состоянию на март 2026)
        n_matches_2526 = 120
        for i in range(n_matches_2526):
            home = np.random.choice(teams)
            away = np.random.choice([t for t in teams if t != home])
            referee = np.random.choice(referees_2025_26)
            strictness = referee_strictness.get(referee, 1.0)

            base_yellow = 3.4 * strictness
            home_yellow = np.random.poisson(base_yellow * 0.48)
            away_yellow = np.random.poisson(base_yellow * 0.52)

            home_attack = np.random.uniform(0.8, 1.2)
            away_attack = np.random.uniform(0.8, 1.2)
            home_corners = np.random.poisson(5.0 * home_attack)
            away_corners = np.random.poisson(4.8 * away_attack)

            has_penalty = int(np.random.random() < 0.28)
            penalty_count = np.random.poisson(0.35) if has_penalty else 0
            medical_exits = np.random.poisson(1.8)
            home_goals = np.random.poisson(1.5)
            away_goals = np.random.poisson(1.1)
            home_fouls = np.random.poisson(12.5)
            away_fouls = np.random.poisson(13.2)

            # Даты: июль 2025 — март 2026
            month = 7 + (i * 8 // n_matches_2526)
            year = 2025 if month <= 12 else 2026
            month = month if month <= 12 else month - 12
            day = (i % 28) + 1

            data.append({
                "match_id": match_id,
                "date": f"{year}-{month:02d}-{day:02d}",
                "season": "2025-26",
                "home_team": home,
                "away_team": away,
                "referee": referee,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_yellow_cards": home_yellow,
                "away_yellow_cards": away_yellow,
                "total_yellow_cards": home_yellow + away_yellow,
                "home_red_cards": int(np.random.random() < 0.05),
                "away_red_cards": int(np.random.random() < 0.05),
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": home_corners + away_corners,
                "home_fouls": home_fouls,
                "away_fouls": away_fouls,
                "total_fouls": home_fouls + away_fouls,
                "has_penalty": has_penalty,
                "penalty_count": penalty_count,
                "medical_exits": medical_exits,
                "referee_strictness": strictness,
            })
            match_id += 1

        df = pd.DataFrame(data)

        output_path = PROCESSED_DIR / "rpl_synthetic.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Синтетические данные сохранены: {len(df)} матчей")
        return df


class RefereeStatsCollector:
    """Сборщик статистики судей"""

    def get_referee_stats(
        self, matches_df: pd.DataFrame, season: str = None
    ) -> pd.DataFrame:
        """
        Вычислить статистику каждого судьи.
        Если season задан — берём только судей, судивших матчи в этом сезоне.
        По умолчанию используется текущий сезон (SEASONS["current"]).
        Дополнительно обогащает данные из Smart Tables (smart-tables.ru).
        """
        if matches_df.empty or "referee" not in matches_df.columns:
            return pd.DataFrame()

        if season is None:
            season = SEASONS["current"]

        # Определяем судей, работавших в указанном сезоне
        if season and "season" in matches_df.columns:
            season_referees = set(
                matches_df[matches_df["season"] == season]["referee"].unique()
            )
            if season_referees:
                matches_df = matches_df[matches_df["referee"].isin(season_referees)]
            logger.info(
                f"Статистика судей: фильтр по сезону {season} "
                f"({len(season_referees)} судей)"
            )

        referee_stats = (
            matches_df.groupby("referee")
            .agg(
                matches_count=("match_id", "count"),
                avg_yellow_cards=("total_yellow_cards", "mean"),
                avg_corners=("total_corners", "mean"),
                avg_fouls=("total_fouls", "mean"),
                avg_penalties=("has_penalty", "mean"),
                avg_medical_exits=("medical_exits", "mean"),
            )
            .reset_index()
        )

        # Индекс строгости (относительно среднего)
        avg_yellow = referee_stats["avg_yellow_cards"].mean()
        referee_stats["strictness_index"] = (
            referee_stats["avg_yellow_cards"] / avg_yellow
        )

        # Индекс пенальти (относительно среднего)
        avg_penalty = referee_stats["avg_penalties"].mean()
        referee_stats["penalty_index"] = (
            referee_stats["avg_penalties"] / avg_penalty
        )

        # Обогащение данными из Smart Tables
        referee_stats = self._enrich_from_smart_tables(referee_stats)

        output_path = PROCESSED_DIR / "referee_stats.csv"
        referee_stats.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Статистика судей: {len(referee_stats)} судей")

        return referee_stats

    def _enrich_from_smart_tables(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Обогатить базовую статистику данными из Smart Tables.
        Если Smart Tables возвращает данные — перезаписываем avg_yellow_cards,
        avg_fouls, avg_penalties и пересчитываем индексы.
        """
        try:
            scraper = SmartTablesScraper()
            st_df = scraper.get_rpl_referee_stats()

            if st_df.empty:
                logger.info("Smart Tables: данные судей недоступны, используем базовые")
                base_df["source"] = "synthetic"
                return base_df

            logger.info(
                f"Smart Tables: получена статистика по {len(st_df)} судьям — "
                "обогащаем базовые данные"
            )

            # Объединяем по имени судьи
            merged = base_df.merge(
                st_df[["referee", "avg_yellow_cards", "avg_fouls", "avg_penalties"]],
                on="referee",
                how="left",
                suffixes=("_base", "_st"),
            )

            # Предпочитаем Smart Tables там, где данные есть
            for col in ["avg_yellow_cards", "avg_fouls", "avg_penalties"]:
                st_col = f"{col}_st"
                base_col = f"{col}_base"
                if st_col in merged.columns:
                    merged[col] = merged[st_col].combine_first(merged[base_col])
                    merged.drop(columns=[st_col, base_col], errors="ignore", inplace=True)

            # Пересчитать индексы
            avg_y = merged["avg_yellow_cards"].mean()
            if avg_y and avg_y > 0:
                merged["strictness_index"] = merged["avg_yellow_cards"] / avg_y

            avg_p = merged["avg_penalties"].mean()
            if avg_p and avg_p > 0:
                merged["penalty_index"] = merged["avg_penalties"] / avg_p

            merged["source"] = "smart-tables.ru"
            return merged

        except Exception as e:
            logger.warning(f"Smart Tables: ошибка обогащения данных судей: {e}")
            base_df["source"] = "synthetic"
            return base_df


class TeamStatsCollector:
    """Сборщик статистики команд"""

    def get_team_stats(
        self, matches_df: pd.DataFrame, last_n: int = 20
    ) -> pd.DataFrame:
        """
        Вычислить статистику каждой команды за последние N матчей
        """
        if matches_df.empty:
            return pd.DataFrame()

        # Дата должна быть в правильном формате
        if "date" in matches_df.columns:
            matches_df["date"] = pd.to_datetime(matches_df["date"], errors="coerce")
            matches_df = matches_df.sort_values("date")

        team_stats = {}
        teams = set(
            list(matches_df["home_team"].unique())
            + list(matches_df["away_team"].unique())
        )

        for team in teams:
            team_home = matches_df[matches_df["home_team"] == team].copy()
            team_away = matches_df[matches_df["away_team"] == team].copy()

            # Унифицированные данные для команды
            team_matches = []

            for _, row in team_home.iterrows():
                team_matches.append({
                    "date": row.get("date"),
                    "is_home": True,
                    "yellow_cards": row.get("home_yellow_cards", 0),
                    "opponent_yellow": row.get("away_yellow_cards", 0),
                    "corners": row.get("home_corners", 0),
                    "opponent_corners": row.get("away_corners", 0),
                    "fouls": row.get("home_fouls", 0),
                    "has_penalty": row.get("has_penalty", 0),
                    "medical_exits": row.get("medical_exits", 0),
                    "goals": row.get("home_goals", 0),
                    "goals_conceded": row.get("away_goals", 0),
                })

            for _, row in team_away.iterrows():
                team_matches.append({
                    "date": row.get("date"),
                    "is_home": False,
                    "yellow_cards": row.get("away_yellow_cards", 0),
                    "opponent_yellow": row.get("home_yellow_cards", 0),
                    "corners": row.get("away_corners", 0),
                    "opponent_corners": row.get("home_corners", 0),
                    "fouls": row.get("away_fouls", 0),
                    "has_penalty": row.get("has_penalty", 0),
                    "medical_exits": row.get("medical_exits", 0),
                    "goals": row.get("away_goals", 0),
                    "goals_conceded": row.get("home_goals", 0),
                })

            if not team_matches:
                continue

            team_df = pd.DataFrame(team_matches)
            team_df = team_df.sort_values("date").tail(last_n)

            team_stats[team] = {
                "team": team,
                "matches_analyzed": len(team_df),
                # Желтые карточки
                "avg_yellow_cards": team_df["yellow_cards"].mean(),
                "avg_yellow_cards_home": team_df[team_df["is_home"]]["yellow_cards"].mean()
                if len(team_df[team_df["is_home"]]) > 0
                else team_df["yellow_cards"].mean(),
                "avg_yellow_cards_away": team_df[~team_df["is_home"]]["yellow_cards"].mean()
                if len(team_df[~team_df["is_home"]]) > 0
                else team_df["yellow_cards"].mean(),
                # Угловые
                "avg_corners": team_df["corners"].mean(),
                "avg_corners_home": team_df[team_df["is_home"]]["corners"].mean()
                if len(team_df[team_df["is_home"]]) > 0
                else team_df["corners"].mean(),
                "avg_corners_away": team_df[~team_df["is_home"]]["corners"].mean()
                if len(team_df[~team_df["is_home"]]) > 0
                else team_df["corners"].mean(),
                # Нарушения
                "avg_fouls": team_df["fouls"].mean(),
                # Пенальти (вероятность)
                "penalty_rate": team_df["has_penalty"].mean(),
                # Медицина
                "avg_medical": team_df["medical_exits"].mean(),
                # Атаковая/защитная сила
                "avg_goals": team_df["goals"].mean(),
                "avg_goals_conceded": team_df["goals_conceded"].mean(),
            }

        stats_df = pd.DataFrame(list(team_stats.values()))

        # Обогащение данными из Smart Tables
        stats_df = self._enrich_from_smart_tables(stats_df)

        output_path = PROCESSED_DIR / "team_stats.csv"
        stats_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Статистика команд: {len(stats_df)} команд")

        return stats_df

    def _enrich_from_smart_tables(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Обогатить базовую статистику команд данными из Smart Tables.
        Smart Tables даёт актуальные данные текущего сезона РПЛ 2025/2026.
        """
        try:
            scraper = SmartTablesScraper()
            st_df = scraper.get_rpl_team_stats()

            if st_df.empty:
                logger.info("Smart Tables: данные команд недоступны, используем базовые")
                base_df["source"] = "synthetic"
                return base_df

            logger.info(
                f"Smart Tables: получена статистика по {len(st_df)} командам — "
                "обогащаем базовые данные"
            )

            # Объединяем по названию команды
            enrich_cols = [
                c for c in ["avg_corners", "avg_yellow_cards", "avg_fouls", "penalty_rate"]
                if c in st_df.columns
            ]
            merged = base_df.merge(
                st_df[["team"] + enrich_cols],
                on="team",
                how="left",
                suffixes=("_base", "_st"),
            )

            # Предпочитаем Smart Tables там, где данные есть
            for col in enrich_cols:
                st_col = f"{col}_st"
                base_col = f"{col}_base"
                if st_col in merged.columns:
                    merged[col] = merged[st_col].combine_first(merged[base_col])
                    merged.drop(columns=[st_col, base_col], errors="ignore", inplace=True)

            merged["source"] = "smart-tables.ru"
            return merged

        except Exception as e:
            logger.warning(f"Smart Tables: ошибка обогащения данных команд: {e}")
            base_df["source"] = "synthetic"
            return base_df

    def get_h2h_stats(
        self, matches_df: pd.DataFrame, home_team: str, away_team: str
    ) -> Dict:
        """Получить статистику очных встреч"""
        h2h = matches_df[
            (
                (matches_df["home_team"] == home_team)
                & (matches_df["away_team"] == away_team)
            )
            | (
                (matches_df["home_team"] == away_team)
                & (matches_df["away_team"] == home_team)
            )
        ].copy()

        if h2h.empty:
            return {}

        return {
            "matches_count": len(h2h),
            "avg_total_yellow": h2h["total_yellow_cards"].mean(),
            "avg_total_corners": h2h["total_corners"].mean(),
            "penalty_rate": h2h["has_penalty"].mean(),
            "avg_medical": h2h["medical_exits"].mean()
            if "medical_exits" in h2h.columns
            else 1.8,
        }


class SmartTablesScraper:
    """
    Сборщик данных с Smart Tables (smart-tables.ru)
    Источник: статистика команд и судей РПЛ
    Страницы:
      - /league/russia/premier_league  — статистика команд
      - /referee                       — статистика судей (фильтр по РПЛ)
    """

    BASE_URL = "https://smart-tables.ru"
    RPL_URL = "https://smart-tables.ru/league/russia/premier_league"
    REFEREE_URL = "https://smart-tables.ru/referee"

    # Заголовки, имитирующие браузер
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }

    # Русские названия команд РПЛ → нормализованные
    TEAM_NAME_MAP = {
        "Зенит": "Зенит",
        "ЦСКА": "ЦСКА",
        "Спартак": "Спартак",
        "Локомотив": "Локомотив",
        "Краснодар": "Краснодар",
        "Динамо": "Динамо",
        "Рубин": "Рубин",
        "Ростов": "Ростов",
        "Факел": "Факел",
        "Оренбург": "Оренбург",
        "Химки": "Химки",
        "Ахмат": "Ахмат",
        "Сочи": "Сочи",
        "Пари НН": "Пари НН",
        "Урал": "Урал",
        "Крылья Советов": "Крылья Советов",
        # Альтернативные названия
        "Нижний Новгород": "Пари НН",
        "Крылья": "Крылья Советов",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._session_initialized = False

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def get_rpl_team_stats(self) -> pd.DataFrame:
        """
        Получить статистику команд РПЛ текущего сезона.
        Возвращает DataFrame с колонками:
          team, avg_corners, avg_yellow_cards, avg_fouls, penalty_rate, source
        """
        cache_path = CACHE_DIR / "smart_tables_team_stats.csv"
        cached = self._load_cache(cache_path, ttl_hours=6)
        if cached is not None:
            logger.info("Smart Tables: статистика команд загружена из кеша")
            return cached

        self._init_session()
        html = self._fetch(self.RPL_URL)
        if not html:
            logger.warning("Smart Tables: не удалось получить страницу лиги РПЛ")
            return pd.DataFrame()

        df = self._parse_league_page(html)
        if not df.empty:
            df.to_csv(cache_path, index=False, encoding="utf-8")
            logger.info(f"Smart Tables: команд загружено {len(df)}")
        return df

    def get_rpl_referee_stats(self) -> pd.DataFrame:
        """
        Получить статистику судей РПЛ текущего сезона.
        Возвращает DataFrame с колонками:
          referee, matches_count, avg_yellow_cards, avg_fouls,
          avg_penalties, strictness_index, penalty_index, source
        """
        cache_path = CACHE_DIR / "smart_tables_referee_stats.csv"
        cached = self._load_cache(cache_path, ttl_hours=6)
        if cached is not None:
            logger.info("Smart Tables: статистика судей загружена из кеша")
            return cached

        self._init_session()

        # Пробуем URL судей с фильтром по лиге
        for url in [
            f"{self.REFEREE_URL}/russia/premier_league",
            f"{self.REFEREE_URL}?league=russia&competition=premier_league",
            self.REFEREE_URL,
        ]:
            html = self._fetch(url, referer=self.RPL_URL)
            if html:
                df = self._parse_referee_page(html)
                if not df.empty:
                    df.to_csv(cache_path, index=False, encoding="utf-8")
                    logger.info(f"Smart Tables: судей загружено {len(df)}")
                    return df

        logger.warning("Smart Tables: не удалось получить статистику судей")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Парсинг страниц
    # ------------------------------------------------------------------

    def _parse_league_page(self, html: str) -> pd.DataFrame:
        """Разобрать страницу лиги РПЛ"""
        # 1. Попытка извлечь данные из встроенного JSON (Nuxt/Next/Vue)
        df = self._extract_from_json(html, data_type="teams")
        if not df.empty:
            return df

        # 2. Парсинг HTML-таблицы
        soup = BeautifulSoup(html, "html.parser")
        return self._parse_stats_table(soup, data_type="teams")

    def _parse_referee_page(self, html: str) -> pd.DataFrame:
        """Разобрать страницу статистики судей"""
        # 1. Попытка извлечь данные из встроенного JSON
        df = self._extract_from_json(html, data_type="referees")
        if not df.empty:
            return df

        # 2. Парсинг HTML-таблицы
        soup = BeautifulSoup(html, "html.parser")
        return self._parse_stats_table(soup, data_type="referees")

    def _extract_from_json(self, html: str, data_type: str) -> pd.DataFrame:
        """
        Извлечь данные из встроенного JSON на странице.
        Ищет: window.__NUXT__, window.__NEXT_DATA__, window.initialData,
               __STATE__, data-page и другие паттерны.
        """
        patterns = [
            r"window\.__NUXT__\s*=\s*(\{.*?\});?\s*</script>",
            r"window\.__NEXT_DATA__\s*=\s*(\{.*?\})\s*</script>",
            r"window\.initialData\s*=\s*(\{.*?\});?\s*</script>",
            r"window\.__STATE__\s*=\s*(\{.*?\});?\s*</script>",
            r"<script[^>]+id=[\"']__NUXT_DATA__[\"'][^>]*>(\[.*?\])</script>",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, html, re.DOTALL)
            for raw in matches:
                try:
                    data = json.loads(raw)
                    df = self._json_to_dataframe(data, data_type)
                    if not df.empty:
                        return df
                except (json.JSONDecodeError, Exception):
                    continue

        # Поиск JSON массивов в inline-скриптах
        script_jsons = re.findall(
            r'"(?:teams|referees|stats|rows?)"\s*:\s*(\[.*?\])',
            html, re.DOTALL
        )
        for raw in script_jsons:
            try:
                items = json.loads(raw)
                if items and isinstance(items, list):
                    df = pd.DataFrame(items)
                    normalized = self._normalize_json_df(df, data_type)
                    if not normalized.empty:
                        return normalized
            except (json.JSONDecodeError, Exception):
                continue

        return pd.DataFrame()

    def _json_to_dataframe(self, data: dict, data_type: str) -> pd.DataFrame:
        """Конвертировать JSON данные в DataFrame нужного формата"""
        if not isinstance(data, dict):
            return pd.DataFrame()

        # Обходим вложенные структуры в поисках списков команд/судей
        candidates = []

        def find_lists(obj, depth=0):
            if depth > 8:
                return
            if isinstance(obj, list) and len(obj) >= 5:
                candidates.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    find_lists(v, depth + 1)

        find_lists(data)

        for items in candidates:
            if not items or not isinstance(items[0], dict):
                continue
            df = pd.DataFrame(items)
            normalized = self._normalize_json_df(df, data_type)
            if not normalized.empty:
                return normalized

        return pd.DataFrame()

    def _normalize_json_df(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Привести JSON DataFrame к стандартному формату"""
        if data_type == "teams":
            return self._normalize_team_json(df)
        elif data_type == "referees":
            return self._normalize_referee_json(df)
        return pd.DataFrame()

    def _normalize_team_json(self, df: pd.DataFrame) -> pd.DataFrame:
        """Нормализация JSON данных команд"""
        # Маппинг возможных имен полей
        field_candidates = {
            "team": ["team", "name", "teamName", "club", "title"],
            "avg_corners": ["corners", "avgCorners", "avg_corners", "corner"],
            "avg_yellow_cards": ["yellowCards", "avg_yellow", "yc", "yellow", "ЖК"],
            "avg_fouls": ["fouls", "avgFouls", "avg_fouls", "foul"],
            "penalty_rate": ["penalty", "penalties", "penaltyRate", "pen"],
        }

        result = {}
        for target, candidates in field_candidates.items():
            for c in candidates:
                if c in df.columns:
                    result[target] = df[c]
                    break

        if "team" not in result:
            return pd.DataFrame()

        out = pd.DataFrame(result)
        out["source"] = "smart-tables.ru"
        # Нормализация названий команд
        if "team" in out.columns:
            out["team"] = out["team"].apply(
                lambda x: self.TEAM_NAME_MAP.get(str(x).strip(), str(x).strip())
            )
        return out

    def _normalize_referee_json(self, df: pd.DataFrame) -> pd.DataFrame:
        """Нормализация JSON данных судей"""
        field_candidates = {
            "referee": ["referee", "name", "refereeName", "судья"],
            "matches_count": ["matches", "matchesCount", "gamesCount", "games"],
            "avg_yellow_cards": ["yellowCards", "avgYellow", "yc", "yellow"],
            "avg_fouls": ["fouls", "avgFouls"],
            "avg_penalties": ["penalties", "avgPenalties", "pen"],
        }

        result = {}
        for target, candidates in field_candidates.items():
            for c in candidates:
                if c in df.columns:
                    result[target] = df[c]
                    break

        if "referee" not in result:
            return pd.DataFrame()

        out = pd.DataFrame(result)
        out["source"] = "smart-tables.ru"
        out = self._add_referee_indices(out)
        return out

    def _parse_stats_table(self, soup: BeautifulSoup, data_type: str) -> pd.DataFrame:
        """Разобрать HTML-таблицу статистики"""
        # Ищем таблицы с данными
        tables = soup.find_all("table")
        if not tables:
            # Попробуем div-based таблицы (SPA сайты часто используют их)
            tables = soup.find_all(
                ["div", "section"],
                class_=re.compile(r"table|grid|stats|stat|row", re.I)
            )

        for table in tables:
            rows = table.find_all("tr") if table.name == "table" else []
            if len(rows) < 3:
                continue

            # Заголовки
            header_row = rows[0]
            headers = [
                th.get_text(strip=True).lower()
                for th in header_row.find_all(["th", "td"])
            ]

            if not headers:
                continue

            # Данные строк
            data_rows = []
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if cells:
                    data_rows.append([c.get_text(strip=True) for c in cells])

            if not data_rows:
                continue

            df = pd.DataFrame(data_rows, columns=headers[:len(data_rows[0])])

            if data_type == "teams":
                normalized = self._normalize_team_table(df)
            else:
                normalized = self._normalize_referee_table(df)

            if not normalized.empty:
                return normalized

        return pd.DataFrame()

    def _normalize_team_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Нормализовать HTML-таблицу команд"""
        # Маппинг заголовков таблицы (на русском и английском)
        col_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(k in col_lower for k in ["команда", "team", "клуб"]):
                col_map[col] = "team"
            elif any(k in col_lower for k in ["угл", "corner", "ук"]):
                col_map[col] = "avg_corners"
            elif any(k in col_lower for k in ["жёлт", "yellow", "жк", "ж.к"]):
                col_map[col] = "avg_yellow_cards"
            elif any(k in col_lower for k in ["фол", "foul", "нарушен"]):
                col_map[col] = "avg_fouls"
            elif any(k in col_lower for k in ["пенальт", "penalty", "пен"]):
                col_map[col] = "penalty_rate"

        if "team" not in col_map.values():
            return pd.DataFrame()

        df = df.rename(columns=col_map)
        keep = [c for c in ["team", "avg_corners", "avg_yellow_cards",
                             "avg_fouls", "penalty_rate"] if c in df.columns]
        out = df[keep].copy()

        # Конвертация числовых полей
        for col in keep:
            if col != "team":
                out[col] = pd.to_numeric(out[col].str.replace(",", "."), errors="coerce")

        out["team"] = out["team"].apply(
            lambda x: self.TEAM_NAME_MAP.get(str(x).strip(), str(x).strip())
        )
        out["source"] = "smart-tables.ru"
        return out.dropna(subset=["team"])

    def _normalize_referee_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Нормализовать HTML-таблицу судей"""
        col_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(k in col_lower for k in ["судья", "referee", "арбитр"]):
                col_map[col] = "referee"
            elif any(k in col_lower for k in ["матч", "игр", "match", "game"]):
                col_map[col] = "matches_count"
            elif any(k in col_lower for k in ["жёлт", "yellow", "жк"]):
                col_map[col] = "avg_yellow_cards"
            elif any(k in col_lower for k in ["фол", "foul"]):
                col_map[col] = "avg_fouls"
            elif any(k in col_lower for k in ["пенальт", "penalty"]):
                col_map[col] = "avg_penalties"

        if "referee" not in col_map.values():
            return pd.DataFrame()

        df = df.rename(columns=col_map)
        keep = [c for c in ["referee", "matches_count", "avg_yellow_cards",
                             "avg_fouls", "avg_penalties"] if c in df.columns]
        out = df[keep].copy()

        for col in keep:
            if col != "referee":
                out[col] = pd.to_numeric(out[col].str.replace(",", "."), errors="coerce")

        out["source"] = "smart-tables.ru"
        out = self._add_referee_indices(out)
        return out.dropna(subset=["referee"])

    def _add_referee_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить производные индексы (строгость, пенальти)"""
        if "avg_yellow_cards" in df.columns and df["avg_yellow_cards"].notna().any():
            avg = df["avg_yellow_cards"].mean()
            if avg > 0:
                df["strictness_index"] = df["avg_yellow_cards"] / avg

        if "avg_penalties" in df.columns and df["avg_penalties"].notna().any():
            avg = df["avg_penalties"].mean()
            if avg > 0:
                df["penalty_index"] = df["avg_penalties"] / avg

        return df

    # ------------------------------------------------------------------
    # Сетевые утилиты
    # ------------------------------------------------------------------

    def _init_session(self):
        """Инициализировать сессию: получить cookies с главной страницы"""
        if self._session_initialized:
            return
        try:
            resp = self.session.get(
                self.BASE_URL, timeout=15, allow_redirects=True
            )
            if resp.status_code == 200:
                self._session_initialized = True
                logger.debug("Smart Tables: сессия инициализирована")
        except Exception as e:
            logger.debug(f"Smart Tables: не удалось инициализировать сессию: {e}")

    def _fetch(self, url: str, referer: str = None) -> Optional[str]:
        """Выполнить GET-запрос с правильными заголовками"""
        headers = {}
        if referer:
            headers["Referer"] = referer
        else:
            headers["Referer"] = self.BASE_URL

        try:
            resp = self.session.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                logger.debug(f"Smart Tables: загружено {url}")
                return resp.text
            logger.warning(
                f"Smart Tables: {url} вернул статус {resp.status_code}"
            )
        except Exception as e:
            logger.warning(f"Smart Tables: ошибка при запросе {url}: {e}")
        return None

    @staticmethod
    def _load_cache(path: Path, ttl_hours: int = 6) -> Optional[pd.DataFrame]:
        """Загрузить из кеша если файл не старше ttl_hours"""
        if not path.exists():
            return None
        age_hours = (
            datetime.now().timestamp() - path.stat().st_mtime
        ) / 3600
        if age_hours > ttl_hours:
            return None
        try:
            df = pd.read_csv(path, encoding="utf-8")
            return df if not df.empty else None
        except Exception:
            return None

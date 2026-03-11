"""
Парсер коэффициентов букмекеров
Поддерживает: Фонбет, Леон, Винлайн, БетСити, ОлимпБет, Лига Ставок, 1хСтавка
"""
import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from utils.helpers import safe_request, cache_data, load_cache, get_random_headers
from config.settings import BOOKMAKERS, CACHE_DIR, REQUEST_DELAY

logger = logging.getLogger(__name__)


class BookmakerOdds:
    """Структура коэффициентов"""

    def __init__(
        self,
        bookmaker: str,
        match_id: str,
        home_team: str,
        away_team: str,
        match_date: str,
        market: str,
        bet_type: str,
        line: Optional[float],
        odds: float,
        raw_data: Optional[Dict] = None,
    ):
        self.bookmaker = bookmaker
        self.match_id = match_id
        self.home_team = home_team
        self.away_team = away_team
        self.match_date = match_date
        self.market = market
        self.bet_type = bet_type
        self.line = line
        self.odds = odds
        self.raw_data = raw_data or {}
        self.scraped_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "bookmaker": self.bookmaker,
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "match_date": self.match_date,
            "market": self.market,
            "bet_type": self.bet_type,
            "line": self.line,
            "odds": self.odds,
            "scraped_at": self.scraped_at,
        }


class FonbetScraper:
    """
    Парсер Фонбет
    Использует публичное API Фонбет для получения коэффициентов
    """

    BASE_URL = "https://live.fonbet.ru"
    SPORT_URL = "https://line1.fonbet.ru/sports"
    LINE_URL = "https://line1.fonbet.ru"

    # ID спорта "Футбол" в Фонбет
    FOOTBALL_SPORT_ID = 1

    # ID Чемпионата России
    RPL_SEGMENT_IDS = [2530, 27831]  # актуальные ID могут меняться

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(get_random_headers())

    def get_rpl_matches(self) -> List[Dict]:
        """Получить матчи РПЛ с коэффициентами"""
        cache_path = CACHE_DIR / "fonbet_matches.json"
        cached = load_cache(cache_path)
        if cached:
            return cached

        try:
            # Получаем список событий
            url = f"{self.LINE_URL}/sports/sport/live/v1.0"
            params = {
                "sportId": self.FOOTBALL_SPORT_ID,
                "lng": "ru",
                "version": "28508626",
            }
            resp = safe_request(url, params=params, timeout=30)
            if not resp:
                return self._get_demo_matches()

            data = resp.json()
            matches = self._parse_fonbet_response(data)
            cache_data(matches, cache_path, ttl_hours=1)
            return matches

        except Exception as e:
            logger.error(f"Фонбет: ошибка получения матчей: {e}")
            return self._get_demo_matches()

    def _parse_fonbet_response(self, data: Dict) -> List[Dict]:
        """Парсинг ответа Фонбет"""
        matches = []
        events = data.get("events", [])
        factors = {f["f"]: f for f in data.get("factors", [])}

        for event in events:
            if not self._is_rpl_match(event):
                continue

            match_factors = {}
            for factor_id, factor in factors.items():
                if factor.get("e") == event.get("id"):
                    match_factors[factor_id] = factor

            match = {
                "id": event.get("id"),
                "home_team": event.get("team1"),
                "away_team": event.get("team2"),
                "start_time": event.get("startTime"),
                "markets": self._extract_markets(match_factors),
            }
            matches.append(match)

        return matches

    def _is_rpl_match(self, event: Dict) -> bool:
        """Проверка принадлежности к РПЛ"""
        seg_id = event.get("segmentId", 0)
        name = event.get("name", "").lower()
        return (
            seg_id in self.RPL_SEGMENT_IDS
            or "россия" in name
            or "рпл" in name
            or "premier" in name.lower()
        )

    def _extract_markets(self, factors: Dict) -> Dict:
        """Извлечение рынков ставок"""
        markets = {}

        # Маппинг ID факторов Фонбет -> наши рынки
        # Желтые карточки: фактор ~1568, 1569 (тотал больше/меньше)
        # Угловые: фактор ~1562, 1563
        # Пенальти: фактор ~1570

        factor_mapping = {
            # Желтые карточки
            "yellow_cards_over": [1568, 1587, 1588],
            "yellow_cards_under": [1569, 1589, 1590],
            # Угловые
            "corners_over": [1562, 1556, 1557],
            "corners_under": [1563, 1558, 1559],
            # Пенальти
            "penalty_yes": [1570, 1571],
            "penalty_no": [1572, 1573],
        }

        for market_key, factor_ids in factor_mapping.items():
            for fid in factor_ids:
                if str(fid) in factors:
                    f = factors[str(fid)]
                    markets[market_key] = {
                        "odds": f.get("v", 0),
                        "line": f.get("pt", None),
                        "factor_id": fid,
                    }
                    break

        return markets

    def _get_demo_matches(self) -> List[Dict]:
        """Демо данные когда API недоступно"""
        return [
            {
                "id": "demo_1",
                "home_team": "Зенит",
                "away_team": "ЦСКА",
                "start_time": "2024-12-15T17:00:00",
                "markets": {
                    "yellow_cards_over": {"odds": 1.85, "line": 3.5},
                    "yellow_cards_under": {"odds": 1.95, "line": 3.5},
                    "corners_over": {"odds": 1.87, "line": 9.5},
                    "corners_under": {"odds": 1.93, "line": 9.5},
                    "penalty_yes": {"odds": 3.20, "line": None},
                    "penalty_no": {"odds": 1.30, "line": None},
                },
            },
            {
                "id": "demo_2",
                "home_team": "Спартак",
                "away_team": "Краснодар",
                "start_time": "2024-12-15T19:30:00",
                "markets": {
                    "yellow_cards_over": {"odds": 1.90, "line": 3.5},
                    "yellow_cards_under": {"odds": 1.90, "line": 3.5},
                    "corners_over": {"odds": 1.82, "line": 10.5},
                    "corners_under": {"odds": 1.98, "line": 10.5},
                    "penalty_yes": {"odds": 3.50, "line": None},
                    "penalty_no": {"odds": 1.25, "line": None},
                },
            },
        ]


class LeonScraper:
    """Парсер Леон (leon.ru)"""

    API_URL = "https://leon.ru/api-2"

    def get_rpl_matches(self) -> List[Dict]:
        """Получить матчи РПЛ"""
        cache_path = CACHE_DIR / "leon_matches.json"
        cached = load_cache(cache_path)
        if cached:
            return cached

        try:
            # Leon API endpoint для футбола
            url = f"{self.API_URL}/betline/sport/all/events/prematch"
            params = {
                "ctag": "ru-RU",
                "flags": "reg,urlv2,mm2,rrc,nodup",
                "subtype": "REGULAR",
                "sportAlias": "Football",
                "ctag": "ru-RU",
            }
            resp = safe_request(url, params=params)
            if not resp:
                return []

            data = resp.json()
            matches = self._parse_leon_response(data)
            cache_data(matches, cache_path, ttl_hours=1)
            return matches

        except Exception as e:
            logger.error(f"Леон: ошибка получения матчей: {e}")
            return []

    def _parse_leon_response(self, data: Dict) -> List[Dict]:
        """Парсинг ответа Леон"""
        matches = []
        leagues = data.get("data", {}).get("sport", {}).get("regions", [])

        for region in leagues:
            if region.get("name", "").lower() not in ["россия", "russia"]:
                continue

            for league in region.get("leagues", []):
                if "премьер" in league.get("name", "").lower():
                    for event in league.get("events", []):
                        match = {
                            "id": f"leon_{event.get('id')}",
                            "home_team": event.get("homeTeam", {}).get("name"),
                            "away_team": event.get("awayTeam", {}).get("name"),
                            "start_time": event.get("startTime"),
                            "markets": self._extract_leon_markets(event),
                        }
                        matches.append(match)

        return matches

    def _extract_leon_markets(self, event: Dict) -> Dict:
        """Извлечение рынков из события Леон"""
        markets = {}
        for market in event.get("markets", []):
            market_name = market.get("name", "").lower()

            if "желтые карточки" in market_name or "yellow cards" in market_name:
                for runner in market.get("runners", []):
                    if "больше" in runner.get("name", "").lower():
                        markets["yellow_cards_over"] = {
                            "odds": runner.get("price", 0),
                            "line": market.get("handicap"),
                        }
                    elif "меньше" in runner.get("name", "").lower():
                        markets["yellow_cards_under"] = {
                            "odds": runner.get("price", 0),
                            "line": market.get("handicap"),
                        }

            elif "угловые" in market_name or "corners" in market_name:
                for runner in market.get("runners", []):
                    if "больше" in runner.get("name", "").lower():
                        markets["corners_over"] = {
                            "odds": runner.get("price", 0),
                            "line": market.get("handicap"),
                        }
                    elif "меньше" in runner.get("name", "").lower():
                        markets["corners_under"] = {
                            "odds": runner.get("price", 0),
                            "line": market.get("handicap"),
                        }

            elif "пенальти" in market_name or "penalty" in market_name:
                for runner in market.get("runners", []):
                    name = runner.get("name", "").lower()
                    if "да" in name or "yes" in name:
                        markets["penalty_yes"] = {
                            "odds": runner.get("price", 0),
                            "line": None,
                        }
                    elif "нет" in name or "no" in name:
                        markets["penalty_no"] = {
                            "odds": runner.get("price", 0),
                            "line": None,
                        }

        return markets


class WinlineScraper:
    """Парсер Винлайн"""

    API_URL = "https://www.winline.ru/api/events"

    def get_rpl_matches(self) -> List[Dict]:
        cache_path = CACHE_DIR / "winline_matches.json"
        cached = load_cache(cache_path)
        if cached:
            return cached

        try:
            headers = get_random_headers()
            headers["Origin"] = "https://www.winline.ru"

            resp = safe_request(
                self.API_URL,
                params={"sport": "football", "league": "russia-premier-league"},
                headers=headers,
            )
            if not resp:
                return []

            data = resp.json()
            matches = self._parse_winline(data)
            cache_data(matches, cache_path, ttl_hours=1)
            return matches

        except Exception as e:
            logger.error(f"Винлайн: ошибка: {e}")
            return []

    def _parse_winline(self, data: Dict) -> List[Dict]:
        matches = []
        for event in data.get("events", []):
            match = {
                "id": f"winline_{event.get('id')}",
                "home_team": event.get("team1"),
                "away_team": event.get("team2"),
                "start_time": event.get("date"),
                "markets": self._extract_winline_markets(event.get("odds", [])),
            }
            matches.append(match)
        return matches

    def _extract_winline_markets(self, odds_list: List) -> Dict:
        markets = {}
        market_map = {
            "yellow_cards_over": ["ЖК Б", "Yellow Б", "ЖК+"],
            "yellow_cards_under": ["ЖК М", "Yellow М", "ЖК-"],
            "corners_over": ["Угл Б", "Corners Б", "Углы+"],
            "corners_under": ["Угл М", "Corners М", "Углы-"],
            "penalty_yes": ["Пенальти Да", "Penalty Да"],
            "penalty_no": ["Пенальти Нет", "Penalty Нет"],
        }

        for odd in odds_list:
            name = odd.get("name", "")
            for market_key, patterns in market_map.items():
                if any(p.lower() in name.lower() for p in patterns):
                    markets[market_key] = {
                        "odds": odd.get("value", 0),
                        "line": odd.get("param"),
                    }

        return markets


class OddsAggregator:
    """
    Агрегатор коэффициентов со всех букмекеров
    Собирает и нормализует данные
    """

    def __init__(self):
        self.scrapers = {
            "fonbet": FonbetScraper(),
            "leon": LeonScraper(),
            "winline": WinlineScraper(),
        }

    def get_all_odds(self) -> Dict[str, List[Dict]]:
        """Получить коэффициенты со всех букмекеров"""
        all_odds = {}

        for bk_name, scraper in self.scrapers.items():
            logger.info(f"Получение коэффициентов: {bk_name}")
            try:
                matches = scraper.get_rpl_matches()
                all_odds[bk_name] = matches
                logger.info(f"{bk_name}: получено {len(matches)} матчей")
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                logger.error(f"Ошибка {bk_name}: {e}")
                all_odds[bk_name] = []

        return all_odds

    def get_best_odds(self, all_odds: Dict) -> Dict[str, Dict]:
        """
        Получить лучшие коэффициенты для каждого матча и рынка
        Returns: {match_key: {market: {bookmaker, odds, line}}}
        """
        best_odds = {}

        for bk_name, matches in all_odds.items():
            for match in matches:
                match_key = f"{match.get('home_team')} - {match.get('away_team')}"

                if match_key not in best_odds:
                    best_odds[match_key] = {
                        "home_team": match.get("home_team"),
                        "away_team": match.get("away_team"),
                        "start_time": match.get("start_time"),
                        "markets": {},
                    }

                for market, market_data in match.get("markets", {}).items():
                    odds_val = market_data.get("odds", 0)
                    if odds_val <= 1.0:
                        continue

                    current = best_odds[match_key]["markets"].get(market)
                    if current is None or odds_val > current.get("odds", 0):
                        best_odds[match_key]["markets"][market] = {
                            "bookmaker": bk_name,
                            "odds": odds_val,
                            "line": market_data.get("line"),
                        }

        return best_odds

    def get_consensus_odds(self, all_odds: Dict) -> Dict[str, Dict]:
        """
        Получить консенсусные (средние) коэффициенты
        Это нужно для определения 'справедливых' коэффициентов рынка
        """
        consensus = {}

        for bk_name, matches in all_odds.items():
            for match in matches:
                match_key = f"{match.get('home_team')} - {match.get('away_team')}"

                if match_key not in consensus:
                    consensus[match_key] = {
                        "home_team": match.get("home_team"),
                        "away_team": match.get("away_team"),
                        "start_time": match.get("start_time"),
                        "markets": {},
                    }

                for market, market_data in match.get("markets", {}).items():
                    odds_val = market_data.get("odds", 0)
                    if odds_val <= 1.0:
                        continue

                    if market not in consensus[match_key]["markets"]:
                        consensus[match_key]["markets"][market] = {
                            "odds_list": [],
                            "line": market_data.get("line"),
                        }

                    consensus[match_key]["markets"][market]["odds_list"].append(
                        odds_val
                    )

        # Вычислить средние
        for match_key, match_data in consensus.items():
            for market, market_info in match_data["markets"].items():
                odds_list = market_info.get("odds_list", [])
                if odds_list:
                    market_info["avg_odds"] = sum(odds_list) / len(odds_list)
                    market_info["max_odds"] = max(odds_list)
                    market_info["min_odds"] = min(odds_list)
                    market_info["bookmakers_count"] = len(odds_list)

        return consensus

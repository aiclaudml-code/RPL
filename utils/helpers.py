"""
Вспомогательные утилиты
"""
import time
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import requests
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)

ua = UserAgent()


def get_random_headers() -> Dict[str, str]:
    """Случайные заголовки для запросов"""
    return {
        "User-Agent": ua.random,
        "Accept": "application/json, text/html, */*",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
    }


def safe_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    timeout: int = 30,
    retries: int = 3,
    delay: float = 1.5,
) -> Optional[requests.Response]:
    """Безопасный HTTP запрос с retry логикой"""
    if headers is None:
        headers = get_random_headers()

    for attempt in range(retries):
        try:
            if method == "GET":
                response = requests.get(
                    url, headers=headers, params=params, timeout=timeout
                )
            else:
                response = requests.post(
                    url, headers=headers, json=json_data, timeout=timeout
                )

            response.raise_for_status()
            time.sleep(delay)
            return response

        except requests.exceptions.RequestException as e:
            logger.warning(f"Попытка {attempt + 1}/{retries} - Ошибка: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))

    return None


def cache_data(data: Any, cache_path: Path, ttl_hours: int = 6) -> None:
    """Кеширование данных"""
    cache_obj = {
        "timestamp": datetime.now().isoformat(),
        "ttl_hours": ttl_hours,
        "data": data,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_obj, f, ensure_ascii=False, indent=2)


def load_cache(cache_path: Path) -> Optional[Any]:
    """Загрузка данных из кеша"""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_obj = json.load(f)

        cached_time = datetime.fromisoformat(cache_obj["timestamp"])
        ttl = timedelta(hours=cache_obj.get("ttl_hours", 6))

        if datetime.now() - cached_time < ttl:
            return cache_obj["data"]

    except (json.JSONDecodeError, KeyError):
        pass

    return None


def odds_to_probability(odds: float) -> float:
    """Конвертация коэффициента в вероятность"""
    if odds <= 1.0:
        return 0.0
    return 1.0 / odds


def probability_to_odds(prob: float) -> float:
    """Конвертация вероятности в коэффициент"""
    if prob <= 0 or prob >= 1:
        return 0.0
    return 1.0 / prob


def remove_bookmaker_margin(
    odds_list: list, method: str = "basic"
) -> list:
    """Удаление маржи букмекера из коэффициентов"""
    if not odds_list:
        return []

    probs = [1.0 / o for o in odds_list if o > 1.0]
    if not probs:
        return []

    total_prob = sum(probs)
    if method == "basic":
        # Пропорциональное удаление маржи
        fair_probs = [p / total_prob for p in probs]
    else:
        fair_probs = [p / total_prob for p in probs]

    return fair_probs


def calculate_value(model_prob: float, bookmaker_odds: float) -> float:
    """
    Расчет ценности ставки (Value)
    Value > 0 означает выгодную ставку
    """
    return (model_prob * bookmaker_odds) - 1.0


def kelly_criterion(
    probability: float, odds: float, fraction: float = 0.25
) -> float:
    """
    Критерий Келли для определения размера ставки
    fraction - дробный Келли (0.25 = четверть Келли)
    """
    if odds <= 1.0 or probability <= 0:
        return 0.0

    q = 1 - probability
    b = odds - 1

    kelly = (b * probability - q) / b
    return max(0, kelly * fraction)


def format_odds(odds: float) -> str:
    """Форматирование коэффициента"""
    return f"{odds:.2f}"


def format_probability(prob: float) -> str:
    """Форматирование вероятности"""
    return f"{prob * 100:.1f}%"


def normalize_team_name(name: str) -> str:
    """Нормализация названия команды"""
    replacements = {
        "Spartak": "Спартак",
        "Zenit": "Зенит",
        "CSKA": "ЦСКА",
        "Lokomotiv": "Локомотив",
        "Krasnodar": "Краснодар",
        "Dynamo": "Динамо",
        "Rubin": "Рубин",
        "Rostov": "Ростов",
        "Fakel": "Факел",
        "Orenburg": "Оренбург",
        "Khimki": "Химки",
        "Akhmat": "Ахмат",
        "Sochi": "Сочи",
        "Ural": "Урал",
        "Krylia Sovetov": "Крылья Советов",
        "Krylya Sovetov": "Крылья Советов",
    }
    for eng, rus in replacements.items():
        if eng.lower() in name.lower():
            return rus
    return name


def setup_logging(level: str = "INFO") -> None:
    """Настройка логирования"""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(__file__).parent.parent / "data" / "app.log",
                encoding="utf-8"
            ),
        ],
    )

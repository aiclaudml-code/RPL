"""
Настройки для предиктивной модели ставок на Чемпионат России по футболу
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Создать директории если не существуют
for d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==================== БУКМЕКЕРЫ ====================
BOOKMAKERS = {
    "fonbet": {
        "name": "Фонбет",
        "base_url": "https://www.fonbet.ru",
        "api_url": "https://live.fonbet.ru/api/v1",
        "enabled": True,
    },
    "leon": {
        "name": "Леон",
        "base_url": "https://leon.ru",
        "api_url": "https://leon.ru/api-2",
        "enabled": True,
    },
    "winline": {
        "name": "Винлайн",
        "base_url": "https://www.winline.ru",
        "api_url": "https://www.winline.ru/betapi",
        "enabled": True,
    },
    "betcity": {
        "name": "БетСити",
        "base_url": "https://betcity.ru",
        "api_url": "https://betcity.ru/api",
        "enabled": True,
    },
    "olimpbet": {
        "name": "ОлимпБет",
        "base_url": "https://www.olimpbet.ru",
        "api_url": "https://www.olimpbet.ru/api",
        "enabled": True,
    },
    "ligastavok": {
        "name": "Лига Ставок",
        "base_url": "https://www.ligastavok.ru",
        "api_url": "https://www.ligastavok.ru/api",
        "enabled": True,
    },
    "parimatch": {
        "name": "Пари Матч",
        "base_url": "https://www.parimatch.ru",
        "api_url": "https://www.parimatch.ru/api",
        "enabled": True,
    },
    "1xstavka": {
        "name": "1хСтавка",
        "base_url": "https://1xstavka.ru",
        "api_url": "https://1xstavka.ru/LineFeed/GetChampsZip",
        "enabled": True,
    },
}

# ==================== РЫНКИ СТАВОК ====================
MARKETS = {
    "yellow_cards": {
        "name": "Желтые карточки",
        "description": "Тотал желтых карточек в матче",
        "types": ["total_over", "total_under", "team_total", "both_teams"],
    },
    "corners": {
        "name": "Угловые",
        "description": "Тотал угловых в матче",
        "types": ["total_over", "total_under", "team_total", "handicap"],
    },
    "medical": {
        "name": "Выход медицинской бригады",
        "description": "Количество выходов медбригады (замены из-за травм)",
        "types": ["total_over", "total_under"],
    },
    "penalties": {
        "name": "Пенальти",
        "description": "Будет ли пенальти в матче",
        "types": ["yes_no", "total_over", "total_under"],
    },
}

# ==================== ИСТОЧНИКИ СТАТИСТИКИ ====================
STATS_SOURCES = {
    "fbref": {
        "name": "FBref",
        "base_url": "https://fbref.com",
        "rpl_url": "https://fbref.com/en/comps/30/Russian-Premier-League-Stats",
        "enabled": True,
    },
    "understat": {
        "name": "Understat",
        "base_url": "https://understat.com",
        "rpl_url": "https://understat.com/league/RFPL",
        "enabled": True,
    },
    "transfermarkt": {
        "name": "Transfermarkt",
        "base_url": "https://www.transfermarkt.ru",
        "rpl_url": "https://www.transfermarkt.ru/premier-liga/startseite/wettbewerb/RU1",
        "enabled": True,
    },
    "football_data": {
        "name": "Football-Data",
        "base_url": "https://www.football-data.co.uk",
        "rpl_url": "https://www.football-data.co.uk/russiaE.php",
        "enabled": True,
    },
    "soccerway": {
        "name": "Soccerway",
        "base_url": "https://int.soccerway.com",
        "enabled": True,
    },
}

# ==================== ПАРАМЕТРЫ МОДЕЛИ ====================
MODEL_PARAMS = {
    # Окно исторических данных (матчей)
    "history_window": 20,
    # Окно формы команды
    "form_window": 5,
    # Минимальная вероятность для рекомендации
    "min_value_threshold": 0.05,  # 5% преимущество над букмекером
    # Веса факторов
    "weights": {
        "team_form": 0.25,
        "h2h_stats": 0.20,
        "referee_stats": 0.20,
        "home_away": 0.15,
        "season_stats": 0.20,
    },
    # Параметры модели Пуассона
    "poisson_max_events": 15,
    # Bankroll management
    "kelly_fraction": 0.25,  # Четверть Келли
    "max_bet_fraction": 0.05,  # Максимум 5% банкролла
}

# ==================== СЕЗОНЫ ====================
SEASONS = {
    "current": "2024-25",
    "available": ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"],
}

# ==================== HEADERS ДЛЯ ЗАПРОСОВ ====================
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

REQUEST_TIMEOUT = 30
REQUEST_DELAY = 1.5  # секунды между запросами

# ==================== КОМАНДЫ РПЛ ====================
RPL_TEAMS = {
    "Зенит": {"city": "Санкт-Петербург", "alias": ["Zenit", "Зенит СПб"]},
    "ЦСКА": {"city": "Москва", "alias": ["CSKA", "ПФК ЦСКА"]},
    "Спартак": {"city": "Москва", "alias": ["Spartak", "Спартак М"]},
    "Локомотив": {"city": "Москва", "alias": ["Lokomotiv", "Локомотив М"]},
    "Краснодар": {"city": "Краснодар", "alias": ["Krasnodar", "ФК Краснодар"]},
    "Динамо": {"city": "Москва", "alias": ["Dynamo", "Динамо М"]},
    "Рубин": {"city": "Казань", "alias": ["Rubin", "Рубин Казань"]},
    "Ростов": {"city": "Ростов-на-Дону", "alias": ["Rostov", "ФК Ростов"]},
    "Факел": {"city": "Воронеж", "alias": ["Fakel", "Факел В"]},
    "Оренбург": {"city": "Оренбург", "alias": ["Orenburg", "ФК Оренбург"]},
    "Химки": {"city": "Химки", "alias": ["Khimki", "ФК Химки"]},
    "Ахмат": {"city": "Грозный", "alias": ["Akhmat", "Ахмат Грозный"]},
    "Сочи": {"city": "Сочи", "alias": ["Sochi", "ФК Сочи"]},
    "Пари НН": {"city": "Нижний Новгород", "alias": ["Pari NN", "Нижний Новгород"]},
    "Урал": {"city": "Екатеринбург", "alias": ["Ural", "ФК Урал"]},
    "Крылья Советов": {"city": "Самара", "alias": ["Krylia", "Крылья"]},
}

# ==================== СУДЬИ ====================
# Данные о судьях будут загружаться динамически
REFEREE_STATS_FIELDS = [
    "avg_yellow_cards",
    "avg_red_cards",
    "avg_fouls",
    "avg_penalties",
    "strictness_index",
    "home_bias",
    "matches_count",
]

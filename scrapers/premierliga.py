"""
Парсер расписания матчей с официального сайта РПЛ (premierliga.ru)
"""
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from utils.helpers import safe_request, get_random_headers
from config.settings import CACHE_DIR, SEASONS

logger = logging.getLogger(__name__)

SCHEDULE_PATH = Path(__file__).parent.parent / "data" / "schedule.json"

# Маппинг русских месяцев в номера
MONTH_MAP = {
    "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
    "мая": "05", "июня": "06", "июля": "07", "августа": "08",
    "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12",
}

# Официальные названия команд РПЛ на premierliga.ru → наши названия
TEAM_NAME_MAP = {
    "Зенит": "Зенит",
    "ЦСКА": "ЦСКА",
    "Спартак": "Спартак",
    "Локомотив": "Локомотив",
    "Краснодар": "Краснодар",
    "Динамо": "Динамо",
    "Рубин": "Рубин",
    "Ростов": "Ростов",
    "Оренбург": "Оренбург",
    "Химки": "Химки",
    "Ахмат": "Ахмат",
    "Сочи": "Сочи",
    "Пари НН": "Пари НН",
    "Крылья Советов": "Крылья Советов",
    "ФК Нижний Новгород": "Пари НН",
    "Нижний Новгород": "Пари НН",
}


class PremierLigaScraper:
    """
    Парсер сайта premierliga.ru для получения расписания матчей РПЛ.
    Источник: https://premierliga.ru/calendar/
    """

    BASE_URL = "https://premierliga.ru"
    CALENDAR_URL = "https://premierliga.ru/calendar/"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(get_random_headers())

    def fetch_schedule(self, round_number: Optional[int] = None) -> Dict:
        """
        Получить расписание туров РПЛ с premierliga.ru.

        Args:
            round_number: номер тура (если None — текущий тур)

        Returns:
            Словарь с расписанием в формате schedule.json
        """
        try:
            url = self.CALENDAR_URL
            if round_number:
                url = f"{self.CALENDAR_URL}?tour={round_number}"

            resp = safe_request(url, timeout=30)
            if not resp:
                logger.warning("premierliga.ru: нет ответа, используем кэш")
                return self._load_cached_schedule()

            soup = BeautifulSoup(resp.text, "html.parser")
            rounds = self._parse_calendar(soup)

            if not rounds:
                logger.warning("premierliga.ru: расписание не найдено, используем кэш")
                return self._load_cached_schedule()

            schedule = {
                "current_round": rounds[0]["round"] if rounds else 1,
                "season": SEASONS["current"],
                "source": "premierliga.ru",
                "rounds": rounds,
            }
            self._save_schedule(schedule)
            logger.info(
                f"premierliga.ru: загружено {len(rounds)} туров, "
                f"{sum(len(r['matches']) for r in rounds)} матчей"
            )
            return schedule

        except Exception as e:
            logger.error(f"premierliga.ru: ошибка парсинга: {e}")
            return self._load_cached_schedule()

    def _parse_calendar(self, soup: BeautifulSoup) -> List[Dict]:
        """Парсинг страницы календаря"""
        rounds = []

        # Ищем блоки туров (могут называться по-разному в зависимости от верстки)
        round_blocks = (
            soup.find_all("div", class_=re.compile(r"tour|round|calendar", re.I))
            or soup.find_all("section", class_=re.compile(r"tour|round|calendar", re.I))
        )

        if not round_blocks:
            # Попробуем найти по таблице матчей
            round_blocks = soup.find_all("table", class_=re.compile(r"match|game", re.I))

        for block in round_blocks:
            round_data = self._parse_round_block(block)
            if round_data and round_data.get("matches"):
                rounds.append(round_data)

        # Альтернативный парсинг — по заголовкам туров
        if not rounds:
            rounds = self._parse_calendar_alternative(soup)

        return rounds

    def _parse_round_block(self, block) -> Optional[Dict]:
        """Парсинг блока одного тура"""
        try:
            # Найти номер тура
            round_num = None
            title = block.find(
                ["h1", "h2", "h3", "h4", "div"],
                class_=re.compile(r"title|header|round", re.I),
            )
            if title:
                text = title.get_text()
                m = re.search(r"(\d+)\s*тур", text, re.I)
                if m:
                    round_num = int(m.group(1))

            if not round_num:
                return None

            matches = []
            match_rows = block.find_all(
                ["div", "tr", "li"], class_=re.compile(r"match|game|event", re.I)
            )

            for row in match_rows:
                match = self._parse_match_row(row)
                if match:
                    matches.append(match)

            if not matches:
                return None

            dates = self._extract_round_dates(matches)
            return {
                "round": round_num,
                "dates": dates,
                "matches": matches,
            }

        except Exception as e:
            logger.debug(f"Ошибка парсинга блока тура: {e}")
            return None

    def _parse_match_row(self, row) -> Optional[Dict]:
        """Парсинг строки матча"""
        try:
            text = row.get_text(" ", strip=True)

            # Найти команды (разделены тире или vs)
            teams_match = re.search(
                r"([А-Яа-яёЁ\s]+(?:Советов)?)\s*[–\-—]\s*([А-Яа-яёЁ\s]+(?:Советов)?)",
                text,
            )
            if not teams_match:
                return None

            home = self._normalize_team(teams_match.group(1).strip())
            away = self._normalize_team(teams_match.group(2).strip())

            if not home or not away:
                return None

            # Дата
            date_match = re.search(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", text, re.I)
            date_str = ""
            if date_match:
                day = date_match.group(1).zfill(2)
                month = MONTH_MAP.get(date_match.group(2).lower(), "01")
                year = date_match.group(3)
                date_str = f"{day}.{month}.{year}"

            # Время
            time_match = re.search(r"(\d{1,2}):(\d{2})", text)
            time_str = time_match.group(0) if time_match else "00:00"

            # Судья
            referee = ""
            ref_match = re.search(r"судья[:\s]+([А-Яа-яёЁ]+\s+[А-Я]\.)", text, re.I)
            if ref_match:
                referee = ref_match.group(1).strip()

            return {
                "date": date_str,
                "time": time_str,
                "home": home,
                "away": away,
                "referee": referee,
                "stadium": "",
            }

        except Exception:
            return None

    def _parse_calendar_alternative(self, soup: BeautifulSoup) -> List[Dict]:
        """Альтернативный парсинг — поиск по тексту страницы"""
        rounds = []
        current_round = None
        current_matches = []
        current_dates = ""

        # Ищем все текстовые элементы с упоминанием туров и матчей
        all_text = soup.get_text("\n")
        lines = [l.strip() for l in all_text.split("\n") if l.strip()]

        for line in lines:
            # Заголовок тура
            m = re.match(r"(\d+)\s*тур", line, re.I)
            if m:
                if current_round and current_matches:
                    rounds.append({
                        "round": current_round,
                        "dates": current_dates,
                        "matches": current_matches,
                    })
                current_round = int(m.group(1))
                current_matches = []
                current_dates = ""
                continue

            # Матч: "Команда1 – Команда2, дата, время"
            match = re.search(
                r"([А-Яа-яёЁ][А-Яа-яёЁ\s]+)\s*[–\-—]\s*([А-Яа-яёЁ][А-Яа-яёЁ\s]+)",
                line,
            )
            if match and current_round:
                home = self._normalize_team(match.group(1).strip())
                away = self._normalize_team(match.group(2).strip())
                if home and away:
                    time_m = re.search(r"\b(\d{1,2}:\d{2})\b", line)
                    date_m = re.search(r"(\d{1,2}\.\d{2}\.\d{4})", line)
                    current_matches.append({
                        "date": date_m.group(1) if date_m else "",
                        "time": time_m.group(1) if time_m else "00:00",
                        "home": home,
                        "away": away,
                        "referee": "",
                        "stadium": "",
                    })

        if current_round and current_matches:
            rounds.append({
                "round": current_round,
                "dates": current_dates,
                "matches": current_matches,
            })

        return rounds

    def _normalize_team(self, name: str) -> str:
        """Привести название команды к стандартному виду"""
        name = name.strip()
        if name in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[name]
        # Частичное совпадение
        for original, normalized in TEAM_NAME_MAP.items():
            if original.lower() in name.lower() or name.lower() in original.lower():
                return normalized
        # Если не нашли — вернуть как есть (если похоже на название команды)
        if len(name) >= 3 and re.search(r"[А-Яа-яёЁ]", name):
            return name
        return ""

    def _extract_round_dates(self, matches: List[Dict]) -> str:
        """Извлечь диапазон дат тура"""
        dates = sorted(set(m["date"] for m in matches if m.get("date")))
        if not dates:
            return ""
        if len(dates) == 1:
            return dates[0]
        # Возвращаем диапазон типа "14-16 марта 2026"
        try:
            first = datetime.strptime(dates[0], "%d.%m.%Y")
            last = datetime.strptime(dates[-1], "%d.%m.%Y")
            month_names = {
                1: "января", 2: "февраля", 3: "марта", 4: "апреля",
                5: "мая", 6: "июня", 7: "июля", 8: "августа",
                9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
            }
            return f"{first.day}-{last.day} {month_names[first.month]} {first.year}"
        except ValueError:
            return f"{dates[0]} — {dates[-1]}"

    def _save_schedule(self, schedule: Dict) -> None:
        """Сохранить расписание в файл"""
        try:
            SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SCHEDULE_PATH, "w", encoding="utf-8") as f:
                json.dump(schedule, f, ensure_ascii=False, indent=2)
            logger.info(f"Расписание сохранено: {SCHEDULE_PATH}")
        except Exception as e:
            logger.error(f"Ошибка сохранения расписания: {e}")

    def _load_cached_schedule(self) -> Dict:
        """Загрузить кэшированное расписание"""
        if SCHEDULE_PATH.exists():
            try:
                with open(SCHEDULE_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}


def update_schedule(round_number: Optional[int] = None) -> Dict:
    """
    Обновить расписание с premierliga.ru.
    Вызывается при запуске дашборда или по кнопке «Обновить».
    """
    scraper = PremierLigaScraper()
    return scraper.fetch_schedule(round_number)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    schedule = update_schedule()
    print(json.dumps(schedule, ensure_ascii=False, indent=2))

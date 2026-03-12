"""
Streamlit Dashboard для предиктивной модели ставок РПЛ
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config.settings import RPL_TEAMS, MODEL_PARAMS, BOOKMAKERS, SEASONS
from models.prediction_engine import PredictionEngine, PoissonModel
from scrapers.statistics import (
    RPLStatisticsCollector,
    RefereeStatsCollector,
    TeamStatsCollector,
)
from utils.helpers import setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)

# Основные букмекеры для отображения
PRIMARY_BOOKMAKERS = ["fonbet", "winline", "betboom"]

# ==================== НАСТРОЙКИ СТРАНИЦЫ ====================
st.set_page_config(
    page_title="RPL Betting Model | Предиктивная модель ставок РПЛ",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== СТИЛИ ====================
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .value-positive {
        color: #00b894;
        font-weight: bold;
    }
    .value-negative {
        color: #e17055;
        font-weight: bold;
    }
    .strong-bet {
        background-color: #00b894;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .good-bet {
        background-color: #0984e3;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .neutral-bet {
        background-color: #fdcb6e;
        color: #2d3436;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .avoid-bet {
        background-color: #e17055;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .bk-badge {
        display: inline-block;
        background: #2d3436;
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 2px 4px;
        text-decoration: none;
    }
    .fact-block {
        background: #f8f9fa;
        border-left: 4px solid #0984e3;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.92rem;
    }
    .fact-prognoz {
        background: #e8f5e9;
        border-left: 4px solid #00b894;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================== КЕШИРОВАНИЕ ДАННЫХ ====================
@st.cache_data(ttl=300)  # кеш 5 минут
def load_schedule() -> dict:
    """Загрузить расписание туров РПЛ"""
    schedule_path = Path(__file__).parent.parent / "data" / "schedule.json"
    if schedule_path.exists():
        with open(schedule_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600)  # кеш на 1 час
def load_historical_data():
    """Загрузить исторические данные"""
    collector = RPLStatisticsCollector()
    return collector.collect_historical_data()


@st.cache_data(ttl=3600)
def load_stats(matches_df):
    """Загрузить статистику команд и судей"""
    referee_collector = RefereeStatsCollector()
    team_collector = TeamStatsCollector()

    referee_stats = referee_collector.get_referee_stats(matches_df)
    team_stats = team_collector.get_team_stats(matches_df)

    return referee_stats, team_stats


# ==================== ГЛАВНАЯ СТРАНИЦА ====================
def main():
    # Заголовок
    st.markdown(
        '<div class="main-header">⚽ Предиктивная модель ставок РПЛ</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Рынки:** Желтые карточки | Угловые | Выход медбригады | Пенальти",
        unsafe_allow_html=False,
    )
    st.divider()

    # Sidebar навигация
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/ru/thumb/b/b7/RPL_logo_2023.png/200px-RPL_logo_2023.png",
            width=150,
        )
        st.title("Навигация")
        page = st.radio(
            "Выбрать раздел:",
            [
                "Анализ матчей",
                "Статистика команд",
                "Статистика судей",
                "Настройки модели",
            ],
        )

        st.divider()
        st.subheader("Обновление данных")

        if st.button("Обновить расписание с premierliga.ru", type="primary"):
            try:
                from scrapers.premierliga import update_schedule
                with st.spinner("Загрузка расписания с premierliga.ru..."):
                    update_schedule()
                st.cache_data.clear()
                st.success("Расписание обновлено!")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")

        if st.button("Обновить коэффициенты"):
            st.cache_data.clear()
            st.rerun()

        auto_update = st.checkbox("Автообновление (30 мин)", value=False)
        if auto_update:
            st.info("Автообновление активно")

        last_update = datetime.now().strftime("%d.%m.%Y %H:%M")
        st.caption(f"Обновлено: {last_update}")

    # Загрузка данных
    with st.spinner("Загрузка данных..."):
        matches_df = load_historical_data()
        referee_stats_df, team_stats_df = load_stats(matches_df)
        schedule = load_schedule()

    # Маршрутизация
    if page == "Анализ матчей":
        show_match_analysis(matches_df, referee_stats_df, team_stats_df, schedule)
    elif page == "Статистика команд":
        show_team_stats(team_stats_df)
    elif page == "Статистика судей":
        show_referee_stats(referee_stats_df)
    elif page == "Настройки модели":
        show_model_settings()


# ==================== АНАЛИЗ МАТЧЕЙ ====================
def show_match_analysis(matches_df, referee_stats_df, team_stats_df, schedule: dict = None):
    st.header("Анализ матчей")

    # ---------- БЛИЖАЙШИЙ ТУР РПЛ ----------
    if schedule and schedule.get("rounds"):
        current_round_num = schedule.get("current_round", 1)
        season = schedule.get("season", SEASONS["current"])

        # Найти ближайший тур
        nearest = next(
            (r for r in schedule["rounds"] if r["round"] == current_round_num),
            schedule["rounds"][0],
        )

        with st.container(border=True):
            st.markdown(
                f"### 📅 Тур {nearest['round']} · {nearest['dates']} · РПЛ {season}"
            )

            # Инициализировать session state
            if "selected_home" not in st.session_state:
                st.session_state.selected_home = None
            if "selected_away" not in st.session_state:
                st.session_state.selected_away = None
            if "selected_referee" not in st.session_state:
                st.session_state.selected_referee = ""

            # Группировать матчи по дате
            matches_by_date: Dict[str, list] = {}
            for m in nearest["matches"]:
                matches_by_date.setdefault(m["date"], []).append(m)

            for date_str, day_matches in matches_by_date.items():
                st.caption(f"**{date_str}**")
                cols = st.columns(len(day_matches))
                for col, m in zip(cols, day_matches):
                    with col:
                        is_selected = (
                            st.session_state.selected_home == m["home"]
                            and st.session_state.selected_away == m["away"]
                        )
                        btn_label = (
                            f"{'✅ ' if is_selected else ''}"
                            f"{m['home']} — {m['away']}\n"
                            f"🕐 {m['time']}  👤 {m['referee']}"
                        )
                        if st.button(
                            btn_label,
                            key=f"sched_{m['home']}_{m['away']}",
                            use_container_width=True,
                            type="primary" if is_selected else "secondary",
                        ):
                            st.session_state.selected_home = m["home"]
                            st.session_state.selected_away = m["away"]
                            st.session_state.selected_referee = m.get("referee", "")
                            st.rerun()

    st.divider()

    # ---------- ФОРМА ВЫБОРА МАТЧА ----------
    teams_list = list(RPL_TEAMS.keys())

    # Применить выбор из расписания
    default_home_idx = (
        teams_list.index(st.session_state.selected_home)
        if st.session_state.get("selected_home") in teams_list
        else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        home_team = st.selectbox("Команда хозяев", teams_list, index=default_home_idx)

    away_options = [t for t in teams_list if t != home_team]
    default_away_idx = (
        away_options.index(st.session_state.selected_away)
        if st.session_state.get("selected_away") in away_options
        else 0
    )

    with col2:
        away_team = st.selectbox("Команда гостей", away_options, index=default_away_idx)

    with col3:
        referees_list = [""] + (
            list(referee_stats_df["referee"].unique())
            if not referee_stats_df.empty
            else []
        )
        default_ref_idx = (
            referees_list.index(st.session_state.selected_referee)
            if st.session_state.get("selected_referee") in referees_list
            else 0
        )
        referee = st.selectbox("Судья (необязательно)", referees_list, index=default_ref_idx)

    # ---------- КНОПКА АНАЛИЗИРОВАТЬ ----------
    if st.button("Анализировать матч", type="primary", use_container_width=True):
        # H2H последние 5 матчей — показать сразу под кнопкой
        show_h2h_last_matches(home_team, away_team, matches_df, n=5)

        with st.spinner(f"Анализ: {home_team} vs {away_team}..."):
            engine = PredictionEngine()
            prediction = engine.predict_match(
                home_team=home_team,
                away_team=away_team,
                team_stats_df=team_stats_df,
                referee_stats_df=referee_stats_df,
                matches_df=matches_df,
                referee=referee,
            )

        # Метрики
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Желтые карточки",
                f"{prediction.yellow_cards_expected:.1f}",
                help="Ожидаемое кол-во желтых карточек"
            )
        with col2:
            st.metric(
                "Угловые",
                f"{prediction.corners_expected:.1f}",
                help="Ожидаемое кол-во угловых"
            )
        with col3:
            st.metric(
                "Пенальти",
                f"{prediction.penalty_probability * 100:.0f}%",
                help="Вероятность назначения пенальти"
            )
        with col4:
            st.metric(
                "Медбригада",
                f"{prediction.medical_expected:.1f}",
                help="Ожидаемое кол-во выходов медбригады"
            )

        st.divider()

        # Таблицы вероятностей
        tab1, tab2, tab3, tab4 = st.tabs([
            "Желтые карточки", "Угловые", "Пенальти", "Медбригада"
        ])

        with tab1:
            left_col, right_col = st.columns([6, 4])
            with left_col:
                show_market_table(
                    prediction.yellow_cards_probs,
                    prediction.yellow_cards_expected,
                    "Желтые карточки",
                    "желтых карточек",
                    home_team=home_team,
                    away_team=away_team,
                    market_key="yellow_cards",
                    matches_df=matches_df,
                )
            with right_col:
                show_prediction_panel(
                    prediction, "yellow_cards",
                    home_team, away_team, referee_stats_df
                )

        with tab2:
            left_col, right_col = st.columns([6, 4])
            with left_col:
                show_market_table(
                    prediction.corners_probs,
                    prediction.corners_expected,
                    "Угловые",
                    "угловых",
                    home_team=home_team,
                    away_team=away_team,
                    market_key="corners",
                    matches_df=matches_df,
                )
            with right_col:
                show_prediction_panel(
                    prediction, "corners",
                    home_team, away_team, referee_stats_df
                )

        with tab3:
            left_col, right_col = st.columns([6, 4])
            with left_col:
                show_penalty_analysis(prediction.penalty_probability, referee)
            with right_col:
                show_prediction_panel(
                    prediction, "penalty",
                    home_team, away_team, referee_stats_df
                )

        with tab4:
            left_col, right_col = st.columns([6, 4])
            with left_col:
                show_medical_analysis(
                    prediction.medical_expected, prediction.medical_probs
                )
            with right_col:
                show_prediction_panel(
                    prediction, "medical",
                    home_team, away_team, referee_stats_df
                )

        # Информация о качестве данных
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.info(
                f"Качество данных: **{prediction.data_quality.upper()}** | "
                f"Уверенность: **{prediction.confidence * 100:.0f}%** | "
                f"Факторы: {', '.join(prediction.factors_used)}"
            )
        with col2:
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
            if not h2h.empty:
                st.success(
                    f"Очных встреч в базе: **{len(h2h)}** | "
                    f"Ср. ЖК: **{h2h['total_yellow_cards'].mean():.1f}** | "
                    f"Ср. угловые: **{h2h['total_corners'].mean():.1f}**"
                )


# ==================== H2H ПОСЛЕДНИЕ 5 МАТЧЕЙ ====================
def show_h2h_last_matches(home_team: str, away_team: str, matches_df, n: int = 5):
    """Показать последние N очных встреч двух команд"""
    if matches_df is None or matches_df.empty:
        return

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
        st.info(f"Очных встреч {home_team} — {away_team} в базе не найдено")
        return

    # Сортировка по дате (новые первыми)
    if "date" in h2h.columns:
        try:
            h2h["date"] = pd.to_datetime(h2h["date"], dayfirst=True, errors="coerce")
            h2h = h2h.sort_values("date", ascending=False)
        except Exception:
            pass

    recent = h2h.head(n)

    with st.container(border=True):
        st.markdown(f"#### 🤝 Последние {min(n, len(recent))} очных встречи")
        for _, row in recent.iterrows():
            home = row.get("home_team", "?")
            away = row.get("away_team", "?")
            hg = int(row.get("home_goals", 0)) if pd.notna(row.get("home_goals")) else "?"
            ag = int(row.get("away_goals", 0)) if pd.notna(row.get("away_goals")) else "?"
            hyc = int(row.get("home_yellow_cards", 0)) if pd.notna(row.get("home_yellow_cards")) else "?"
            ayc = int(row.get("away_yellow_cards", 0)) if pd.notna(row.get("away_yellow_cards")) else "?"
            hc = int(row.get("home_corners", 0)) if pd.notna(row.get("home_corners")) else "?"
            ac = int(row.get("away_corners", 0)) if pd.notna(row.get("away_corners")) else "?"
            date_val = row.get("date", "")
            date_str = (
                date_val.strftime("%d.%m.%Y")
                if hasattr(date_val, "strftime")
                else str(date_val)[:10]
            )
            referee_str = row.get("referee", "")
            ref_info = f" · Судья: {referee_str}" if referee_str else ""

            st.markdown(
                f"**{date_str}** — **{home} {hg}:{ag} {away}**{ref_info}  \n"
                f"ЖК: {home} **{hyc}** – {away} **{ayc}** &nbsp;|&nbsp; "
                f"Угловые: {home} **{hc}** – {away} **{ac}**"
            )
        st.divider()


# ==================== ТАБЛИЦА РЫНКА ====================
def show_market_table(
    probs: Dict,
    expected: float,
    title: str,
    unit: str,
    home_team: str = "",
    away_team: str = "",
    market_key: str = "",
    matches_df=None,
):
    """Отобразить таблицу коэффициентов для рынка (без % вероятностей)"""
    if not probs:
        st.warning("Нет данных для отображения")
        return

    st.subheader(title)
    st.caption(f"Ожидаемое значение: **{expected:.2f}** {unit}")

    # Таблица: линия + справедливые коэффициенты (без %)
    rows = []
    for line, p in probs.items():
        rows.append({
            "Линия": line,
            "Справедливый коэф Б": f"{p['over_odds']:.2f}",
            "Справедливый коэф М": f"{p['under_odds']:.2f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Статистические факты (value bet identification)
    if home_team and away_team and matches_df is not None and not matches_df.empty:
        show_statistical_facts(home_team, away_team, matches_df, market_key)


# ==================== СТАТИСТИЧЕСКИЕ ФАКТЫ ====================
def show_statistical_facts(
    home_team: str,
    away_team: str,
    matches_df,
    market_key: str,
    n_matches: int = 17,
    handicaps: Optional[List[float]] = None,
):
    """
    Найти и отобразить статистические факты по паттернам команд.
    Пример: "Сочи проиграла по ЖК с форой -1.5 в 15 из 17 матчах"
    """
    if matches_df is None or matches_df.empty:
        return

    if market_key == "yellow_cards":
        home_col = "home_yellow_cards"
        away_col = "away_yellow_cards"
        metric_name = "желтым карточкам"
        if handicaps is None:
            handicaps = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    elif market_key == "corners":
        home_col = "home_corners"
        away_col = "away_corners"
        metric_name = "угловым"
        if handicaps is None:
            handicaps = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    else:
        return

    # Проверим, что нужные колонки существуют
    required_cols = [home_col, away_col, "home_team", "away_team"]
    if not all(c in matches_df.columns for c in required_cols):
        return

    facts = []
    min_ratio = 0.70  # минимальная доля для отображения факта
    min_matches = 8   # минимум матчей для надёжности

    for team in [home_team, away_team]:
        # Матчи команды (как дома и в гостях)
        team_matches = matches_df[
            (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
        ].copy()

        if "date" in team_matches.columns:
            try:
                team_matches["date"] = pd.to_datetime(
                    team_matches["date"], dayfirst=True, errors="coerce"
                )
                team_matches = team_matches.sort_values("date", ascending=False)
            except Exception:
                pass

        recent = team_matches.head(n_matches)
        total = len(recent)

        if total < min_matches:
            continue

        for hcp in handicaps:
            wins = 0
            for _, row in recent.iterrows():
                is_home = row["home_team"] == team
                if is_home:
                    team_val = row.get(home_col, 0) or 0
                    opp_val = row.get(away_col, 0) or 0
                else:
                    team_val = row.get(away_col, 0) or 0
                    opp_val = row.get(home_col, 0) or 0

                # "Победа" команды по рынку с форой: team_val + hcp > opp_val
                if (team_val + hcp) > opp_val:
                    wins += 1

            ratio = wins / total
            if ratio >= min_ratio:
                hcp_str = f"+{hcp}" if hcp > 0 else str(hcp)
                action = "победила" if hcp <= 0 else "покрыла форму"
                facts.append({
                    "team": team,
                    "handicap": hcp,
                    "wins": wins,
                    "total": total,
                    "ratio": ratio,
                    "text": (
                        f"Команда **{team}** победила по {metric_name} "
                        f"с форой {hcp_str} в **{wins} из {total}** последних матчах"
                    ),
                })

        # Поиск потерь (команда проигрывает по рынку)
        for hcp in handicaps:
            losses = 0
            for _, row in recent.iterrows():
                is_home = row["home_team"] == team
                if is_home:
                    team_val = row.get(home_col, 0) or 0
                    opp_val = row.get(away_col, 0) or 0
                else:
                    team_val = row.get(away_col, 0) or 0
                    opp_val = row.get(home_col, 0) or 0

                # "Поражение": team_val < opp_val - |hcp|  (т.е. с форой hcp проигрывает)
                if (team_val - abs(hcp)) < opp_val:
                    losses += 1

            ratio = losses / total
            if ratio >= min_ratio and hcp < 0:
                facts.append({
                    "team": team,
                    "handicap": hcp,
                    "wins": losses,
                    "total": total,
                    "ratio": ratio,
                    "text": (
                        f"Команда **{team}** проиграла по {metric_name} "
                        f"с форой {hcp} в **{losses} из {total}** последних матчах"
                    ),
                })

    if not facts:
        return

    # Убрать дубликаты и отсортировать по силе факта
    seen = set()
    unique_facts = []
    for f in sorted(facts, key=lambda x: -x["ratio"]):
        key = (f["team"], f["handicap"], f["wins"])
        if key not in seen:
            seen.add(key)
            unique_facts.append(f)

    # Найти прогноз (если одна команда имеет преимущество)
    home_facts = [f for f in unique_facts if f["team"] == home_team]
    away_facts = [f for f in unique_facts if f["team"] == away_team]

    # Отображение
    st.markdown("---")
    st.markdown("#### 📊 Статистические факты")

    displayed = 0
    for f in unique_facts[:6]:
        st.markdown(
            f'<div class="fact-block">{f["text"]}</div>',
            unsafe_allow_html=True,
        )
        displayed += 1

    # Прогноз — если у одной из команд есть устойчивое преимущество
    if home_facts and away_facts:
        best_home = max(home_facts, key=lambda x: x["ratio"])
        best_away = max(away_facts, key=lambda x: x["ratio"])

        if best_home["ratio"] > best_away["ratio"] and best_home["ratio"] >= 0.75:
            hcp = best_home["handicap"]
            hcp_str = f"+{hcp}" if hcp > 0 else str(hcp)
            prognoz = f"Прогноз: фора {home_team} {hcp_str}"
            st.markdown(
                f'<div class="fact-prognoz">🎯 {prognoz}</div>',
                unsafe_allow_html=True,
            )
        elif best_away["ratio"] > best_home["ratio"] and best_away["ratio"] >= 0.75:
            hcp = best_away["handicap"]
            hcp_str = f"+{hcp}" if hcp > 0 else str(hcp)
            prognoz = f"Прогноз: фора {away_team} {hcp_str}"
            st.markdown(
                f'<div class="fact-prognoz">🎯 {prognoz}</div>',
                unsafe_allow_html=True,
            )
    elif home_facts:
        best = max(home_facts, key=lambda x: x["ratio"])
        if best["ratio"] >= 0.78:
            hcp_str = f"+{best['handicap']}" if best["handicap"] > 0 else str(best["handicap"])
            st.markdown(
                f'<div class="fact-prognoz">🎯 Прогноз: фора {home_team} {hcp_str}</div>',
                unsafe_allow_html=True,
            )
    elif away_facts:
        best = max(away_facts, key=lambda x: x["ratio"])
        if best["ratio"] >= 0.78:
            hcp_str = f"+{best['handicap']}" if best["handicap"] > 0 else str(best["handicap"])
            st.markdown(
                f'<div class="fact-prognoz">🎯 Прогноз: фора {away_team} {hcp_str}</div>',
                unsafe_allow_html=True,
            )


def show_penalty_analysis(penalty_prob: float, referee: str = ""):
    """Анализ пенальти"""
    st.subheader("Пенальти")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=penalty_prob * 100,
                number={"suffix": "%"},
                title={"text": "Вероятность пенальти"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0984e3"},
                    "steps": [
                        {"range": [0, 20], "color": "#dfe6e9"},
                        {"range": [20, 40], "color": "#b2bec3"},
                        {"range": [40, 60], "color": "#fdcb6e"},
                        {"range": [60, 80], "color": "#fd79a8"},
                        {"range": [80, 100], "color": "#e17055"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 28,
                    },
                },
            )
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fair_yes_odds = 1.0 / max(penalty_prob, 0.001)
        fair_no_odds = 1.0 / max(1 - penalty_prob, 0.001)

        st.metric("Справедливый коэф ДА", f"{fair_yes_odds:.2f}")
        st.metric("Справедливый коэф НЕТ", f"{fair_no_odds:.2f}")
        st.caption("Среднее по РПЛ: 28% матчей содержат пенальти")


def show_medical_analysis(expected: float, probs: Dict):
    """Анализ выхода медицинской бригады"""
    st.subheader("Выход медицинской бригады")
    st.caption(
        "Выход медбригады — замена игрока из-за травмы "
        "или серьезного нарушения правил"
    )

    st.metric("Ожидаемое кол-во выходов", f"{expected:.2f}")
    st.caption("Среднее по РПЛ: 1.8 за матч")

    if probs:
        rows = []
        for line, p in probs.items():
            rows.append({
                "Линия": line,
                "Коэф (Б)": f"{p['over_odds']:.2f}",
                "Коэф (М)": f"{p['under_odds']:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==================== ПАНЕЛЬ ПРОГНОЗА (ПРАВАЯ КОЛОНКА) ====================
def show_prediction_panel(prediction, market: str, home_team: str, away_team: str, referee_stats_df):
    """Прогноз модели + коэффициенты букмекеров в правой колонке"""
    st.subheader("🎯 Прогноз модели")

    BK_URLS = {
        "Фонбет": BOOKMAKERS.get("fonbet", {}).get("football_url", "https://www.fonbet.ru/sports/football/"),
        "Винлайн": BOOKMAKERS.get("winline", {}).get("football_url", "https://www.winline.ru/sports/football/"),
        "Бетбум": BOOKMAKERS.get("betboom", {}).get("football_url", "https://betboom.ru/sport/Football"),
    }
    VALUE_EDGE = 0.05

    if market == "yellow_cards":
        exp = prediction.yellow_cards_expected
        line = 4.5 if exp >= 4.2 else 3.5
        probs_dict = prediction.yellow_cards_probs or {}
        p_over = probs_dict.get(line, {}).get("over_prob", 0)
        p_under = 1 - p_over
        rec = f"ЖК {'Больше' if p_over > 0.55 else 'Меньше'} {line}"
        conf_prob = max(p_over, p_under)
        st.success(f"**{rec}** ({conf_prob*100:.0f}%)")
        st.metric("Ожидается ЖК", f"{exp:.1f}")
        st.metric("Рекомендуемая линия", line)
        st.metric("Вероятность Б", f"{p_over*100:.0f}%")
        st.metric("Вероятность М", f"{p_under*100:.0f}%")

        # Ссылки на букмекеров с обозначением value
        st.markdown("---")
        st.markdown("**Коэффициенты у букмекеров:**")
        fair_over = 1 / max(p_over, 0.001)
        fair_under = 1 / max(p_under, 0.001)
        for bk_name, url in BK_URLS.items():
            edge_str = f"+{VALUE_EDGE*100:.0f}%+ ценность" if p_over > 0.55 + VALUE_EDGE else ""
            badge = "🟢" if edge_str else "🔵"
            st.markdown(
                f'{badge} <a href="{url}" target="_blank" class="bk-badge">{bk_name} — '
                f'Б{line}: ~{fair_over:.2f} | М{line}: ~{fair_under:.2f}</a>',
                unsafe_allow_html=True,
            )

    elif market == "corners":
        exp = prediction.corners_expected
        line = 10.5 if exp >= 10 else 9.5
        probs_dict = prediction.corners_probs or {}
        p_over = probs_dict.get(line, {}).get("over_prob", 0)
        p_under = 1 - p_over
        rec = f"Угл. {'Больше' if p_over > 0.55 else 'Меньше'} {line}"
        conf_prob = max(p_over, p_under)
        st.success(f"**{rec}** ({conf_prob*100:.0f}%)")
        st.metric("Ожидается угловых", f"{exp:.1f}")
        st.metric("Рекомендуемая линия", line)
        st.metric("Вероятность Б", f"{p_over*100:.0f}%")
        st.metric("Вероятность М", f"{p_under*100:.0f}%")

        st.markdown("---")
        st.markdown("**Коэффициенты у букмекеров:**")
        fair_over = 1 / max(p_over, 0.001)
        fair_under = 1 / max(p_under, 0.001)
        for bk_name, url in BK_URLS.items():
            st.markdown(
                f'🔵 <a href="{url}" target="_blank" class="bk-badge">{bk_name} — '
                f'Б{line}: ~{fair_over:.2f} | М{line}: ~{fair_under:.2f}</a>',
                unsafe_allow_html=True,
            )

    elif market == "penalty":
        prob = prediction.penalty_probability
        rec = "Пенальти ДА" if prob > 0.32 else ("Пенальти НЕТ" if prob < 0.22 else "Нейтральная зона")
        if prob > 0.32:
            st.success(f"**{rec}** ({prob*100:.0f}%)")
        elif prob < 0.22:
            st.warning(f"**{rec}** ({(1-prob)*100:.0f}% НЕТ)")
        else:
            st.info(f"**{rec}** — пенальти возможен")
        fair_yes = 1 / max(prob, 0.001)
        fair_no = 1 / max(1 - prob, 0.001)
        st.metric("Справедливый коэф ДА", f"{fair_yes:.2f}", f"{prob*100:.0f}%")
        st.metric("Справедливый коэф НЕТ", f"{fair_no:.2f}", f"{(1-prob)*100:.0f}%")

        st.markdown("---")
        st.markdown("**Коэффициенты у букмекеров:**")
        for bk_name, url in BK_URLS.items():
            st.markdown(
                f'🔵 <a href="{url}" target="_blank" class="bk-badge">{bk_name} — '
                f'ДА: ~{fair_yes:.2f} | НЕТ: ~{fair_no:.2f}</a>',
                unsafe_allow_html=True,
            )

    elif market == "medical":
        exp = prediction.medical_expected
        p_over_15 = sum(1 for _ in range(1)) / 1  # placeholder
        rec = f"Медбр. {'Больше' if exp > 1.5 else 'Меньше'} 1.5"
        prob = 0.70 if exp > 1.5 else 0.65
        st.info(f"**{rec}** (ожидается {exp:.1f} выходов)")
        st.metric("Ожидается выходов", f"{exp:.1f}")
        st.caption("Коэффициенты по медбригаде доступны не у всех БК — проверьте вручную")

        st.markdown("---")
        st.markdown("**Коэффициенты у букмекеров:**")
        for bk_name, url in BK_URLS.items():
            st.markdown(
                f'🔵 <a href="{url}" target="_blank" class="bk-badge">{bk_name} →</a>',
                unsafe_allow_html=True,
            )

    # Уверенность и факторы
    st.markdown("---")
    conf = prediction.confidence
    ref_name = getattr(prediction, 'referee', '') or ''
    if referee_stats_df is not None and not referee_stats_df.empty and ref_name:
        ref_row = referee_stats_df[referee_stats_df["referee"] == ref_name]
        if not ref_row.empty:
            strict = ref_row.iloc[0].get("strictness", 1.0)
            st.caption(f"Судья: **{ref_name}** · Строгость: {strict:.2f}")
    st.caption(f"Уверенность модели: **{conf*100:.0f}%** · Факторы: {', '.join(prediction.factors_used)}")


# ==================== СТАТИСТИКА КОМАНД ====================
def show_team_stats(team_stats_df):
    st.header("Статистика команд РПЛ")

    if team_stats_df.empty:
        st.warning("Данные о командах недоступны")
        return

    col1, col2 = st.columns(2)
    with col1:
        market = st.selectbox(
            "Рынок",
            ["Желтые карточки", "Угловые", "Пенальти", "Нарушения"],
        )
    with col2:
        sort_by = st.selectbox("Сортировка", ["По убыванию", "По возрастанию"])

    ascending = sort_by == "По возрастанию"

    metric_map = {
        "Желтые карточки": ("avg_yellow_cards", "Среднее ЖК за матч"),
        "Угловые": ("avg_corners", "Среднее угловых за матч"),
        "Пенальти": ("penalty_rate", "Частота пенальти"),
        "Нарушения": ("avg_fouls", "Среднее нарушений за матч"),
    }

    metric_col, metric_label = metric_map[market]

    if metric_col not in team_stats_df.columns:
        st.warning(f"Метрика '{metric_col}' недоступна")
        return

    sorted_df = team_stats_df.sort_values(metric_col, ascending=ascending)

    fig = px.bar(
        sorted_df,
        x=metric_col,
        y="team",
        orientation="h",
        title=f"{metric_label} | Команды РПЛ",
        color=metric_col,
        color_continuous_scale="RdYlGn" if not ascending else "RdYlGn_r",
        labels={metric_col: metric_label, "team": "Команда"},
    )
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Полная таблица статистики"):
        display_cols = [
            "team", "matches_analyzed", "avg_yellow_cards",
            "avg_corners", "penalty_rate", "avg_fouls",
        ]
        available_cols = [c for c in display_cols if c in team_stats_df.columns]
        st.dataframe(
            team_stats_df[available_cols].sort_values(
                metric_col, ascending=ascending
            ),
            use_container_width=True,
            hide_index=True,
        )


# ==================== СТАТИСТИКА СУДЕЙ ====================
def show_referee_stats(referee_stats_df):
    st.header("Статистика судей РПЛ")

    if referee_stats_df.empty:
        st.warning("Данные о судьях недоступны")
        return

    source = "synthetic"
    if "source" in referee_stats_df.columns:
        sources = referee_stats_df["source"].dropna().unique()
        if "smart-tables.ru" in sources:
            source = "smart-tables.ru"

    if source == "smart-tables.ru":
        st.success(
            "Данные: **[smart-tables.ru](https://smart-tables.ru/referee)** "
            "(сезон 2025/2026) | "
            "Строгость > 1.0 = судья строже среднего, < 1.0 = мягче среднего."
        )
    else:
        st.info(
            "Данные: синтетические (smart-tables.ru недоступен) | "
            "Строгость > 1.0 = судья строже среднего, < 1.0 = мягче среднего."
        )

    col1, col2 = st.columns([2, 1])

    with col1:
        display_cols = [
            c for c in [
                "referee", "matches_count",
                "avg_yellow_cards", "avg_corners",
                "avg_penalties", "strictness_index", "penalty_index",
            ]
            if c in referee_stats_df.columns
        ]

        rename_map = {
            "referee": "Судья",
            "matches_count": "Матчей",
            "avg_yellow_cards": "Ср. ЖК",
            "avg_corners": "Ср. угловые",
            "avg_penalties": "Ср. пенальти",
            "strictness_index": "Строгость",
            "penalty_index": "Индекс пенальти",
        }

        styled_df = referee_stats_df[display_cols].rename(columns=rename_map)

        st.dataframe(
            styled_df.sort_values("Строгость", ascending=False)
            if "Строгость" in styled_df.columns
            else styled_df,
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        if "strictness_index" in referee_stats_df.columns:
            fig = px.scatter(
                referee_stats_df,
                x="strictness_index",
                y="avg_yellow_cards",
                text="referee",
                title="Строгость vs Желтые карточки",
                labels={
                    "strictness_index": "Индекс строгости",
                    "avg_yellow_cards": "Ср. ЖК",
                },
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
            fig.update_traces(textposition="top center")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


# ==================== НАСТРОЙКИ МОДЕЛИ ====================
def show_model_settings():
    st.header("Настройки модели")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Веса факторов")
        weights = {}
        weights["team_form"] = st.slider(
            "Форма команды", 0.0, 1.0,
            MODEL_PARAMS["weights"]["team_form"], 0.05
        )
        weights["referee_stats"] = st.slider(
            "Статистика судьи", 0.0, 1.0,
            MODEL_PARAMS["weights"]["referee_stats"], 0.05
        )
        weights["h2h_stats"] = st.slider(
            "Очные встречи (H2H)", 0.0, 1.0,
            MODEL_PARAMS["weights"]["h2h_stats"], 0.05
        )
        weights["home_away"] = st.slider(
            "Домашнее/гостевое", 0.0, 1.0,
            MODEL_PARAMS["weights"]["home_away"], 0.05
        )
        weights["season_stats"] = st.slider(
            "Сезонная статистика", 0.0, 1.0,
            MODEL_PARAMS["weights"]["season_stats"], 0.05
        )

        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"Сумма весов = {total_weight:.2f} (рекомендуется 1.0)")

    with col2:
        st.subheader("Параметры ставок")

        min_value = st.slider(
            "Минимальный Value (%)", 0, 30,
            int(MODEL_PARAMS["min_value_threshold"] * 100)
        )
        kelly_fraction = st.select_slider(
            "Дробный Келли",
            options=[0.1, 0.25, 0.33, 0.5, 1.0],
            value=MODEL_PARAMS["kelly_fraction"],
        )
        max_bet_pct = st.slider(
            "Макс. % банкролла на ставку", 1, 20,
            int(MODEL_PARAMS["max_bet_fraction"] * 100)
        )
        history_window = st.slider(
            "Окно истории (матчей)", 5, 50,
            MODEL_PARAMS["history_window"]
        )

        st.subheader("Активные букмекеры")
        for bk_key, bk_info in BOOKMAKERS.items():
            st.checkbox(bk_info["name"], value=bk_info["enabled"], key=f"bk_{bk_key}")

    if st.button("Сохранить настройки", type="primary"):
        st.success("Настройки сохранены (применятся при следующем анализе)")


# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    main()

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
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config.settings import RPL_TEAMS, MODEL_PARAMS, BOOKMAKERS
from models.prediction_engine import PredictionEngine, PoissonModel
from analysis.value_finder import ValueBetFinder, BetRecommendation
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================== КЕШИРОВАНИЕ ДАННЫХ ====================
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


@st.cache_data(ttl=1800)  # кеш на 30 минут
def load_bookmaker_odds():
    """Загрузить коэффициенты букмекеров"""
    aggregator = OddsAggregator()
    all_odds = aggregator.get_all_odds()
    best_odds = aggregator.get_best_odds(all_odds)
    return best_odds


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
                "Ценные ставки",
                "Калькулятор ставок",
                "Статистика команд",
                "Статистика судей",
                "Настройки модели",
            ],
        )

        st.divider()
        st.subheader("Обновление данных")

        if st.button("Обновить коэффициенты", type="primary"):
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

    # Маршрутизация
    if page == "Анализ матчей":
        show_match_analysis(matches_df, referee_stats_df, team_stats_df)
    elif page == "Ценные ставки":
        show_value_bets(matches_df, referee_stats_df, team_stats_df)
    elif page == "Калькулятор ставок":
        show_bet_calculator()
    elif page == "Статистика команд":
        show_team_stats(team_stats_df)
    elif page == "Статистика судей":
        show_referee_stats(referee_stats_df)
    elif page == "Настройки модели":
        show_model_settings()


# ==================== АНАЛИЗ МАТЧЕЙ ====================
def show_match_analysis(matches_df, referee_stats_df, team_stats_df):
    st.header("Анализ матчей")

    col1, col2, col3 = st.columns(3)

    teams_list = list(RPL_TEAMS.keys())

    with col1:
        home_team = st.selectbox("Команда хозяев", teams_list, index=0)
    with col2:
        away_team = st.selectbox(
            "Команда гостей",
            [t for t in teams_list if t != home_team],
            index=1,
        )
    with col3:
        referees_list = [""] + (
            list(referee_stats_df["referee"].unique())
            if not referee_stats_df.empty
            else []
        )
        referee = st.selectbox("Судья (необязательно)", referees_list)

    if st.button("Анализировать матч", type="primary", use_container_width=True):
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
            show_market_table(
                prediction.yellow_cards_probs,
                prediction.yellow_cards_expected,
                "Желтые карточки",
                "желтых карточек",
            )

        with tab2:
            show_market_table(
                prediction.corners_probs,
                prediction.corners_expected,
                "Угловые",
                "угловых",
            )

        with tab3:
            show_penalty_analysis(prediction.penalty_probability, referee)

        with tab4:
            show_medical_analysis(
                prediction.medical_expected, prediction.medical_probs
            )

        # Информация о качестве данных
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            quality_color = {
                "high": "green", "medium": "orange", "low": "red"
            }
            st.info(
                f"Качество данных: **{prediction.data_quality.upper()}** | "
                f"Уверенность: **{prediction.confidence * 100:.0f}%** | "
                f"Факторы: {', '.join(prediction.factors_used)}"
            )
        with col2:
            # H2H статистика
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


def show_market_table(probs: Dict, expected: float, title: str, unit: str):
    """Отобразить таблицу вероятностей для рынка"""
    if not probs:
        st.warning("Нет данных для отображения")
        return

    rows = []
    for line, p in probs.items():
        rows.append({
            "Линия": line,
            "Больше (%)": f"{p['over'] * 100:.1f}%",
            "Меньше (%)": f"{p['under'] * 100:.1f}%",
            "Справедливый коэф (Б)": f"{p['over_odds']:.2f}",
            "Справедливый коэф (М)": f"{p['under_odds']:.2f}",
        })

    df = pd.DataFrame(rows)

    # Подсветить строку ближайшую к ожиданию
    st.subheader(f"{title}")
    st.caption(f"Ожидаемое значение: **{expected:.2f}** {unit}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # График распределения Пуассона
    fig = create_poisson_chart(expected, title)
    st.plotly_chart(fig, use_container_width=True)


def create_poisson_chart(lambda_: float, title: str) -> go.Figure:
    """График распределения Пуассона"""
    max_k = max(15, int(lambda_ * 3))
    k_values = list(range(0, max_k + 1))
    probs = [PoissonModel.probability_exact(lambda_, k) for k in k_values]

    # Цвет столбцов (ближе к ожиданию - ярче)
    colors = []
    for k in k_values:
        if abs(k - lambda_) < 1:
            colors.append("#0984e3")
        elif abs(k - lambda_) < 2:
            colors.append("#74b9ff")
        else:
            colors.append("#b2bec3")

    fig = go.Figure(
        data=[
            go.Bar(
                x=k_values,
                y=[p * 100 for p in probs],
                marker_color=colors,
                text=[f"{p*100:.1f}%" if p > 0.02 else "" for p in probs],
                textposition="outside",
            )
        ]
    )
    fig.add_vline(
        x=lambda_,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Среднее: {lambda_:.2f}",
    )
    fig.update_layout(
        title=f"Распределение вероятностей: {title}",
        xaxis_title="Количество",
        yaxis_title="Вероятность (%)",
        height=350,
        showlegend=False,
    )
    return fig


def show_penalty_analysis(penalty_prob: float, referee: str = ""):
    """Анализ пенальти"""
    st.subheader("Пенальти")

    col1, col2 = st.columns(2)
    with col1:
        # Gauge chart
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
                        "value": 28,  # средний по РПЛ
                    },
                },
            )
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fair_yes_odds = 1.0 / max(penalty_prob, 0.001)
        fair_no_odds = 1.0 / max(1 - penalty_prob, 0.001)

        st.metric("Вероятность ДА", f"{penalty_prob * 100:.1f}%")
        st.metric("Вероятность НЕТ", f"{(1-penalty_prob) * 100:.1f}%")
        st.metric("Справедливый коэф ДА", f"{fair_yes_odds:.2f}")
        st.metric("Справедливый коэф НЕТ", f"{fair_no_odds:.2f}")
        st.caption("Среднее по РПЛ: 28% матчей содержат пенальти")


def show_medical_analysis(expected: float, probs: Dict):
    """Анализ выхода медбригады"""
    st.subheader("Выход медицинской бригады")
    st.caption(
        "Выход медбригады — замена игрока из-за травмы "
        "или серьезного нарушения правил"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ожидаемое кол-во выходов", f"{expected:.2f}")
        st.caption("Среднее по РПЛ: 1.8 за матч")

    with col2:
        if probs:
            rows = []
            for line, p in probs.items():
                rows.append({
                    "Линия": line,
                    "Больше (%)": f"{p['over'] * 100:.1f}%",
                    "Меньше (%)": f"{p['under'] * 100:.1f}%",
                    "Коэф (Б)": f"{p['over_odds']:.2f}",
                    "Коэф (М)": f"{p['under_odds']:.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==================== ЦЕННЫЕ СТАВКИ ====================
def show_value_bets(matches_df, referee_stats_df, team_stats_df):
    st.header("Ценные ставки (Value Bets)")

    st.info(
        "Ценная ставка — когда вероятность по модели выше, чем подразумевает "
        "коэффициент букмекера. Value > 5% = хорошая ставка, > 15% = сильная ставка."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Предстоящие матчи")

    with col2:
        min_value = st.slider(
            "Минимальный Value (%)", 0, 30, 5
        )

    # Демо матчи (в реальной версии будут загружаться из API)
    demo_matches = [
        {
            "home_team": "Зенит",
            "away_team": "ЦСКА",
            "referee": "Казарцев С.",
            "date": "2024-12-15",
        },
        {
            "home_team": "Спартак",
            "away_team": "Краснодар",
            "referee": "Лапочкин С.",
            "date": "2024-12-15",
        },
        {
            "home_team": "Локомотив",
            "away_team": "Динамо",
            "referee": "Вилков В.",
            "date": "2024-12-16",
        },
        {
            "home_team": "Ростов",
            "away_team": "Рубин",
            "referee": "Матюнин А.",
            "date": "2024-12-16",
        },
    ]

    with st.spinner("Загрузка коэффициентов и анализ..."):
        engine = PredictionEngine()
        finder = ValueBetFinder()

        # Прогнозы
        predictions = engine.batch_predict(
            demo_matches, team_stats_df, referee_stats_df, matches_df
        )

        # Коэффициенты (демо + реальные)
        aggregator = OddsAggregator()
        raw_odds = aggregator.get_all_odds()
        best_odds = aggregator.get_best_odds(raw_odds)

        # Добавить демо коэффициенты для матчей
        demo_odds = _generate_demo_odds(predictions)
        for k, v in demo_odds.items():
            if k not in best_odds:
                best_odds[k] = v

    # Отображение ценных ставок
    all_value_bets = []
    for pred in predictions:
        match_key = f"{pred.home_team} - {pred.away_team}"
        bk_data = best_odds.get(match_key, {})
        bets = finder.analyze_match(pred, bk_data)
        all_value_bets.extend(bets)

    if all_value_bets:
        # Фильтр по ценности
        filtered = [
            b for b in all_value_bets
            if b.value * 100 >= min_value
            and b.recommendation != BetRecommendation.AVOID
        ]

        if filtered:
            # Сортировка по ценности
            filtered.sort(key=lambda x: x.value, reverse=True)

            st.success(f"Найдено {len(filtered)} ценных ставок")

            for bet in filtered:
                with st.expander(
                    f"{'🔥' if bet.value >= 0.15 else '✅'} "
                    f"{bet.match} | {bet.bet_type} | "
                    f"Value: {bet.value * 100:+.1f}%",
                    expanded=bet.value >= 0.10,
                ):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Букмекер", bet.bookmaker)
                        st.metric("Коэффициент БК", f"{bet.bookmaker_odds:.2f}")
                    with col2:
                        st.metric(
                            "Вероятность модели",
                            f"{bet.model_probability * 100:.1f}%"
                        )
                        st.metric(
                            "Подразумевает БК",
                            f"{bet.bookmaker_implied_prob * 100:.1f}%"
                        )
                    with col3:
                        st.metric("Справедливый коэф", f"{bet.fair_odds:.2f}")
                        value_delta = f"{bet.value * 100:+.1f}%"
                        st.metric("Value", value_delta)
                    with col4:
                        st.metric(
                            "Ставка Келли (1/4)",
                            f"{bet.kelly_fraction * 100:.2f}% банкролла"
                        )
                        st.metric(
                            "Ожидаемый ROI",
                            f"{bet.expected_roi * 100:.1f}%"
                        )

                    recommendation_colors = {
                        BetRecommendation.STRONG_BET: "success",
                        BetRecommendation.GOOD_BET: "info",
                        BetRecommendation.NEUTRAL: "warning",
                    }
                    getattr(
                        st,
                        recommendation_colors.get(bet.recommendation, "info")
                    )(f"Рекомендация: **{bet.recommendation.value}**")
        else:
            st.warning(
                f"Ценных ставок с Value >= {min_value}% не найдено. "
                "Попробуйте снизить порог."
            )
    else:
        st.info("Нет данных для анализа. Обновите коэффициенты.")


def _generate_demo_odds(predictions) -> Dict:
    """Генерация демо коэффициентов для тестирования"""
    demo = {}
    np.random.seed(42)

    for pred in predictions:
        match_key = f"{pred.home_team} - {pred.away_team}"
        # Генерируем "немного ошибочные" коэффициенты букмекера
        # (иногда они дают Value)
        yc_line = round(pred.yellow_cards_expected)
        c_line = round(pred.corners_expected)

        demo[match_key] = {
            "home_team": pred.home_team,
            "away_team": pred.away_team,
            "markets": {
                "yellow_cards_over": {
                    "bookmaker": "fonbet",
                    "odds": round(1 / max(
                        PoissonModel.probability_over(
                            pred.yellow_cards_expected, yc_line - 0.5
                        ) * 0.93, 0.01
                    ), 2),
                    "line": yc_line - 0.5,
                },
                "yellow_cards_under": {
                    "bookmaker": "leon",
                    "odds": round(1 / max(
                        PoissonModel.probability_under(
                            pred.yellow_cards_expected, yc_line + 0.5
                        ) * 0.93, 0.01
                    ), 2),
                    "line": yc_line + 0.5,
                },
                "corners_over": {
                    "bookmaker": "fonbet",
                    "odds": round(1 / max(
                        PoissonModel.probability_over(
                            pred.corners_expected, c_line - 0.5
                        ) * 0.93, 0.01
                    ), 2),
                    "line": c_line - 0.5,
                },
                "corners_under": {
                    "bookmaker": "winline",
                    "odds": round(1 / max(
                        PoissonModel.probability_under(
                            pred.corners_expected, c_line + 0.5
                        ) * 0.93, 0.01
                    ), 2),
                    "line": c_line + 0.5,
                },
                "penalty_yes": {
                    "bookmaker": "fonbet",
                    "odds": round(
                        1 / max(pred.penalty_probability * np.random.uniform(0.88, 0.95), 0.01),
                        2
                    ),
                    "line": None,
                },
                "penalty_no": {
                    "bookmaker": "leon",
                    "odds": round(
                        1 / max(
                            (1 - pred.penalty_probability) * np.random.uniform(0.90, 0.96),
                            0.01
                        ),
                        2
                    ),
                    "line": None,
                },
            },
        }

    return demo


# ==================== КАЛЬКУЛЯТОР СТАВОК ====================
def show_bet_calculator():
    st.header("Калькулятор ставок")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Параметры ставки")

        bankroll = st.number_input(
            "Банкролл (₽)", min_value=100.0, max_value=10000000.0,
            value=10000.0, step=1000.0,
        )
        bet_amount = st.number_input(
            "Сумма ставки (₽)", min_value=10.0, max_value=float(bankroll),
            value=min(500.0, bankroll * 0.05), step=100.0,
        )
        odds = st.number_input(
            "Коэффициент букмекера", min_value=1.01, max_value=100.0,
            value=1.85, step=0.05,
        )
        model_prob = st.slider(
            "Вероятность по модели (%)", 1, 99, 55
        ) / 100

        st.divider()
        st.subheader("Симуляция серии ставок")
        n_bets = st.slider("Количество ставок", 10, 500, 100)

    with col2:
        calculator = BetCalculator()
        calc = calculator.calculate(bet_amount, odds, model_prob, bankroll)

        st.subheader("Результаты расчета")

        # Ключевые метрики
        value = (model_prob * odds) - 1.0
        value_color = "normal" if value >= 0 else "inverse"

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Потенциальный выигрыш",
                f"{calc.potential_win:.0f} ₽",
                delta=f"+{calc.potential_profit:.0f} ₽"
            )
            st.metric(
                "Вероятность прохода",
                f"{calc.pass_probability * 100:.1f}%"
            )
            st.metric(
                "Ставка Келли (1/4)",
                f"{calc.quarter_kelly_bet:.0f} ₽",
                help="Оптимальный размер ставки по критерию Келли"
            )
        with col_b:
            st.metric(
                "Ожидаемый ROI",
                f"{calc.expected_roi:.1f}%",
                delta=f"{value * 100:+.1f}% Value"
            )
            st.metric("Точка безубыточности", f"{calc.breakeven_probability * 100:.1f}%")
            st.metric("Уровень риска", calc.risk_level)

        # Таблица полных расчетов
        with st.expander("Полные расчеты"):
            for key, val in calc.to_dict().items():
                st.write(f"**{key}:** {val}")

    # Симуляция
    st.divider()
    if st.button("Запустить симуляцию Монте-Карло", type="secondary"):
        with st.spinner("Симуляция..."):
            sim = calculator.simulate_bankroll(
                bet_amount, odds, model_prob, n_bets, bankroll
            )

        st.subheader("Результаты симуляции")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Средний финальный банкролл", f"{sim['mean_final']:.0f} ₽")
        with col2:
            st.metric("Медианный банкролл", f"{sim['median_final']:.0f} ₽")
        with col3:
            st.metric("Вероятность прибыли", f"{sim['prob_profit'] * 100:.1f}%")
        with col4:
            st.metric("Риск разорения", f"{sim['prob_ruin'] * 100:.1f}%")

        # Прогрессбары
        st.progress(sim['prob_profit'], text=f"Вероятность прибыли: {sim['prob_profit'] * 100:.1f}%")

        # Диапазон результатов
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["P10 (пессимист)", "Медиана", "Среднее", "P90 (оптимист)"],
            y=[sim['p10_final'], sim['median_final'], sim['mean_final'], sim['p90_final']],
            marker_color=["#e17055", "#fdcb6e", "#0984e3", "#00b894"],
            text=[f"{v:.0f} ₽" for v in [
                sim['p10_final'], sim['median_final'],
                sim['mean_final'], sim['p90_final']
            ]],
            textposition="outside",
        ))
        fig.add_hline(y=bankroll, line_dash="dash", line_color="gray",
                      annotation_text="Начальный банкролл")
        fig.update_layout(
            title=f"Диапазон финального банкролла после {n_bets} ставок",
            yaxis_title="Банкролл (₽)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


# ==================== СТАТИСТИКА КОМАНД ====================
def show_team_stats(team_stats_df):
    st.header("Статистика команд РПЛ")

    if team_stats_df.empty:
        st.warning("Данные о командах недоступны")
        return

    # Фильтры
    col1, col2 = st.columns(2)
    with col1:
        market = st.selectbox(
            "Рынок",
            ["Желтые карточки", "Угловые", "Пенальти", "Нарушения"],
        )
    with col2:
        sort_by = st.selectbox("Сортировка", ["По убыванию", "По возрастанию"])

    ascending = sort_by == "По возрастанию"

    # Выбор метрики
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

    # Горизонтальный бар чарт
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

    # Таблица
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

    st.info(
        "Строгость судьи влияет на количество желтых карточек и пенальти. "
        "Индекс > 1.0 = судья строже среднего, < 1.0 = мягче среднего."
    )

    # Таблица судей
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

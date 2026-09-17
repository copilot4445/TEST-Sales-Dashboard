from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESOURCE_DIR = Path(__file__).parent / "resource"
CHART_WIDTH = "stretch"
DATE_PRESETS = ["전체", "최신 연도", "올해", "작년", "최근 12개월", "사용자 지정"]
FILTER_STATE_KEYS = [
    "date_preset",
    "date_range",
    "filter_customer_search",
    "filter_customer_names",
    "filter_manual_quarters",
    "filter_quarters",
    "filter_regions",
    "filter_customer_types",
    "filter_sales_reps",
    "filter_product_groups",
    "filter_order_statuses",
    "filter_grades",
    "filter_sales_date",
    "filter_activity_date",
    "applied_filters",
]


def format_currency(value: float) -> str:
    return f"₩{value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def empty_frames(
    sales: pd.DataFrame,
    customer: pd.DataFrame,
    activity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return sales.iloc[0:0].copy(), customer.iloc[0:0].copy(), activity.iloc[0:0].copy()


def apply_money_chart_format(fig, money_cols: list[str] | None = None) -> None:
    fig.update_layout(hovermode="x unified")
    if money_cols:
        for col in money_cols:
            fig.update_traces(
                selector={"name": col},
                hovertemplate=f"{col}: %{{y:,.0f}}원<extra></extra>",
            )
    fig.update_yaxes(tickformat=",")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(RESOURCE_DIR / "sales_data.csv", encoding="utf-8-sig")
    customer = pd.read_csv(RESOURCE_DIR / "customer_data.csv", encoding="utf-8-sig")
    activity = pd.read_csv(RESOURCE_DIR / "activity_data.csv", encoding="utf-8-sig")

    sales["수주일"] = pd.to_datetime(sales["수주일"])
    sales["납품일"] = pd.to_datetime(sales["납품일"])
    customer["첫거래일"] = pd.to_datetime(customer["첫거래일"])
    activity["활동일"] = pd.to_datetime(activity["활동일"])

    sales = sales.merge(
        customer[["거래처명", "등급", "첫거래일", "누적매출"]],
        on="거래처명",
        how="left",
        suffixes=("", "_고객"),
    )

    return sales, customer, activity


def get_data_date_bounds(
    sales: pd.DataFrame,
    activity: pd.DataFrame,
) -> tuple[date, date]:
    min_date = min(sales["수주일"].min(), activity["활동일"].min()).date()
    max_date = max(sales["수주일"].max(), activity["활동일"].max()).date()
    return min_date, max_date


def multiselect_filter(
    label: str,
    options: list,
    key: str,
    container: st.delta_generator.DeltaGenerator | None = None,
) -> list:
    widget = container if container is not None else st
    return widget.multiselect(label, options, default=options, key=key)


def resolve_date_range(preset: str, min_date: date, max_date: date) -> tuple[date, date]:
    anchor_year = max_date.year

    if preset == "전체":
        return min_date, max_date
    if preset == "최신 연도":
        start = date(anchor_year, 1, 1)
        end = date(anchor_year, 12, 31)
        return max(start, min_date), min(end, max_date)
    if preset == "올해":
        start = date(anchor_year, 1, 1)
        return max(start, min_date), max_date
    if preset == "작년":
        year = anchor_year - 1
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        return max(start, min_date), min(end, max_date)
    if preset == "최근 12개월":
        end = max_date
        start = end - timedelta(days=365)
        return max(start, min_date), end
    return min_date, max_date


def quarters_in_date_range(
    sales: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> list:
    in_range = sales[
        (sales["수주일"].dt.date >= start_date)
        & (sales["수주일"].dt.date <= end_date)
    ]
    return sorted(in_range["분기"].unique())


def build_default_filters(
    sales: pd.DataFrame,
    customer: pd.DataFrame,
    activity: pd.DataFrame,
) -> dict:
    min_date, max_date = get_data_date_bounds(sales, activity)
    all_quarters = sorted(sales["분기"].unique())
    return {
        "date_preset": "전체",
        "date_range": (min_date, max_date),
        "filter_sales_date": True,
        "filter_activity_date": True,
        "manual_quarters": False,
        "customer_search": "",
        "customer_names": sorted(customer["거래처명"].unique()),
        "quarters": all_quarters,
        "regions": sorted(sales["지역"].unique()),
        "customer_types": sorted(sales["거래처유형"].unique()),
        "sales_reps": sorted(sales["담당영업"].unique()),
        "product_groups": sorted(sales["제품군"].unique()),
        "order_statuses": sorted(sales["수주상태"].unique()),
        "grades": sorted(customer["등급"].unique()),
        "invalid_selection": False,
    }


def collect_filters_from_widgets(
    sales: pd.DataFrame,
    customer: pd.DataFrame,
    activity: pd.DataFrame,
) -> dict:
    min_date, max_date = get_data_date_bounds(sales, activity)
    preset = st.session_state.get("date_preset", "전체")

    if preset == "사용자 지정":
        date_range = st.session_state.get("date_range", (min_date, max_date))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            parsed_date_range = date_range
        else:
            parsed_date_range = (min_date, max_date)
    else:
        parsed_date_range = resolve_date_range(preset, min_date, max_date)

    manual_quarters = st.session_state.get("filter_manual_quarters", False)
    all_quarters = sorted(sales["분기"].unique())
    selected_quarters = st.session_state.get("filter_quarters", all_quarters)
    if manual_quarters:
        effective_quarters = selected_quarters
    else:
        effective_quarters = quarters_in_date_range(
            sales,
            parsed_date_range[0],
            parsed_date_range[1],
        )

    multiselect_values = {
        "customer_names": st.session_state.get(
            "filter_customer_names",
            sorted(customer["거래처명"].unique()),
        ),
        "quarters": effective_quarters,
        "regions": st.session_state.get(
            "filter_regions",
            sorted(sales["지역"].unique()),
        ),
        "customer_types": st.session_state.get(
            "filter_customer_types",
            sorted(sales["거래처유형"].unique()),
        ),
        "sales_reps": st.session_state.get(
            "filter_sales_reps",
            sorted(sales["담당영업"].unique()),
        ),
        "product_groups": st.session_state.get(
            "filter_product_groups",
            sorted(sales["제품군"].unique()),
        ),
        "order_statuses": st.session_state.get(
            "filter_order_statuses",
            sorted(sales["수주상태"].unique()),
        ),
        "grades": st.session_state.get(
            "filter_grades",
            sorted(customer["등급"].unique()),
        ),
    }

    invalid_selection = any(not value for value in multiselect_values.values())
    if manual_quarters and not selected_quarters:
        invalid_selection = True

    return {
        "date_preset": preset,
        "date_range": parsed_date_range,
        "filter_sales_date": st.session_state.get("filter_sales_date", True),
        "filter_activity_date": st.session_state.get("filter_activity_date", True),
        "manual_quarters": manual_quarters,
        "manual_quarter_selection": selected_quarters if manual_quarters else [],
        "customer_search": st.session_state.get("filter_customer_search", "").strip(),
        "invalid_selection": invalid_selection,
        **multiselect_values,
    }


def apply_quick_preset(
    preset_name: str,
    sales: pd.DataFrame,
    customer: pd.DataFrame,
    activity: pd.DataFrame,
) -> None:
    defaults = build_default_filters(sales, customer, activity)

    st.session_state["date_preset"] = defaults["date_preset"]
    st.session_state["date_range"] = defaults["date_range"]
    st.session_state["filter_sales_date"] = defaults["filter_sales_date"]
    st.session_state["filter_activity_date"] = defaults["filter_activity_date"]
    st.session_state["filter_manual_quarters"] = defaults["manual_quarters"]
    st.session_state["filter_customer_search"] = defaults["customer_search"]
    st.session_state["filter_customer_names"] = defaults["customer_names"]
    st.session_state["filter_quarters"] = defaults["quarters"]
    st.session_state["filter_regions"] = defaults["regions"]
    st.session_state["filter_customer_types"] = defaults["customer_types"]
    st.session_state["filter_sales_reps"] = defaults["sales_reps"]
    st.session_state["filter_product_groups"] = defaults["product_groups"]
    st.session_state["filter_order_statuses"] = defaults["order_statuses"]
    st.session_state["filter_grades"] = defaults["grades"]

    if preset_name == "vip":
        st.session_state["filter_grades"] = ["VIP"]
    elif preset_name == "in_progress":
        st.session_state["filter_order_statuses"] = ["진행중"]
    elif preset_name == "seoul":
        st.session_state["filter_regions"] = ["서울"]

    st.session_state["applied_filters"] = collect_filters_from_widgets(
        sales,
        customer,
        activity,
    )


def reset_filters(
    sales: pd.DataFrame,
    customer: pd.DataFrame,
    activity: pd.DataFrame,
) -> None:
    for key in FILTER_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state["applied_filters"] = build_default_filters(sales, customer, activity)


def apply_filters(
    sales: pd.DataFrame,
    customer: pd.DataFrame,
    activity: pd.DataFrame,
    filters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if filters.get("invalid_selection"):
        return empty_frames(sales, customer, activity)

    multiselect_keys = [
        "customer_names",
        "quarters",
        "regions",
        "customer_types",
        "sales_reps",
        "product_groups",
        "order_statuses",
        "grades",
    ]
    if any(not filters[key] for key in multiselect_keys):
        return empty_frames(sales, customer, activity)

    filtered_sales = sales.copy()
    filtered_customer = customer.copy()
    filtered_activity = activity.copy()

    customer_names = filters["customer_names"]
    search_text = filters.get("customer_search", "")
    if search_text:
        customer_names = [
            name
            for name in customer_names
            if search_text.lower() in name.lower()
        ]
        if not customer_names:
            return empty_frames(sales, customer, activity)

    filtered_sales = filtered_sales[filtered_sales["거래처명"].isin(customer_names)]
    filtered_customer = filtered_customer[
        filtered_customer["거래처명"].isin(customer_names)
    ]
    filtered_activity = filtered_activity[
        filtered_activity["거래처명"].isin(customer_names)
    ]

    filtered_sales = filtered_sales[filtered_sales["분기"].isin(filters["quarters"])]
    filtered_sales = filtered_sales[filtered_sales["지역"].isin(filters["regions"])]
    filtered_customer = filtered_customer[
        filtered_customer["지역"].isin(filters["regions"])
    ]
    filtered_sales = filtered_sales[
        filtered_sales["거래처유형"].isin(filters["customer_types"])
    ]
    filtered_customer = filtered_customer[
        filtered_customer["거래처유형"].isin(filters["customer_types"])
    ]
    filtered_sales = filtered_sales[filtered_sales["담당영업"].isin(filters["sales_reps"])]
    filtered_activity = filtered_activity[
        filtered_activity["담당영업"].isin(filters["sales_reps"])
    ]
    filtered_sales = filtered_sales[
        filtered_sales["제품군"].isin(filters["product_groups"])
    ]
    filtered_sales = filtered_sales[
        filtered_sales["수주상태"].isin(filters["order_statuses"])
    ]
    filtered_sales = filtered_sales[filtered_sales["등급"].isin(filters["grades"])]
    filtered_customer = filtered_customer[
        filtered_customer["등급"].isin(filters["grades"])
    ]

    start_date, end_date = filters["date_range"]
    if filters.get("filter_sales_date", True):
        filtered_sales = filtered_sales[
            (filtered_sales["수주일"].dt.date >= start_date)
            & (filtered_sales["수주일"].dt.date <= end_date)
        ]
    if filters.get("filter_activity_date", True):
        filtered_activity = filtered_activity[
            (filtered_activity["활동일"].dt.date >= start_date)
            & (filtered_activity["활동일"].dt.date <= end_date)
        ]

    sales_customers = set(filtered_sales["거래처명"].unique())
    customer_customers = set(filtered_customer["거래처명"].unique())
    activity_customers = set(filtered_activity["거래처명"].unique())
    matched_customers = sales_customers & customer_customers
    if activity_customers:
        matched_customers &= activity_customers

    filtered_sales = filtered_sales[filtered_sales["거래처명"].isin(matched_customers)]
    filtered_customer = filtered_customer[
        filtered_customer["거래처명"].isin(matched_customers)
    ]
    filtered_activity = filtered_activity[
        filtered_activity["거래처명"].isin(matched_customers)
    ]

    return filtered_sales, filtered_customer, filtered_activity


def detect_filter_conflict(sales: pd.DataFrame, filters: dict) -> bool:
    if not filters.get("manual_quarters"):
        return False

    start_date, end_date = filters["date_range"]
    manual_quarters = filters.get("manual_quarter_selection") or filters["quarters"]
    in_date = sales[
        (sales["수주일"].dt.date >= start_date)
        & (sales["수주일"].dt.date <= end_date)
    ]
    in_quarter = sales[sales["분기"].isin(manual_quarters)]
    if in_date.empty or in_quarter.empty:
        return False
    overlap = in_date[in_date["분기"].isin(manual_quarters)]
    return overlap.empty


def format_active_filters(
    filters: dict,
    sales: pd.DataFrame,
    customer: pd.DataFrame,
    activity: pd.DataFrame,
) -> str:
    defaults = build_default_filters(sales, customer, activity)
    parts: list[str] = []

    if filters.get("date_preset") != defaults["date_preset"]:
        parts.append(filters["date_preset"])

    start_date, end_date = filters["date_range"]
    if start_date != defaults["date_range"][0] or end_date != defaults["date_range"][1]:
        parts.append(f"기간={start_date}~{end_date}")

    if not filters.get("filter_sales_date", True):
        parts.append("매출 기간 미적용")
    if not filters.get("filter_activity_date", True):
        parts.append("활동 기간 미적용")

    if filters.get("customer_search"):
        parts.append(f'거래처 검색="{filters["customer_search"]}"')

    label_map = [
        ("regions", "지역"),
        ("customer_types", "거래처유형"),
        ("sales_reps", "담당영업"),
        ("product_groups", "제품군"),
        ("order_statuses", "수주상태"),
        ("grades", "등급"),
    ]
    for key, label in label_map:
        if set(filters[key]) != set(defaults[key]):
            parts.append(f"{label}={','.join(filters[key])}")

    if filters.get("manual_quarters"):
        manual = filters.get("manual_quarter_selection") or filters["quarters"]
        parts.append(f"분기 직접={','.join(manual)}")

    if set(filters["customer_names"]) != set(defaults["customer_names"]):
        parts.append(f"거래처 {len(filters['customer_names'])}개 선택")

    if not parts:
        return "활성 필터: 없음 (전체 데이터)"
    return "활성 필터: " + " | ".join(parts)


def build_filter_summary(
    filtered_sales: pd.DataFrame,
    filtered_customer: pd.DataFrame,
    filtered_activity: pd.DataFrame,
    filters: dict,
) -> str:
    start_date, end_date = filters["date_range"]
    return (
        f"기간 {start_date} ~ {end_date} | "
        f"거래 {len(filtered_sales):,}건 | "
        f"거래처 {filtered_customer['거래처명'].nunique():,}개 | "
        f"활동 {len(filtered_activity):,}건"
    )


def compute_prior_period_sales(
    sales: pd.DataFrame,
    customer: pd.DataFrame,
    activity: pd.DataFrame,
    filters: dict,
) -> tuple[float, int]:
    start_date, end_date = filters["date_range"]
    period_days = (end_date - start_date).days + 1
    prior_end = start_date - timedelta(days=1)
    prior_start = prior_end - timedelta(days=period_days - 1)

    prior_filters = {**filters, "date_range": (prior_start, prior_end)}
    prior_sales, _, _ = apply_filters(sales, customer, activity, prior_filters)
    return prior_sales["매출금액"].sum(), len(prior_sales)


def download_csv_button(
    df: pd.DataFrame,
    filename: str,
    label: str = "CSV 다운로드",
    key: str | None = None,
) -> None:
    buffer = BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def render_sidebar_filters(
    sales: pd.DataFrame,
    customer: pd.DataFrame,
    activity: pd.DataFrame,
) -> dict:
    st.sidebar.header("필터")
    min_date, max_date = get_data_date_bounds(sales, activity)

    if "applied_filters" not in st.session_state:
        st.session_state["applied_filters"] = build_default_filters(
            sales,
            customer,
            activity,
        )

    quick_col1, quick_col2, quick_col3 = st.sidebar.columns(3)
    if quick_col1.button("VIP만", use_container_width=True):
        apply_quick_preset("vip", sales, customer, activity)
        st.rerun()
    if quick_col2.button("진행중", use_container_width=True):
        apply_quick_preset("in_progress", sales, customer, activity)
        st.rerun()
    if quick_col3.button("서울", use_container_width=True):
        apply_quick_preset("seoul", sales, customer, activity)
        st.rerun()

    if st.sidebar.button("필터 초기화", width="stretch"):
        reset_filters(sales, customer, activity)
        st.rerun()

    if "filter_sales_date" not in st.session_state:
        st.session_state["filter_sales_date"] = True
    if "filter_activity_date" not in st.session_state:
        st.session_state["filter_activity_date"] = True
    if "filter_manual_quarters" not in st.session_state:
        st.session_state["filter_manual_quarters"] = False
    if "filter_customer_search" not in st.session_state:
        st.session_state["filter_customer_search"] = ""

    with st.sidebar.form("filter_form"):
        preset = st.selectbox("기간 프리셋", DATE_PRESETS, key="date_preset")

        if preset == "사용자 지정":
            st.date_input(
                "기간",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="date_range",
            )
        else:
            preview_range = resolve_date_range(preset, min_date, max_date)
            st.info(f"적용 기간: {preview_range[0]} ~ {preview_range[1]}")

        st.checkbox("매출(수주일)에 기간 적용", key="filter_sales_date")
        st.checkbox("영업활동(활동일)에 기간 적용", key="filter_activity_date")
        st.text_input("거래처 검색", key="filter_customer_search", placeholder="이름 일부 입력")

        multiselect_filter(
            "지역",
            sorted(sales["지역"].unique()),
            "filter_regions",
        )
        multiselect_filter(
            "담당영업",
            sorted(sales["담당영업"].unique()),
            "filter_sales_reps",
        )

        with st.expander("상세 필터", expanded=False):
            multiselect_filter(
                "거래처 직접 선택",
                sorted(customer["거래처명"].unique()),
                "filter_customer_names",
            )
            multiselect_filter(
                "거래처유형",
                sorted(sales["거래처유형"].unique()),
                "filter_customer_types",
            )
            manual_quarters = st.checkbox("분기 직접 지정", key="filter_manual_quarters")
            if manual_quarters:
                multiselect_filter(
                    "분기",
                    sorted(sales["분기"].unique()),
                    "filter_quarters",
                )
            multiselect_filter(
                "제품군",
                sorted(sales["제품군"].unique()),
                "filter_product_groups",
            )
            multiselect_filter(
                "수주상태",
                sorted(sales["수주상태"].unique()),
                "filter_order_statuses",
            )
            multiselect_filter(
                "거래처 등급",
                sorted(customer["등급"].unique()),
                "filter_grades",
            )

        submitted = st.form_submit_button("필터 적용", width="stretch")
        if submitted:
            candidate = collect_filters_from_widgets(sales, customer, activity)
            if candidate["invalid_selection"]:
                st.error("모든 필터에서 1개 이상 선택해 주세요.")
            else:
                st.session_state["applied_filters"] = candidate

    return st.session_state["applied_filters"]


def render_overview_tab(
    sales: pd.DataFrame,
    customer: pd.DataFrame,
    activity: pd.DataFrame,
    summary_text: str,
) -> None:
    st.caption(summary_text)

    if sales.empty and customer.empty and activity.empty:
        st.info("조건에 맞는 데이터가 없습니다.")
        return

    total_revenue = sales["매출금액"].sum() if not sales.empty else 0
    total_receivable = sales["미수금"].sum() if not sales.empty else 0
    activity_count = len(activity)
    positive_rate = (
        (activity["활동결과"] == "긍정").mean() * 100 if not activity.empty else 0
    )
    vip_count = (
        customer[customer["등급"] == "VIP"]["거래처명"].nunique()
        if not customer.empty
        else 0
    )
    order_count = len(sales)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("총 매출금액", format_currency(total_revenue))
    c2.metric("미수금 합계", format_currency(total_receivable))
    c3.metric("거래 건수", f"{order_count:,}건")
    c4.metric("활동 건수", f"{activity_count:,}건")
    c5.metric("긍정 비율", format_percent(positive_rate))
    c6.metric("VIP 거래처", f"{vip_count:,}개")

    col1, col2 = st.columns(2)

    with col1:
        if not sales.empty:
            quarter_df = (
                sales.groupby("분기", as_index=False)["매출금액"]
                .sum()
                .sort_values("분기")
            )
            fig_quarter = px.bar(
                quarter_df,
                x="분기",
                y="매출금액",
                title="분기별 매출",
                labels={"매출금액": "매출금액 (원)"},
            )
            apply_money_chart_format(fig_quarter)
            st.plotly_chart(fig_quarter, width=CHART_WIDTH, key="overview_quarter_chart")
        else:
            st.info("매출 데이터가 없습니다.")

    with col2:
        if not activity.empty:
            outcome_df = activity["활동결과"].value_counts().reset_index()
            outcome_df.columns = ["활동결과", "건수"]
            fig_outcome = px.pie(
                outcome_df,
                names="활동결과",
                values="건수",
                title="활동결과 분포",
            )
            st.plotly_chart(fig_outcome, width=CHART_WIDTH, key="overview_outcome_chart")
        else:
            st.info("영업활동 데이터가 없습니다.")


def render_sales_tab(
    sales: pd.DataFrame,
    source_sales: pd.DataFrame,
    source_customer: pd.DataFrame,
    source_activity: pd.DataFrame,
    filters: dict,
) -> None:
    if sales.empty:
        st.info("조건에 맞는 데이터가 없습니다.")
        return

    total_order = sales["수주금액"].sum()
    total_revenue = sales["매출금액"].sum()
    total_receivable = sales["미수금"].sum()
    order_count = len(sales)
    avg_order = sales["수주금액"].mean()
    cancel_rate = (sales["수주상태"] == "취소").mean() * 100

    prior_revenue, prior_count = compute_prior_period_sales(
        source_sales,
        source_customer,
        source_activity,
        filters,
    )
    revenue_delta = total_revenue - prior_revenue
    count_delta = order_count - prior_count

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 수주금액", format_currency(total_order))
    c2.metric(
        "총 매출금액",
        format_currency(total_revenue),
        delta=f"{revenue_delta:,.0f}원",
    )
    c3.metric("거래 건수", f"{order_count:,}건", delta=f"{count_delta:,}건")
    c4.metric("평균 수주금액", format_currency(avg_order))
    c5.metric("미수금 합계", format_currency(total_receivable))

    st.caption(
        f"취소율: {format_percent(cancel_rate)} | "
        "전기 대비: 직전 동일 길이 기간과 비교"
    )

    col1, col2 = st.columns(2)

    with col1:
        quarter_df = (
            sales.groupby("분기", as_index=False)["매출금액"]
            .sum()
            .sort_values("분기")
        )
        fig_quarter = px.bar(
            quarter_df,
            x="분기",
            y="매출금액",
            title="분기별 매출 추이",
            labels={"매출금액": "매출금액 (원)"},
        )
        apply_money_chart_format(fig_quarter)
        st.plotly_chart(fig_quarter, width=CHART_WIDTH, key="sales_quarter_chart")

        product_df = sales.groupby("제품군", as_index=False)["매출금액"].sum()
        fig_product = px.pie(
            product_df,
            names="제품군",
            values="매출금액",
            title="제품군별 매출",
        )
        st.plotly_chart(fig_product, width=CHART_WIDTH, key="sales_product_chart")

    with col2:
        region_df = (
            sales.groupby("지역", as_index=False)["매출금액"]
            .sum()
            .sort_values("매출금액", ascending=True)
        )
        fig_region = px.bar(
            region_df,
            x="매출금액",
            y="지역",
            orientation="h",
            title="지역별 매출",
            labels={"매출금액": "매출금액 (원)"},
        )
        apply_money_chart_format(fig_region)
        st.plotly_chart(fig_region, width=CHART_WIDTH, key="sales_region_chart")

        rep_df = sales.groupby("담당영업", as_index=False)["매출금액"].sum()
        fig_rep = px.bar(
            rep_df,
            x="담당영업",
            y="매출금액",
            title="담당영업별 매출",
            labels={"매출금액": "매출금액 (원)"},
        )
        apply_money_chart_format(fig_rep)
        st.plotly_chart(fig_rep, width=CHART_WIDTH, key="sales_rep_chart")

    status_df = (
        sales.groupby("수주상태", as_index=False)
        .agg(건수=("거래ID", "count"), 금액=("매출금액", "sum"))
    )
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        fig_status_count = px.bar(
            status_df,
            x="수주상태",
            y="건수",
            title="수주상태별 건수",
        )
        st.plotly_chart(fig_status_count, width=CHART_WIDTH, key="sales_status_count_chart")
    with status_col2:
        fig_status_amount = px.bar(
            status_df,
            x="수주상태",
            y="금액",
            title="수주상태별 매출금액",
            labels={"금액": "매출금액 (원)"},
        )
        apply_money_chart_format(fig_status_amount)
        st.plotly_chart(fig_status_amount, width=CHART_WIDTH, key="sales_status_amount_chart")

    export_sales = sales.copy()
    export_sales["수주일"] = export_sales["수주일"].dt.strftime("%Y-%m-%d")
    export_sales["납품일"] = export_sales["납품일"].dt.strftime("%Y-%m-%d")

    display_sales = export_sales.copy()
    for col in ["수주금액", "매출금액", "미수금"]:
        display_sales[col] = display_sales[col].apply(format_currency)

    st.subheader("거래 상세")
    download_csv_button(export_sales, "sales_filtered.csv", key="download_sales_csv")
    st.dataframe(display_sales, width=CHART_WIDTH, hide_index=True, key="sales_detail_table")


def render_customer_tab(customer: pd.DataFrame, sales: pd.DataFrame) -> None:
    if customer.empty:
        st.info("조건에 맞는 데이터가 없습니다.")
        return

    active_customers = customer["거래처명"].nunique()
    grade_counts = customer["등급"].value_counts(normalize=True) * 100
    avg_cumulative = customer["누적매출"].mean()
    period_revenue = sales["매출금액"].sum() if not sales.empty else 0
    period_orders = len(sales)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("활성 거래처 수", f"{active_customers:,}개")
    c2.metric("기간 매출", format_currency(period_revenue))
    c3.metric("기간 거래 건수", f"{period_orders:,}건")
    c4.metric("평균 누적매출", format_currency(avg_cumulative))
    c5.metric("VIP 비율", format_percent(grade_counts.get("VIP", 0)))

    grade_ratio_text = " / ".join(
        f"{grade} {format_percent(ratio)}"
        for grade, ratio in grade_counts.items()
    )
    st.caption(
        f"등급 비율: {grade_ratio_text} | "
        "누적매출은 거래처 마스터 기준, 기간 매출은 현재 필터 기준"
    )

    rank_mode = st.radio(
        "Top 10 기준",
        ["누적매출(마스터)", "기간 매출(필터)"],
        horizontal=True,
        key="customer_top10_mode",
    )

    col1, col2 = st.columns(2)

    with col1:
        type_df = customer["거래처유형"].value_counts().reset_index()
        type_df.columns = ["거래처유형", "거래처 수"]
        fig_type = px.bar(
            type_df,
            x="거래처유형",
            y="거래처 수",
            title="거래처유형별 분포",
        )
        st.plotly_chart(fig_type, width=CHART_WIDTH, key="customer_type_chart")

        region_df = customer["지역"].value_counts().reset_index()
        region_df.columns = ["지역", "거래처 수"]
        fig_region = px.bar(
            region_df,
            x="지역",
            y="거래처 수",
            title="지역별 거래처 수",
        )
        st.plotly_chart(fig_region, width=CHART_WIDTH, key="customer_region_chart")

    with col2:
        grade_df = customer.groupby("등급", as_index=False)["누적매출"].sum()
        fig_grade = px.bar(
            grade_df,
            x="등급",
            y="누적매출",
            title="등급별 누적매출",
            labels={"누적매출": "누적매출 (원)"},
        )
        apply_money_chart_format(fig_grade)
        st.plotly_chart(fig_grade, width=CHART_WIDTH, key="customer_grade_chart")

        if rank_mode == "누적매출(마스터)":
            top10 = customer.nlargest(10, "누적매출")
            value_col = "누적매출"
            chart_title = "Top 10 거래처 (누적매출)"
        else:
            if sales.empty:
                st.info("기간 매출 데이터가 없어 Top 10을 표시할 수 없습니다.")
                top10 = customer.iloc[0:0]
                value_col = "기간매출"
                chart_title = "Top 10 거래처 (기간 매출)"
            else:
                top10 = (
                    sales.groupby("거래처명", as_index=False)["매출금액"]
                    .sum()
                    .rename(columns={"매출금액": "기간매출"})
                    .nlargest(10, "기간매출")
                )
                value_col = "기간매출"
                chart_title = "Top 10 거래처 (기간 매출)"

        if not top10.empty:
            fig_top = px.bar(
                top10,
                x=value_col,
                y="거래처명",
                orientation="h",
                title=chart_title,
                labels={value_col: "금액 (원)"},
            )
            apply_money_chart_format(fig_top)
            fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_top, width=CHART_WIDTH, key="customer_top10_chart")

    export_customer = customer.copy()
    export_customer["첫거래일"] = export_customer["첫거래일"].dt.strftime("%Y-%m-%d")

    display_customer = export_customer.copy()
    display_customer["누적매출"] = display_customer["누적매출"].apply(format_currency)

    st.subheader("거래처 마스터")
    download_csv_button(export_customer, "customers_filtered.csv", key="download_customer_csv")
    st.dataframe(display_customer, width=CHART_WIDTH, hide_index=True, key="customer_detail_table")


def render_activity_tab(activity: pd.DataFrame) -> None:
    if activity.empty:
        st.info("조건에 맞는 데이터가 없습니다.")
        return

    total_activities = len(activity)
    outcome_counts = activity["활동결과"].value_counts(normalize=True) * 100
    type_counts = activity["활동유형"].value_counts()
    top_types = type_counts.head(3)

    c1, c2, c3 = st.columns(3)
    c1.metric("총 활동 건수", f"{total_activities:,}건")
    c2.metric("긍정 비율", format_percent(outcome_counts.get("긍정", 0)))
    c3.metric("주요 활동유형", top_types.index[0])

    outcome_text = " / ".join(
        f"{outcome} {format_percent(ratio)}"
        for outcome, ratio in outcome_counts.items()
    )
    type_text = " / ".join(f"{name} {count}건" for name, count in top_types.items())
    st.caption(f"활동결과: {outcome_text}")
    st.caption(f"활동유형 Top 3: {type_text}")

    col1, col2 = st.columns(2)

    with col1:
        monthly = activity.copy()
        monthly["월"] = monthly["활동일"].dt.to_period("M").astype(str)
        monthly_df = monthly.groupby("월", as_index=False).size().rename(columns={"size": "건수"})
        fig_monthly = px.line(
            monthly_df,
            x="월",
            y="건수",
            title="월별 활동 추이",
            markers=True,
        )
        st.plotly_chart(fig_monthly, width=CHART_WIDTH, key="activity_monthly_chart")

        type_df = activity["활동유형"].value_counts().reset_index()
        type_df.columns = ["활동유형", "건수"]
        fig_type = px.bar(
            type_df,
            x="활동유형",
            y="건수",
            title="활동유형별 건수",
        )
        st.plotly_chart(fig_type, width=CHART_WIDTH, key="activity_type_chart")

    with col2:
        outcome_df = activity["활동결과"].value_counts().reset_index()
        outcome_df.columns = ["활동결과", "건수"]
        fig_outcome = px.pie(
            outcome_df,
            names="활동결과",
            values="건수",
            title="활동결과 분포",
        )
        st.plotly_chart(fig_outcome, width=CHART_WIDTH, key="activity_outcome_chart")

        rep_df = activity.groupby("담당영업").agg(
            총활동=("활동ID", "count"),
            긍정=("활동결과", lambda series: (series == "긍정").sum()),
        ).reset_index()
        rep_df["긍정률"] = rep_df["긍정"] / rep_df["총활동"] * 100

        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            fig_rep_count = px.bar(
                rep_df,
                x="담당영업",
                y="총활동",
                title="담당영업별 활동 건수",
            )
            st.plotly_chart(fig_rep_count, width=CHART_WIDTH, key="activity_rep_count_chart")
        with rep_col2:
            fig_rep_rate = px.bar(
                rep_df,
                x="담당영업",
                y="긍정률",
                title="담당영업별 긍정률 (%)",
            )
            st.plotly_chart(fig_rep_rate, width=CHART_WIDTH, key="activity_rep_rate_chart")

    export_activity = activity.copy()
    export_activity["활동일"] = export_activity["활동일"].dt.strftime("%Y-%m-%d")

    st.subheader("영업활동 상세")
    download_csv_button(export_activity, "activities_filtered.csv", key="download_activity_csv")
    st.dataframe(export_activity, width=CHART_WIDTH, hide_index=True, key="activity_detail_table")


def main() -> None:
    st.set_page_config(page_title="에이텍 CRM 영업 대시보드", layout="wide")
    st.title("에이텍 CRM 영업 대시보드")

    sales, customer, activity = load_data()
    filters = render_sidebar_filters(sales, customer, activity)
    has_conflict = detect_filter_conflict(sales, filters)

    if has_conflict:
        st.sidebar.warning("기간과 분기 조건이 겹치지 않습니다.")

    filtered_sales, filtered_customer, filtered_activity = apply_filters(
        sales, customer, activity, filters
    )

    summary_text = build_filter_summary(
        filtered_sales,
        filtered_customer,
        filtered_activity,
        filters,
    )
    active_filter_text = format_active_filters(filters, sales, customer, activity)
    st.sidebar.markdown("**적용 결과**")
    st.sidebar.write(summary_text)

    st.caption(active_filter_text)
    if has_conflict:
        st.warning("기간과 분기 조건이 겹치지 않습니다.")

    tab_overview, tab_sales, tab_customer, tab_activity = st.tabs(
        ["요약", "매출", "고객", "영업활동"]
    )

    with tab_overview:
        render_overview_tab(
            filtered_sales,
            filtered_customer,
            filtered_activity,
            summary_text,
        )

    with tab_sales:
        render_sales_tab(filtered_sales, sales, customer, activity, filters)

    with tab_customer:
        render_customer_tab(filtered_customer, filtered_sales)

    with tab_activity:
        render_activity_tab(filtered_activity)


if __name__ == "__main__":
    main()

from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESOURCE_DIR = Path(__file__).parent / "resource"
CHART_WIDTH = "stretch"
DATE_PRESETS = ["전체", "올해", "작년", "최근 12개월", "사용자 지정"]
FILTER_STATE_KEYS = [
    "date_preset",
    "date_range",
    "filter_customer_names",
    "filter_quarters",
    "filter_regions",
    "filter_customer_types",
    "filter_sales_reps",
    "filter_product_groups",
    "filter_order_statuses",
    "filter_grades",
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


def multiselect_filter(
    label: str,
    options: list,
    key: str,
    container: st.delta_generator.DeltaGenerator | None = None,
) -> list:
    widget = container if container is not None else st.sidebar
    selected = widget.multiselect(label, options, default=options, key=key)
    if not selected:
        widget.warning(f"{label}: 1개 이상 선택해 주세요.")
    return selected


def resolve_date_range(preset: str, min_date: date, max_date: date) -> tuple[date, date]:
    today = date.today()

    if preset == "전체":
        return min_date, max_date
    if preset == "올해":
        start = date(today.year, 1, 1)
        end = min(today, max_date)
        return max(start, min_date), end
    if preset == "작년":
        start = date(today.year - 1, 1, 1)
        end = date(today.year - 1, 12, 31)
        return max(start, min_date), min(end, max_date)
    if preset == "최근 12개월":
        end = min(today, max_date)
        start = end - timedelta(days=365)
        return max(start, min_date), end
    return min_date, max_date


def reset_filters() -> None:
    for key in FILTER_STATE_KEYS:
        st.session_state.pop(key, None)


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

    filtered_sales = filtered_sales[
        filtered_sales["거래처명"].isin(filters["customer_names"])
    ]
    filtered_customer = filtered_customer[
        filtered_customer["거래처명"].isin(filters["customer_names"])
    ]
    filtered_activity = filtered_activity[
        filtered_activity["거래처명"].isin(filters["customer_names"])
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
    filtered_sales = filtered_sales[
        (filtered_sales["수주일"].dt.date >= start_date)
        & (filtered_sales["수주일"].dt.date <= end_date)
    ]
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
    start_date, end_date = filters["date_range"]
    in_date = sales[
        (sales["수주일"].dt.date >= start_date)
        & (sales["수주일"].dt.date <= end_date)
    ]
    in_quarter = sales[sales["분기"].isin(filters["quarters"])]
    if in_date.empty or in_quarter.empty:
        return False
    overlap = in_date[in_date["분기"].isin(filters["quarters"])]
    return overlap.empty


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


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "CSV 다운로드") -> None:
    buffer = BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
    )


def render_sidebar_filters(
    sales: pd.DataFrame,
    customer: pd.DataFrame,
    activity: pd.DataFrame,
) -> dict:
    st.sidebar.header("필터")

    if st.sidebar.button("필터 초기화", use_container_width=True):
        reset_filters()
        st.rerun()

    min_date = min(sales["수주일"].min(), activity["활동일"].min()).date()
    max_date = max(sales["수주일"].max(), activity["활동일"].max()).date()

    preset = st.sidebar.selectbox("기간 프리셋", DATE_PRESETS, key="date_preset")

    if preset == "사용자 지정":
        date_range = st.sidebar.date_input(
            "기간",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            parsed_date_range = date_range
        else:
            parsed_date_range = (min_date, max_date)
    else:
        parsed_date_range = resolve_date_range(preset, min_date, max_date)
        st.sidebar.caption(
            f"적용 기간: {parsed_date_range[0]} ~ {parsed_date_range[1]}"
        )

    customer_names = multiselect_filter(
        "거래처명",
        sorted(customer["거래처명"].unique()),
        "filter_customer_names",
    )

    quarters = multiselect_filter(
        "분기",
        sorted(sales["분기"].unique()),
        "filter_quarters",
    )
    regions = multiselect_filter(
        "지역",
        sorted(sales["지역"].unique()),
        "filter_regions",
    )
    customer_types = multiselect_filter(
        "거래처유형",
        sorted(sales["거래처유형"].unique()),
        "filter_customer_types",
    )
    sales_reps = multiselect_filter(
        "담당영업",
        sorted(sales["담당영업"].unique()),
        "filter_sales_reps",
    )

    with st.sidebar.expander("고급 필터", expanded=False) as advanced:
        product_groups = multiselect_filter(
            "제품군",
            sorted(sales["제품군"].unique()),
            "filter_product_groups",
            container=advanced,
        )
        order_statuses = multiselect_filter(
            "수주상태",
            sorted(sales["수주상태"].unique()),
            "filter_order_statuses",
            container=advanced,
        )
        grades = multiselect_filter(
            "거래처 등급",
            sorted(customer["등급"].unique()),
            "filter_grades",
            container=advanced,
        )

    multiselect_values = [
        customer_names,
        quarters,
        regions,
        customer_types,
        sales_reps,
        product_groups,
        order_statuses,
        grades,
    ]

    return {
        "date_range": parsed_date_range,
        "customer_names": customer_names,
        "quarters": quarters,
        "regions": regions,
        "customer_types": customer_types,
        "sales_reps": sales_reps,
        "product_groups": product_groups,
        "order_statuses": order_statuses,
        "grades": grades,
        "invalid_selection": any(not value for value in multiselect_values),
    }


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
            st.plotly_chart(fig_quarter, width=CHART_WIDTH)
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
            st.plotly_chart(fig_outcome, width=CHART_WIDTH)
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
        st.plotly_chart(fig_quarter, width=CHART_WIDTH)

        product_df = sales.groupby("제품군", as_index=False)["매출금액"].sum()
        fig_product = px.pie(
            product_df,
            names="제품군",
            values="매출금액",
            title="제품군별 매출",
        )
        st.plotly_chart(fig_product, width=CHART_WIDTH)

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
        st.plotly_chart(fig_region, width=CHART_WIDTH)

        rep_df = sales.groupby("담당영업", as_index=False)["매출금액"].sum()
        fig_rep = px.bar(
            rep_df,
            x="담당영업",
            y="매출금액",
            title="담당영업별 매출",
            labels={"매출금액": "매출금액 (원)"},
        )
        apply_money_chart_format(fig_rep)
        st.plotly_chart(fig_rep, width=CHART_WIDTH)

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
        st.plotly_chart(fig_status_count, width=CHART_WIDTH)
    with status_col2:
        fig_status_amount = px.bar(
            status_df,
            x="수주상태",
            y="금액",
            title="수주상태별 매출금액",
            labels={"금액": "매출금액 (원)"},
        )
        apply_money_chart_format(fig_status_amount)
        st.plotly_chart(fig_status_amount, width=CHART_WIDTH)

    export_sales = sales.copy()
    export_sales["수주일"] = export_sales["수주일"].dt.strftime("%Y-%m-%d")
    export_sales["납품일"] = export_sales["납품일"].dt.strftime("%Y-%m-%d")

    display_sales = export_sales.copy()
    for col in ["수주금액", "매출금액", "미수금"]:
        display_sales[col] = display_sales[col].apply(format_currency)

    st.subheader("거래 상세")
    download_csv_button(export_sales, "sales_filtered.csv")
    st.dataframe(display_sales, width=CHART_WIDTH, hide_index=True)


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
        st.plotly_chart(fig_type, width=CHART_WIDTH)

        region_df = customer["지역"].value_counts().reset_index()
        region_df.columns = ["지역", "거래처 수"]
        fig_region = px.bar(
            region_df,
            x="지역",
            y="거래처 수",
            title="지역별 거래처 수",
        )
        st.plotly_chart(fig_region, width=CHART_WIDTH)

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
        st.plotly_chart(fig_grade, width=CHART_WIDTH)

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
            st.plotly_chart(fig_top, width=CHART_WIDTH)

    export_customer = customer.copy()
    export_customer["첫거래일"] = export_customer["첫거래일"].dt.strftime("%Y-%m-%d")

    display_customer = export_customer.copy()
    display_customer["누적매출"] = display_customer["누적매출"].apply(format_currency)

    st.subheader("거래처 마스터")
    download_csv_button(export_customer, "customers_filtered.csv")
    st.dataframe(display_customer, width=CHART_WIDTH, hide_index=True)


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
        st.plotly_chart(fig_monthly, width=CHART_WIDTH)

        type_df = activity["활동유형"].value_counts().reset_index()
        type_df.columns = ["활동유형", "건수"]
        fig_type = px.bar(
            type_df,
            x="활동유형",
            y="건수",
            title="활동유형별 건수",
        )
        st.plotly_chart(fig_type, width=CHART_WIDTH)

    with col2:
        outcome_df = activity["활동결과"].value_counts().reset_index()
        outcome_df.columns = ["활동결과", "건수"]
        fig_outcome = px.pie(
            outcome_df,
            names="활동결과",
            values="건수",
            title="활동결과 분포",
        )
        st.plotly_chart(fig_outcome, width=CHART_WIDTH)

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
            st.plotly_chart(fig_rep_count, width=CHART_WIDTH)
        with rep_col2:
            fig_rep_rate = px.bar(
                rep_df,
                x="담당영업",
                y="긍정률",
                title="담당영업별 긍정률 (%)",
            )
            st.plotly_chart(fig_rep_rate, width=CHART_WIDTH)

    export_activity = activity.copy()
    export_activity["활동일"] = export_activity["활동일"].dt.strftime("%Y-%m-%d")

    st.subheader("영업활동 상세")
    download_csv_button(export_activity, "activities_filtered.csv")
    st.dataframe(export_activity, width=CHART_WIDTH, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="에이텍 CRM 영업 대시보드", layout="wide")
    st.title("에이텍 CRM 영업 대시보드")

    sales, customer, activity = load_data()
    filters = render_sidebar_filters(sales, customer, activity)

    if detect_filter_conflict(sales, filters):
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
    st.sidebar.markdown("**적용 결과**")
    st.sidebar.write(summary_text)

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

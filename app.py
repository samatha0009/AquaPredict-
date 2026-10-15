"""AquaPredict: enter how you used water today and see where it goes, where it is wasted,
how to save it, and what it costs. Run with:  streamlit run app.py"""
import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from anomaly_detection import ADVICE, analyse_day
from generate_dataset import make_training_data
from model import ModelError, predict_expected, train_model
from preprocessing import DataError, load_training_data

DATA_PATH = Path(__file__).parent / "data" / "training_data.csv"
DAYS_IN_MONTH = 30

# A colour for every area of the house, and colours for the status of a reading
AREA_COLORS = {"Bathing": "#2E86DE", "Washroom": "#8E44AD", "Kitchen": "#F39C12",
               "Laundry": "#16A085", "Garden and outdoor": "#27AE60", "Other": "#E84393"}
RED, YELLOW, GREY, BLUE, GREEN, PURPLE = "#E74C3C", "#F1C40F", "#B8C4CE", "#0072FF", "#27AE60", "#8E44AD"

STYLE = """
<style>
.hero {background: linear-gradient(120deg, #0072ff, #00c6a7 60%, #7ed957); padding: 28px 30px;
       border-radius: 18px; margin-bottom: 18px;}
.hero h1 {color: #fff; margin: 0; font-size: 2.7rem; padding: 0;}
.hero p {color: #f2fffb; margin: 6px 0 0 0; font-size: 1.05rem;}
.sec {border-left: 8px solid var(--c); padding: 4px 0 4px 14px; margin: 28px 0 10px 0;
      font-size: 1.5rem; font-weight: 700;}
.mini {font-weight: 700; font-size: 1.05rem; margin: 10px 0 0 0; color: var(--c);}
[data-testid="stForm"] {background: rgba(0, 198, 167, 0.08); border: 2px solid rgba(0, 114, 255, 0.35);
                        border-radius: 18px; padding: 22px;}
[data-testid="stMetric"] {background: rgba(0, 114, 255, 0.09); border-left: 6px solid #00a8e8;
                          border-radius: 12px; padding: 12px 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);}
[data-testid="stFormSubmitButton"] button, .stButton button {
    background: linear-gradient(90deg, #0072ff, #00c6a7); color: #fff; border: 0;
    border-radius: 12px; padding: 10px 28px; font-weight: 700;}
[data-testid="stFormSubmitButton"] button:hover, .stButton button:hover {filter: brightness(1.1); color: #fff;}
</style>
"""

st.set_page_config(page_title="AquaPredict", layout="centered")
st.markdown(STYLE, unsafe_allow_html=True)


def section(title, color):
    """A big coloured heading."""
    st.markdown(f'<div class="sec" style="--c:{color}">{title}</div>', unsafe_allow_html=True)


def mini(title, color):
    st.markdown(f'<div class="mini" style="--c:{color}">{title}</div>', unsafe_allow_html=True)


@st.cache_resource
def load_everything():
    """Cleans the training data and trains the model (runs once)."""
    df, _ = load_training_data(DATA_PATH if DATA_PATH.exists() else make_training_data())
    return train_model(df)


try:
    model = load_everything()
except (DataError, ModelError) as error:
    st.error(str(error))
    st.stop()

for key in ("entries", "bills"):
    st.session_state.setdefault(key, [])


def cost(litres_per_day, rate):
    """Monthly cost in rupees for a daily usage in litres."""
    return litres_per_day * DAYS_IN_MONTH / 1000 * rate


st.markdown('<div class="hero"><h1>AquaPredict</h1>'
            '<p>Predicting Water Wastage Using Machine Learning</p></div>', unsafe_allow_html=True)
st.write("Enter how much water your home used today. AquaPredict shows where the water goes, "
         "where wastage may be happening, how to reduce it, and what it means for your water bill.")

section("Enter Water Usage Details Below:", BLUE)

with st.expander("Not sure how many litres? Use these typical values"):
    st.markdown("- 1 bucket = about 15 litres\n- 1 washroom flush = about 6 to 10 litres\n"
                "- 1 minute of shower = about 9 litres\n- 1 washing machine load = about 60 litres\n"
                "- 1 minute of running tap = about 6 litres\n- 1 minute of hose = about 15 litres")

with st.form("details"):
    chosen = st.date_input("Date", value=datetime.date.today())
    people = st.number_input("Number of people in the house", 1, 15, 4)
    has_garden = st.selectbox("Do you have a garden or lawn?", ["No", "Yes"]) == "Yes"
    mini("Water used today (litres)", BLUE)
    bathing = st.number_input("Bathing and showers", 0.0, 5000.0, 180.0, 5.0)
    washroom = st.number_input("Washroom (toilet flushing, hand washing)", 0.0, 5000.0, 120.0, 5.0)
    kitchen = st.number_input("Kitchen (cooking, drinking, dishes)", 0.0, 5000.0, 100.0, 5.0)
    laundry = st.number_input("Laundry (washing clothes)", 0.0, 5000.0, 80.0, 5.0)
    garden = st.number_input("Garden and outdoor (0 if no garden)", 0.0, 5000.0, 0.0, 5.0)
    other = st.number_input("Other (cleaning, vehicles, anything else)", 0.0, 5000.0, 40.0, 5.0)
    mini("Your water bill", PURPLE)
    rate = st.number_input("Water rate (₹ per 1,000 litres)", 0.0, 500.0, 12.0, 0.5,
                           help="Example value. Use the rate printed on your water bill.")
    budget = st.number_input("Your monthly water bill budget (₹)", 0.0, 100000.0, 300.0, 10.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    actual = {"Bathing": bathing, "Washroom": washroom, "Kitchen": kitchen, "Laundry": laundry,
              "Garden and outdoor": garden if has_garden else 0.0, "Other": other}
    expected = predict_expected(model, chosen, people, has_garden)
    result = analyse_day(expected, actual, people, model["sigma_pp"])
    if not has_garden:
        result = result[result["Area"] != "Garden and outdoor"]

    total, total_expected = result["Actual (L)"].sum(), result["Expected (L)"].sum()
    wasted = result[result["Status"] != "Normal"]
    save_litres = wasted["Extra (L)"].sum()
    waste_areas = result[result["Status"] == "Possible wastage"]

    section("Result", GREEN)
    if not waste_areas.empty:
        names = ", ".join(waste_areas["Area"])
        high = (waste_areas["Severity"] == "High").any()
        (st.error if high else st.warning)(
            f"{'HIGH' if high else 'MEDIUM'} ALERT: possible water wastage in: {names}. "
            "This is possible wastage, not a confirmed leak. Please check these areas.")
        st.toast(f"Possible wastage found in {names}.")
    elif not wasted.empty:
        st.warning("Some areas are higher than usual: " + ", ".join(wasted["Area"]) + ".")
        st.toast("Some areas are using more water than usual.")
    else:
        st.success("Your water usage looks normal in every area.")
        st.toast("Your usage looks normal.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Used today (L)", f"{total:,.0f}")
    c2.metric("Expected (L)", f"{total_expected:,.0f}")
    c3.metric("Difference", f"{total - total_expected:+,.0f} L",
              f"{(total / total_expected - 1) * 100:+.0f}%" if total_expected else None,
              delta_color="inverse")

    # ----- Where does the water go? -----
    section("Where your water goes", "#2E86DE")
    used = result[result["Actual (L)"] > 0]
    fig = go.Figure(go.Pie(labels=used["Area"], values=used["Actual (L)"], hole=0.5,
                           textinfo="label+percent",
                           marker=dict(colors=[AREA_COLORS[a] for a in used["Area"]])))
    fig.update_layout(height=340, margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    top = used.sort_values("Actual (L)", ascending=False).iloc[0]
    st.write(f"**{top['Area']}** uses the most water: {top['Actual (L)']:,.0f} L, "
             f"{top['Actual (L)'] / total * 100:.0f}% of today's total.")

    # ----- Where is more water used than normal? -----
    section("Where you use more than expected", "#F39C12")
    outline = [RED if s == "Possible wastage" else YELLOW if s == "Higher than usual" else "rgba(0,0,0,0)"
               for s in result["Status"]]
    fig = go.Figure()
    fig.add_bar(x=result["Area"], y=result["Expected (L)"], name="Expected", marker_color=GREY)
    fig.add_bar(x=result["Area"], y=result["Actual (L)"], name="Actual",
                marker=dict(color=[AREA_COLORS[a] for a in result["Area"]],
                            line=dict(color=outline, width=4)))
    fig.update_layout(barmode="group", height=360, margin=dict(t=10, b=10), yaxis_title="Litres",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("Grey = expected. Coloured = your actual use. Red outline = possible wastage. "
               "Yellow outline = higher than usual.")

    table = result.copy()
    for col in ["Actual (L)", "Expected (L)", "Extra (L)"]:
        table[col] = table[col].round(1)
    table["Compared with expected"] = table["Compared with expected"].map(lambda v: f"{v:+.0f}%")
    st.dataframe(table, width="stretch", hide_index=True)

    # ----- How to reduce it -----
    section("How to reduce your water use", "#16A085")
    if wasted.empty:
        st.success("Nothing to fix today. Keep it up!")
    else:
        plan = wasted.copy()
        plan["You can save (L/day)"] = plan["Extra (L)"].round(0)
        plan["Bill saving (₹/month)"] = plan["Extra (L)"].map(lambda e: round(cost(e, rate)))
        plan["What to do"] = plan["Area"].map(ADVICE)
        st.dataframe(plan[["Area", "You can save (L/day)", "Bill saving (₹/month)", "What to do"]],
                     width="stretch", hide_index=True)
        st.success(f"If you bring these areas back to normal, you save about {save_litres:,.0f} litres "
                   f"a day and about ₹{cost(save_litres, rate):,.0f} a month.")

    # ----- Water bill -----
    section("Your water bill", PURPLE)
    now, after, normal = cost(total, rate), cost(total - save_litres, rate), cost(total_expected, rate)
    b1, b2, b3 = st.columns(3)
    b1.metric("Projected monthly bill", f"₹{now:,.0f}")
    b2.metric("If you fix the extra use", f"₹{after:,.0f}", f"-₹{now - after:,.0f}" if now > after else None,
              delta_color="inverse")
    b3.metric("Your budget", f"₹{budget:,.0f}")
    st.caption("Projected bill = today's usage x 30 days x your water rate. It assumes every day is like today.")
    if budget > 0:
        if now > budget:
            allowed = budget / rate * 1000 / DAYS_IN_MONTH if rate > 0 else 0
            st.error(f"At today's usage you will be ₹{now - budget:,.0f} over your budget. To stay within "
                     f"₹{budget:,.0f}, use about {allowed:,.0f} litres a day ({allowed / people:,.0f} per person).")
        else:
            st.success(f"At today's usage you stay ₹{budget - now:,.0f} within your budget.")
    fig = go.Figure(go.Bar(x=["If usage stays like today", "After fixing extra use", "Normal usage"],
                           y=[now, after, normal], marker_color=[RED, GREEN, BLUE]))
    if budget > 0:
        fig.add_hline(y=budget, line_dash="dash", annotation_text="Your budget")
    fig.update_layout(height=320, margin=dict(t=10, b=10), yaxis_title="Monthly bill (₹)")
    st.plotly_chart(fig, width="stretch")

    st.session_state.entries.append({
        "Date": f"{chosen:%d %b %Y}", "People": people, "Used (L)": round(total),
        "Expected (L)": round(total_expected), "Extra use (L)": round(save_litres),
        "Projected bill (₹)": round(now),
        "Biggest problem": waste_areas.sort_values("Extra (L)").iloc[-1]["Area"]
        if not waste_areas.empty else "None"})

# ----- Progress over time -----
if st.session_state.entries:
    section("Is your water use going down?", "#E84393")
    log = pd.DataFrame(st.session_state.entries)
    st.dataframe(log, width="stretch", hide_index=True)
    labels = [f"{i + 1}. {d}" for i, d in enumerate(log["Date"])]
    fig = go.Figure()
    fig.add_scatter(x=labels, y=log["Used (L)"], name="Used", line_color=BLUE, mode="lines+markers")
    fig.add_scatter(x=labels, y=log["Expected (L)"], name="Expected",
                    line=dict(color=GREEN, dash="dash"), mode="lines+markers")
    fig.update_layout(height=300, margin=dict(t=10, b=10), yaxis_title="Litres per day",
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    if len(log) >= 2:
        per_first = log["Used (L)"].iloc[0] / log["People"].iloc[0]
        per_last = log["Used (L)"].iloc[-1] / log["People"].iloc[-1]
        change = (per_last / per_first - 1) * 100 if per_first else 0
        (st.success if change < 0 else st.info)(
            f"Water use per person is {abs(change):.0f}% {'lower' if change < 0 else 'higher'} "
            "than your first entry.")
    if st.button("Clear entries"):
        st.session_state.entries = []
        st.rerun()

# ----- Bill tracker -----
section("Keep your water bills", "#F39C12")
with st.form("bill_form"):
    month = st.text_input("Month (for example Oct 2026)")
    amount = st.number_input("Bill amount paid (₹)", 0.0, 100000.0, 0.0, 10.0)
    saved = st.form_submit_button("Save bill")
if saved:
    if month.strip() and amount > 0:
        st.session_state.bills.append({"Month": month.strip(), "Bill (₹)": round(amount)})
        st.toast("Bill saved.")
    else:
        st.error("Please enter the month and the bill amount.")
if st.session_state.bills:
    bills = pd.DataFrame(st.session_state.bills)
    fig = go.Figure(go.Bar(x=bills["Month"], y=bills["Bill (₹)"], marker_color=PURPLE))
    fig.update_layout(height=280, margin=dict(t=10, b=10), yaxis_title="Bill (₹)")
    st.plotly_chart(fig, width="stretch")
    k1, k2 = st.columns(2)
    k1.metric("Average bill", f"₹{bills['Bill (₹)'].mean():,.0f}")
    if len(bills) >= 2:
        change = bills["Bill (₹)"].iloc[-1] - bills["Bill (₹)"].iloc[-2]
        k2.metric("Latest vs previous bill", f"₹{bills['Bill (₹)'].iloc[-1]:,.0f}", f"{change:+,.0f}",
                  delta_color="inverse")
    st.caption("Saved bills stay only while this page is open. Refreshing the page clears them.")
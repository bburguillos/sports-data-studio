
import math
import statistics
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sports Data Studio", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top right, rgba(37,99,235,.10), transparent 30%),
        linear-gradient(180deg,#071426 0%,#0c1b31 100%);
    color:#f8fafc;
}
.block-container {padding-top:1.6rem; max-width:1200px;}
h1,h2,h3 {color:#ffffff;}
.studio-card {
    background:rgba(15,39,71,.94);
    border:1px solid rgba(96,165,250,.38);
    border-radius:16px;
    padding:16px 18px;
    margin:8px 0 16px;
}
.studio-step {
    font-size:.78rem;
    letter-spacing:.09em;
    font-weight:900;
    color:#93c5fd;
    text-transform:uppercase;
}
.prediction-box {
    background:rgba(30,64,175,.16);
    border:1px solid rgba(96,165,250,.28);
    border-radius:14px;
    padding:12px 14px;
    margin-bottom:12px;
}
div.stButton > button,
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: 1px solid #60a5fa !important;
    border-radius: 12px !important;
    min-height: 44px !important;
    font-weight: 850 !important;
}
div.stButton > button *,
div[data-testid="stDownloadButton"] > button * {color:#ffffff !important;}
input, textarea, [data-baseweb="select"] > div {
    background:#ffffff !important;
    color:#111827 !important;
}
div[data-testid="stMetric"] {
    background:rgba(255,255,255,.05);
    border:1px solid rgba(148,163,184,.18);
    border-radius:12px;
    padding:10px 12px;
}
</style>
""", unsafe_allow_html=True)

def parse_numeric_text(text):
    if not str(text or "").strip():
        return [], []
    raw = str(text).replace("\t", ",").replace("\n", ",").replace(";", ",")
    tokens = []
    for chunk in raw.split(","):
        tokens.extend([p for p in chunk.strip().split() if p])
    values, bad = [], []
    for token in tokens:
        cleaned = token.replace("$","").replace("%","").strip()
        try:
            values.append(float(cleaned))
        except ValueError:
            bad.append(token)
    return values, bad

def parse_categorical_text(text):
    if not str(text or "").strip():
        return []
    raw = str(text).replace("\t", "\n").replace(",", "\n").replace(";", "\n")
    return [x.strip() for x in raw.splitlines() if x.strip()]

def fmt(x):
    if x is None:
        return "—"
    if abs(float(x) - round(float(x))) < 1e-10:
        return str(int(round(float(x))))
    return f"{float(x):.2f}".rstrip("0").rstrip(".")

def calc_mode(values):
    if not values:
        return "—"
    c = Counter(values)
    top = max(c.values())
    if top == 1:
        return "No mode"
    return ", ".join(fmt(x) for x in sorted(k for k,v in c.items() if v == top))

def quartiles(values):
    arr = np.array(sorted(values), dtype=float)
    if len(arr) == 0:
        return None, None
    return float(np.percentile(arr,25)), float(np.percentile(arr,75))

def mad(values):
    if not values:
        return None
    m = statistics.mean(values)
    return statistics.mean(abs(x-m) for x in values)

def numerical_summary(values):
    q1,q3 = quartiles(values)
    return {
        "n":len(values),
        "mean":statistics.mean(values),
        "median":statistics.median(values),
        "mode":calc_mode(values),
        "min":min(values),
        "max":max(values),
        "range":max(values)-min(values),
        "q1":q1,
        "q3":q3,
        "iqr":q3-q1,
        "mad":mad(values),
    }

def suspected_outliers(values):
    if len(values) < 4:
        return []
    q1,q3 = quartiles(values)
    iqr = q3-q1
    lo,hi = q1-1.5*iqr, q3+1.5*iqr
    return [x for x in values if x<lo or x>hi]

def show_summary_metrics(summary, prefix=""):
    cols = st.columns(4)
    cols[0].metric(f"{prefix}Mean", fmt(summary["mean"]))
    cols[1].metric(f"{prefix}Median", fmt(summary["median"]))
    cols[2].metric(f"{prefix}Mode", summary["mode"])
    cols[3].metric(f"{prefix}Range", fmt(summary["range"]))
    cols = st.columns(4)
    cols[0].metric(f"{prefix}Q1", fmt(summary["q1"]))
    cols[1].metric(f"{prefix}Q3", fmt(summary["q3"]))
    cols[2].metric(f"{prefix}IQR", fmt(summary["iqr"]))
    cols[3].metric(f"{prefix}MAD", fmt(summary["mad"]))

def frequency_table_numeric(values, bins=5):
    counts, edges = np.histogram(values, bins=bins)
    rows=[]
    for i,c in enumerate(counts):
        rows.append({
            "Interval":f"{fmt(edges[i])} to {fmt(edges[i+1])}",
            "Frequency":int(c),
            "Relative Frequency":c/len(values)
        })
    return pd.DataFrame(rows)

def make_dotplot(values,title,unit):
    fig,ax=plt.subplots(figsize=(8,3.6))
    counts=Counter(values)
    for x,c in sorted(counts.items()):
        ax.scatter([x]*c, list(range(1,c+1)), s=55)
    ax.set_title(title)
    ax.set_xlabel(unit or "Value")
    ax.set_yticks([])
    ax.grid(axis="x",alpha=.2)
    fig.tight_layout()
    return fig

def make_hist(values,title,unit,bins=6):
    fig,ax=plt.subplots(figsize=(8,4))
    ax.hist(values,bins=bins,edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(unit or "Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

def make_box(values_list,labels,title,unit):
    fig,ax=plt.subplots(figsize=(8,4))
    # Matplotlib 3.10+ renamed labels -> tick_labels and vert -> orientation.
    # Use the current API first, then fall back for older installs.
    try:
        ax.boxplot(
            values_list,
            tick_labels=labels,
            orientation="horizontal"
        )
    except TypeError:
        ax.boxplot(
            values_list,
            labels=labels,
            vert=False
        )
    ax.set_title(title)
    ax.set_xlabel(unit or "Value")
    fig.tight_layout()
    return fig

def make_line(labels,means,title,unit):
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(labels,means,marker="o")
    ax.set_title(title)
    ax.set_ylabel(unit or "Value")
    ax.set_xlabel("Time")
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig

def make_bar(categories,counts,title):
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(categories,counts)
    ax.set_title(title)
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x",rotation=25)
    fig.tight_layout()
    return fig

def make_pie(categories,counts,title):
    fig,ax=plt.subplots(figsize=(6,6))
    ax.pie(counts,labels=categories,autopct="%1.0f%%",startangle=90)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def prediction_section(mode_key,prompts,teacher_mode):
    st.markdown(
        '<div class="prediction-box"><div class="studio-step">Predict Before You Calculate</div>',
        unsafe_allow_html=True
    )
    answers={}
    for i,p in enumerate(prompts):
        answers[p]=st.text_area(
            p,
            key=f"pred_{mode_key}_{i}",
            height=80,
            placeholder="Write your prediction. A short answer is okay."
        )
    st.markdown("</div>",unsafe_allow_html=True)

    lock_key = f"predictions_locked_{mode_key}"

    if teacher_mode:
        st.session_state[lock_key] = True
    else:
        all_answered = all(str(v).strip() for v in answers.values())
        if st.button(
            "🔒 Lock In Predictions & Reveal Analysis",
            key=f"lock_predictions_{mode_key}",
            use_container_width=True,
            disabled=not all_answered
        ):
            st.session_state[lock_key] = True

        if not all_answered:
            st.caption("Answer each prediction question, then click **Lock In Predictions & Reveal Analysis**.")
        elif not st.session_state.get(lock_key, False):
            st.caption("Your predictions are ready. Click the button above to reveal the analysis.")

    return answers, bool(st.session_state.get(lock_key, False))

st.markdown("""
<div class="studio-card">
  <div class="studio-step">Sports by the Numbers</div>
  <h1 style="margin:.2rem 0 .35rem;">📊 Sports Data Studio</h1>
  <p style="margin:0;">Enter it. Graph it. Analyze it. Defend it.</p>
</div>
""",unsafe_allow_html=True)

top1,top2=st.columns([2,1])
with top1:
    mode=st.radio(
        "Choose a Data Studio mode",
        ["Analyze One Data Set","Compare Two Groups","Change Over Time","Categorical Data"],
        horizontal=True
    )
with top2:
    teacher_mode=st.toggle("🛠️ Teacher Mode",value=False)
    st.caption("Teacher Mode reveals results immediately. Student Mode requires predictions first.")

# Keep each mode's prediction lock separate. Switching modes does not carry a reveal state over.
if "last_studio_mode" not in st.session_state:
    st.session_state["last_studio_mode"] = mode
elif st.session_state["last_studio_mode"] != mode:
    st.session_state["last_studio_mode"] = mode


dataset_name=st.text_input("Dataset / activity name",placeholder="Example: 40-Yard Dash")
unit=st.text_input("Unit",placeholder="Example: seconds, points, inches")

if mode=="Analyze One Data Set":
    st.markdown("## Analyze One Data Set")
    raw=st.text_area("Paste or type your data",height=170,placeholder="6.2\n6.8\n7.1\n6.4\n7.5\n6.9")
    values,bad=parse_numeric_text(raw)
    if bad:
        st.warning("Ignored non-numeric entries: "+", ".join(bad[:10]))
    preds,reveal=prediction_section("one",[
        "About what do you predict the mean will be?",
        "Do you predict there is an outlier? Why?",
        "What do you think the graph will look like?"
    ],teacher_mode)
    if not reveal:
        st.info("In Student Mode, answer the prediction questions and click **Lock In Predictions & Reveal Analysis**.")
    if values and reveal:
        s=numerical_summary(values)
        st.markdown("### Statistical Summary")
        show_summary_metrics(s)
        outs=suspected_outliers(values)
        st.write(f"**Sample size:** {s['n']}")
        st.write("**Possible outliers using the 1.5 × IQR rule:** "+(", ".join(fmt(x) for x in outs) if outs else "None detected"))
        graph=st.selectbox("Choose a graph",["Dot Plot","Histogram","Box-and-Whisker Plot","Frequency Table"])
        title=dataset_name or "Data Set"
        if graph=="Dot Plot":
            st.pyplot(make_dotplot(values,title,unit))
        elif graph=="Histogram":
            bins=st.slider("Number of histogram bins",3,10,6)
            st.pyplot(make_hist(values,title,unit,bins))
        elif graph=="Box-and-Whisker Plot":
            st.pyplot(make_box([values],[title],title,unit))
        else:
            bins=st.slider("Number of frequency intervals",3,10,5)
            df=frequency_table_numeric(values,bins)
            df["Relative Frequency"]=(df["Relative Frequency"]*100).round(1).astype(str)+"%"
            st.dataframe(df,use_container_width=True,hide_index=True)
        st.markdown("### Defend Your Analysis")
        st.text_area("What does the data tell you? Use at least one measure of center and one measure of spread.")

elif mode=="Compare Two Groups":
    st.markdown("## Compare Two Groups")
    c1,c2=st.columns(2)
    with c1:
        name_a=st.text_input("Group A name",value="Group A")
        raw_a=st.text_area("Group A data",height=160,placeholder="Paste one column from Google Sheets")
    with c2:
        name_b=st.text_input("Group B name",value="Group B")
        raw_b=st.text_area("Group B data",height=160,placeholder="Paste one column from Google Sheets")
    a,bad_a=parse_numeric_text(raw_a)
    b,bad_b=parse_numeric_text(raw_b)
    if bad_a or bad_b:
        st.warning("Some non-numeric entries were ignored.")
    preds,reveal=prediction_section("compare",[
        "Which group do you predict will have the higher mean? Why?",
        "Which group do you predict will be more consistent? Why?",
        "Do you predict either group has an outlier?"
    ],teacher_mode)
    if not reveal:
        st.info("In Student Mode, answer the prediction questions and click **Lock In Predictions & Reveal Analysis**.")
    if a and b and reveal:
        sa,sb=numerical_summary(a),numerical_summary(b)
        st.markdown(f"### {name_a}")
        show_summary_metrics(sa)
        st.markdown(f"### {name_b}")
        show_summary_metrics(sb)
        st.markdown("### Comparison")
        higher_mean=name_a if sa["mean"]>sb["mean"] else name_b if sb["mean"]>sa["mean"] else "Same"
        more_consistent=name_a if sa["mad"]<sb["mad"] else name_b if sb["mad"]<sa["mad"] else "Same"
        st.write(f"**Higher mean:** {higher_mean}")
        st.write(f"**More consistent by MAD:** {more_consistent}")
        graph=st.selectbox("Choose a comparison graph",["Side-by-Side Box Plots","Overlaid Histograms","Summary Table"])
        title=dataset_name or "Group Comparison"
        if graph=="Side-by-Side Box Plots":
            st.pyplot(make_box([a,b],[name_a,name_b],title,unit))
        elif graph=="Overlaid Histograms":
            fig,ax=plt.subplots(figsize=(8,4))
            ax.hist(a,bins=6,alpha=.55,label=name_a,edgecolor="black")
            ax.hist(b,bins=6,alpha=.55,label=name_b,edgecolor="black")
            ax.legend(); ax.set_title(title); ax.set_xlabel(unit or "Value"); ax.set_ylabel("Frequency"); fig.tight_layout()
            st.pyplot(fig)
        else:
            df=pd.DataFrame({
                "Statistic":["n","Mean","Median","Mode","Range","Q1","Q3","IQR","MAD"],
                name_a:[sa["n"],sa["mean"],sa["median"],sa["mode"],sa["range"],sa["q1"],sa["q3"],sa["iqr"],sa["mad"]],
                name_b:[sb["n"],sb["mean"],sb["median"],sb["mode"],sb["range"],sb["q1"],sb["q3"],sb["iqr"],sb["mad"]],
            })
            st.dataframe(df,use_container_width=True,hide_index=True)
        st.markdown("### Defend Your Comparison")
        st.text_area("Which group performed differently? Support your claim with center and spread.")

elif mode=="Change Over Time":
    st.markdown("## Change Over Time")
    st.write("Use this when the **same group, team, player, or class** is measured more than once.")
    period_count=st.slider("Number of time periods",2,5,2)
    period_data=[]; labels=[]
    for i in range(period_count):
        c1,c2=st.columns([1,3])
        with c1:
            label=st.text_input(f"Time {i+1} label",value=f"Time {i+1}",key=f"time_label_{i}")
        with c2:
            raw=st.text_area(f"Data for {label}",key=f"time_data_{i}",height=110,placeholder="Paste the same group's measurements for this time period")
        vals,bad=parse_numeric_text(raw)
        labels.append(label); period_data.append(vals)
        if bad:
            st.warning(f"{label}: some entries were ignored.")
    preds,reveal=prediction_section("time",[
        "Do you predict the group improved, declined, or stayed about the same?",
        "Do you predict the group became more or less consistent?",
        "Which time period do you think will have the strongest typical performance?"
    ],teacher_mode)
    if not reveal:
        st.info("In Student Mode, answer the prediction questions and click **Lock In Predictions & Reveal Analysis**.")
    if all(period_data) and reveal:
        summaries=[numerical_summary(v) for v in period_data]
        rows=[]
        for label,s in zip(labels,summaries):
            rows.append({"Time":label,"n":s["n"],"Mean":round(s["mean"],3),"Median":round(s["median"],3),"Range":round(s["range"],3),"MAD":round(s["mad"],3),"IQR":round(s["iqr"],3)})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        first,last=summaries[0],summaries[-1]
        mean_change=last["mean"]-first["mean"]
        pct_change=(mean_change/first["mean"]*100) if first["mean"]!=0 else None
        mad_change=last["mad"]-first["mad"]
        c1,c2,c3=st.columns(3)
        c1.metric("Mean change",fmt(mean_change))
        c2.metric("Percent change in mean","N/A" if pct_change is None else f"{pct_change:.1f}%")
        c3.metric("MAD change",fmt(mad_change))
        graph=st.selectbox("Choose a change-over-time graph",["Mean Trend Line","Side-by-Side Box Plots","Summary Table Only"])
        title=dataset_name or "Change Over Time"
        if graph=="Mean Trend Line":
            st.pyplot(make_line(labels,[s["mean"] for s in summaries],title,unit))
        elif graph=="Side-by-Side Box Plots":
            st.pyplot(make_box(period_data,labels,title,unit))
        st.markdown("### Interpret the Change")
        st.text_area("What changed over time? Use center, spread, and the graph as evidence.")

else:
    st.markdown("## Categorical Data")
    entry_type=st.radio("How are you entering the data?",["Paste individual categories","Enter category counts"],horizontal=True)
    categories=[]; counts=[]
    if entry_type=="Paste individual categories":
        raw=st.text_area("Paste categories",height=180,placeholder="Football\nBasketball\nBasketball\nSoccer\nFootball")
        cats=parse_categorical_text(raw)
        if cats:
            counter=Counter(cats)
            categories=list(counter.keys()); counts=list(counter.values())
    else:
        rows=st.number_input("Number of categories",2,10,4)
        for i in range(int(rows)):
            c1,c2=st.columns([2,1])
            with c1:
                cat=st.text_input(f"Category {i+1}",key=f"cat_name_{i}")
            with c2:
                count=st.number_input(f"Count {i+1}",min_value=0,step=1,key=f"cat_count_{i}")
            if cat.strip():
                categories.append(cat.strip()); counts.append(int(count))
    preds,reveal=prediction_section("cat",[
        "Which category do you predict will be most common?",
        "Which category do you predict will make up the largest percent of the total?"
    ],teacher_mode)
    if not reveal:
        st.info("In Student Mode, answer the prediction questions and click **Lock In Predictions & Reveal Analysis**.")
    if categories and sum(counts)>0 and reveal:
        total=sum(counts)
        df=pd.DataFrame({
            "Category":categories,
            "Frequency":counts,
            "Relative Frequency":[c/total for c in counts],
            "Percent":[c/total*100 for c in counts],
        })
        df["Relative Frequency"]=df["Relative Frequency"].round(3)
        df["Percent"]=df["Percent"].round(1).astype(str)+"%"
        st.markdown("### Frequency Table")
        st.dataframe(df,use_container_width=True,hide_index=True)
        graph=st.selectbox("Choose a graph",["Bar Chart","Pie Chart"])
        title=dataset_name or "Categorical Data"
        if graph=="Bar Chart":
            st.pyplot(make_bar(categories,counts,title))
        else:
            if len(categories)>6:
                st.warning("Pie charts get difficult to read with many categories. A bar chart may be clearer.")
            st.pyplot(make_pie(categories,counts,title))
        most=categories[counts.index(max(counts))]
        st.write(f"**Most common category:** {most}")
        st.write(f"**Total responses:** {total}")
        st.markdown("### Interpret the Categories")
        st.text_area("What pattern do you notice? Use frequencies or percentages as evidence.")

st.markdown("---")
st.caption("Sports Data Studio · Built for 7th-grade sports statistics and classroom data analysis.")

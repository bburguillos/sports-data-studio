
import math
import random
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


DEBATES = [
    {
        "id":"nba_consistency_brunson_mitchell", "sport":"NBA", "kind":"consistency",
        "a":"Jalen Brunson", "b":"Donovan Mitchell", "metric":"points scored", "unit":"points",
        "question":"Who has been the more consistent scorer: Jalen Brunson or Donovan Mitchell?",
        "research":["Points scored in each player's same 8–10 most recent completed games", "The date of every game", "Whether either player left a game early"],
        "why":"Consistency requires game-by-game data. Season totals cannot show how much performances vary.",
        "source_name":"NBA Stats", "source_url":"https://www.nba.com/stats/players/boxscores"
    },
    {
        "id":"nba_typical_curry_edwards", "sport":"NBA", "kind":"typical",
        "a":"Stephen Curry", "b":"Anthony Edwards", "metric":"points scored", "unit":"points",
        "question":"Who has been the stronger typical scorer recently: Stephen Curry or Anthony Edwards?",
        "research":["Points in the same number of recent games for both players", "Games played and dates", "Any unusually high or low scoring game"],
        "why":"Mean and median describe a typical performance; checking outliers helps decide which is more representative.",
        "source_name":"NBA Stats", "source_url":"https://www.nba.com/stats/players/boxscores"
    },
    {
        "id":"nba_efficiency_jokic_giannis", "sport":"NBA", "kind":"efficiency",
        "a":"Nikola Jokić", "b":"Giannis Antetokounmpo", "metric":"field goals", "unit":"field-goal percentage",
        "question":"Who was the more efficient shooter over the same time period: Nikola Jokić or Giannis Antetokounmpo?",
        "research":["Total field goals made during one shared period", "Total field goals attempted during that period", "Use regular-season data for both players"],
        "why":"Efficiency compares makes to attempts. Total makes alone can favor the player who took more shots.",
        "source_name":"NBA Stats", "source_url":"https://www.nba.com/stats/players/traditional"
    },
    {
        "id":"wnba_consistency_wilson_collier", "sport":"WNBA", "kind":"consistency",
        "a":"A'ja Wilson", "b":"Napheesa Collier", "metric":"points scored", "unit":"points",
        "question":"Who has been the more consistent scorer: A'ja Wilson or Napheesa Collier?",
        "research":["Points from the same 8–10 completed games", "Game dates", "Games with unusually low minutes"],
        "why":"A smaller MAD means the game performances stayed closer to that player's average.",
        "source_name":"WNBA Stats", "source_url":"https://stats.wnba.com/players/boxscores/"
    },
    {
        "id":"nfl_consistency_allen_jackson", "sport":"NFL", "kind":"consistency",
        "a":"Josh Allen", "b":"Lamar Jackson", "metric":"passing yards", "unit":"yards",
        "question":"Who has been the more consistent passer: Josh Allen or Lamar Jackson?",
        "research":["Passing yards from the same 6–10 regular-season weeks", "Bye weeks or missed games", "Whether partial games are included"],
        "why":"Weekly data and MAD measure reliability better than one season total.",
        "source_name":"NFL Player Stats", "source_url":"https://www.nfl.com/stats/player-stats/"
    },
    {
        "id":"nfl_typical_jefferson_chase", "sport":"NFL", "kind":"typical",
        "a":"Justin Jefferson", "b":"Ja'Marr Chase", "metric":"receiving yards", "unit":"yards",
        "question":"Who has produced the stronger typical receiving game: Justin Jefferson or Ja'Marr Chase?",
        "research":["Receiving yards from matching regular-season weeks", "Games played", "One-game highs and lows"],
        "why":"Mean and median compare typical production while the full list reveals extreme games.",
        "source_name":"NFL Player Stats", "source_url":"https://www.nfl.com/stats/player-stats/"
    },
    {
        "id":"nfl_frequency_mahomes_burrow", "sport":"NFL", "kind":"frequency",
        "a":"Patrick Mahomes", "b":"Joe Burrow", "metric":"games with 2+ passing touchdowns", "unit":"percent of games",
        "question":"Who more frequently threw at least two touchdown passes?",
        "research":["Number of qualifying games in the same season", "Total games played by each quarterback", "Use the same threshold and season"],
        "why":"Relative frequency makes the comparison fair if the quarterbacks played different numbers of games.",
        "source_name":"NFL Player Stats", "source_url":"https://www.nfl.com/stats/player-stats/"
    },
    {
        "id":"mlb_efficiency_judge_ohtani", "sport":"MLB", "kind":"frequency",
        "a":"Aaron Judge", "b":"Shohei Ohtani", "metric":"games with at least one hit", "unit":"percent of games",
        "question":"Who recorded a hit in a greater percentage of games: Aaron Judge or Shohei Ohtani?",
        "research":["Games with at least one hit during the same season or date range", "Total games played", "Use completed regular-season games"],
        "why":"A percentage compares success fairly even when the players appeared in different numbers of games.",
        "source_name":"MLB Stats", "source_url":"https://www.mlb.com/stats"
    },
    {
        "id":"mlb_consistency_soto_ramirez", "sport":"MLB", "kind":"consistency",
        "a":"Juan Soto", "b":"José Ramírez", "metric":"times reached base", "unit":"times on base",
        "question":"Who has been more consistent at reaching base: Juan Soto or José Ramírez?",
        "research":["Times reached base in the same 10 games", "At-bats or plate appearances in those games", "Any games the player did not start"],
        "why":"The game-by-game list lets MAD show whose results varied less.",
        "source_name":"MLB Stats", "source_url":"https://www.mlb.com/stats"
    },
    {
        "id":"nhl_typical_mcdavid_mackinnon", "sport":"NHL", "kind":"typical",
        "a":"Connor McDavid", "b":"Nathan MacKinnon", "metric":"points", "unit":"points",
        "question":"Who has produced the stronger typical offensive game: Connor McDavid or Nathan MacKinnon?",
        "research":["Points in the same 8–10 completed games", "Game dates", "Goals and assists if you need counterevidence"],
        "why":"Mean and median compare typical point production without relying only on a season total.",
        "source_name":"NHL Stats", "source_url":"https://www.nhl.com/stats/skaters"
    },
    {
        "id":"nhl_efficiency_shesterkin_hellebuyck", "sport":"NHL", "kind":"efficiency",
        "a":"Igor Shesterkin", "b":"Connor Hellebuyck", "metric":"saves", "unit":"save percentage",
        "question":"Who was the more efficient goaltender over the same period: Igor Shesterkin or Connor Hellebuyck?",
        "research":["Total saves in the shared period", "Total shots faced", "Use the same season and game type"],
        "why":"Save percentage accounts for different numbers of shots faced.",
        "source_name":"NHL Goalie Stats", "source_url":"https://www.nhl.com/stats/goalies"
    },
    {
        "id":"soccer_efficiency_haaland_mbappe", "sport":"Soccer", "kind":"efficiency",
        "a":"Erling Haaland", "b":"Kylian Mbappé", "metric":"shots converted into goals", "unit":"goal conversion percentage",
        "question":"Who converted a greater percentage of shots into goals: Erling Haaland or Kylian Mbappé?",
        "research":["Goals in one clearly named competition and season", "Total shots in that same competition", "Do not mix club and national-team data"],
        "why":"Conversion percentage compares scoring efficiency instead of only total goals.",
        "source_name":"UEFA Statistics", "source_url":"https://www.uefa.com/uefachampionsleague/statistics/"
    },
    {
        "id":"f1_consistency_verstappen_norris", "sport":"Formula 1", "kind":"consistency",
        "a":"Max Verstappen", "b":"Lando Norris", "metric":"finishing position", "unit":"place",
        "question":"Who has been the more consistent finisher: Max Verstappen or Lando Norris?",
        "research":["Finishing positions from the same races", "DNFs and how you will handle them", "Sprint and Grand Prix results should not be mixed"],
        "why":"MAD measures consistency, but for finishing place a lower average is better. State how you treated DNFs.",
        "source_name":"Formula 1 Results", "source_url":"https://www.formula1.com/en/results"
    },
    {
        "id":"golf_typical_scheffler_mcilroy", "sport":"PGA", "kind":"typical_low",
        "a":"Scottie Scheffler", "b":"Rory McIlroy", "metric":"round score", "unit":"strokes",
        "question":"Who has posted the stronger typical round: Scottie Scheffler or Rory McIlroy?",
        "research":["Round scores from the same number of recent completed rounds", "Tournament names and dates", "Whether course difficulty could affect the comparison"],
        "why":"Golf scores are reversed: a lower mean or median is stronger.",
        "source_name":"PGA TOUR Stats", "source_url":"https://www.pgatour.com/stats"
    },
]


def parse_one_number(value):
    cleaned = str(value or "").replace(",", "").replace("%", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def pct_change(start, end):
    if start in (None, 0) or end is None:
        return None
    return (end - start) / abs(start) * 100


def claim_choice_label(choice, debate):
    if choice == "Player A":
        return debate["a"]
    if choice == "Player B":
        return debate["b"]
    return "Not enough information yet"


def choose_random_debate(valid_ids):
    current = st.session_state.get("selected_debate_id")
    choices = [x for x in valid_ids if x != current] or valid_ids
    st.session_state["selected_debate_id"] = random.choice(choices)


def render_claim_debate_lab(teacher_mode):
    st.markdown("## 🗣️ Claim & Debate Lab")
    st.write("Use real athlete data to build a claim, test it with math, and defend the conclusion.")

    c1,c2 = st.columns([1,2])
    with c1:
        support = st.selectbox("Support level", ["Training Mode", "Coach Mode", "Independent Mode"])
        sports = ["All Sports"] + sorted({d["sport"] for d in DEBATES})
        sport = st.selectbox("Sport", sports)
    pool = DEBATES if sport == "All Sports" else [d for d in DEBATES if d["sport"] == sport]
    valid_ids = [d["id"] for d in pool]
    if st.session_state.get("selected_debate_id") not in valid_ids:
        st.session_state["selected_debate_id"] = valid_ids[0]
    with c2:
        selected_id = st.selectbox(
            "Choose a real-player debate",
            valid_ids,
            key="selected_debate_id",
            format_func=lambda x: next(d["question"] for d in DEBATES if d["id"] == x)
        )
        st.button(
            "🎲 Give Me Another Debate",
            use_container_width=True,
            on_click=choose_random_debate,
            args=(valid_ids,)
        )

    d = next(item for item in DEBATES if item["id"] == st.session_state["selected_debate_id"])
    st.markdown(f"""
    <div class="studio-card">
      <div class="studio-step">{d['sport']} Investigation</div>
      <h2 style="margin:.25rem 0 .45rem;">{d['question']}</h2>
      <p style="margin:0;"><b>Players:</b> {d['a']} vs. {d['b']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Step 1 — Make an initial claim")
    initial = st.radio(
        "Before researching, what do you predict?",
        ["Player A", "Player B", "Not enough information yet"],
        format_func=lambda x: claim_choice_label(x, d),
        horizontal=True,
        key=f"initial_{d['id']}"
    )
    initial_reason = st.text_area(
        "Why is that your initial prediction?",
        placeholder="This is a prediction, so it is okay if the evidence later changes your mind.",
        key=f"initial_reason_{d['id']}", height=80
    )

    st.markdown("### Step 2 — Build the research plan")
    if support == "Independent Mode":
        st.info("Decide which data would directly measure the words in the debate question.")
        student_plan = st.text_area("What will you research, and why is it relevant?", key=f"plan_{d['id']}")
        with st.expander("Check the app's suggested research plan"):
            for item in d["research"]:
                st.write("• " + item)
            st.write("**Why this works:** " + d["why"])
    else:
        st.markdown('<div class="studio-card">', unsafe_allow_html=True)
        for item in d["research"]:
            st.write("✅ " + item)
        st.write("**Why these numbers matter:** " + d["why"])
        st.markdown('</div>', unsafe_allow_html=True)
        student_plan = "; ".join(d["research"])
    st.markdown(f"**Research source:** [{d['source_name']}]({d['source_url']})")
    st.caption("Record the season or date range. Both athletes must use the same period and type of games.")
    period = st.text_input("Season or exact date range", placeholder="Example: 2025–26 regular season, games through March 1", key=f"period_{d['id']}")

    st.markdown("### Step 3 — Enter the real data")
    result = None
    evidence_lines = []
    fair_sample = True

    if d["kind"] in ("typical", "typical_low", "consistency"):
        a_col,b_col = st.columns(2)
        with a_col:
            raw_a = st.text_area(f"{d['a']} — game-by-game {d['metric']}", placeholder="Enter values separated by commas or new lines", key=f"debate_a_{d['id']}", height=150)
        with b_col:
            raw_b = st.text_area(f"{d['b']} — game-by-game {d['metric']}", placeholder="Enter values separated by commas or new lines", key=f"debate_b_{d['id']}", height=150)
        a_vals,a_bad = parse_numeric_text(raw_a)
        b_vals,b_bad = parse_numeric_text(raw_b)
        if a_bad or b_bad:
            st.warning("Some entries were not numbers and were ignored.")
        if a_vals and b_vals:
            fair_sample = len(a_vals) == len(b_vals)
            if not fair_sample:
                st.warning(f"Fair-comparison check: {d['a']} has {len(a_vals)} values and {d['b']} has {len(b_vals)}. Use matching sample sizes if possible.")
            if min(len(a_vals),len(b_vals)) < 5:
                st.warning("Your sample is small. Try to collect at least 5 games; 8–10 is stronger.")
            sa,sb = numerical_summary(a_vals),numerical_summary(b_vals)
            st.markdown("### Step 4 — Do the math")
            table = pd.DataFrame({
                "Measure":["Games", "Mean", "Median", "Range", "MAD"],
                d["a"]:[fmt(sa["n"]),fmt(sa["mean"]),fmt(sa["median"]),fmt(sa["range"]),fmt(sa["mad"])],
                d["b"]:[fmt(sb["n"]),fmt(sb["mean"]),fmt(sb["median"]),fmt(sb["range"]),fmt(sb["mad"])],
            })
            st.dataframe(table,use_container_width=True,hide_index=True)
            if d["kind"] == "consistency":
                winner = d["a"] if sa["mad"] < sb["mad"] else d["b"] if sb["mad"] < sa["mad"] else "Tie"
                evidence_lines = [f"{d['a']} MAD: {fmt(sa['mad'])} {d['unit']}", f"{d['b']} MAD: {fmt(sb['mad'])} {d['unit']}"]
                result = f"{winner} is more consistent because the smaller MAD shows performances were closer to the player's mean." if winner != "Tie" else "The players have equal MADs in this sample."
                if support == "Training Mode":
                    st.info("MAD means mean absolute deviation. For consistency, **smaller MAD = more consistent**.")
            else:
                center = st.radio("Which measure best represents a typical performance?", ["Mean", "Median"], horizontal=True, key=f"center_{d['id']}")
                av = sa[center.lower()]; bv = sb[center.lower()]
                lower_is_better = d["kind"] == "typical_low"
                if av == bv:
                    winner = "Tie"
                elif (av < bv) == lower_is_better:
                    winner = d["a"]
                else:
                    winner = d["b"]
                evidence_lines = [f"{d['a']} {center.lower()}: {fmt(av)} {d['unit']}", f"{d['b']} {center.lower()}: {fmt(bv)} {d['unit']}"]
                direction = "lower" if lower_is_better else "higher"
                result = f"{winner} has the stronger typical result because the {center.lower()} is {direction}." if winner != "Tie" else f"The players have the same {center.lower()} in this sample."
                if support == "Training Mode":
                    st.info("Use the median when an extreme game pulls the mean away from most performances. Otherwise, the mean uses every value.")
            with st.expander("See the game-by-game comparison graph"):
                st.pyplot(make_box([a_vals,b_vals],[d["a"],d["b"]],d["question"],d["unit"]))

    elif d["kind"] == "efficiency":
        a_col,b_col = st.columns(2)
        with a_col:
            st.markdown(f"**{d['a']}**")
            a_made = parse_one_number(st.text_input("Successful outcomes", key=f"a_made_{d['id']}", placeholder="Example: shots made or saves"))
            a_attempt = parse_one_number(st.text_input("Total attempts/opportunities", key=f"a_att_{d['id']}", placeholder="Example: shots attempted or shots faced"))
        with b_col:
            st.markdown(f"**{d['b']}**")
            b_made = parse_one_number(st.text_input("Successful outcomes", key=f"b_made_{d['id']}", placeholder="Example: shots made or saves"))
            b_attempt = parse_one_number(st.text_input("Total attempts/opportunities", key=f"b_att_{d['id']}", placeholder="Example: shots attempted or shots faced"))
        if all(x is not None for x in [a_made,a_attempt,b_made,b_attempt]):
            if a_attempt <= 0 or b_attempt <= 0 or a_made > a_attempt or b_made > b_attempt:
                st.error("Check the data: attempts must be positive and successes cannot exceed attempts.")
            else:
                ap,bp = a_made/a_attempt*100,b_made/b_attempt*100
                winner = d["a"] if ap > bp else d["b"] if bp > ap else "Tie"
                st.markdown("### Step 4 — Do the math")
                st.latex(r"\text{percentage}=\frac{\text{successful outcomes}}{\text{total opportunities}}\times100")
                c1,c2 = st.columns(2); c1.metric(d["a"],f"{ap:.1f}%"); c2.metric(d["b"],f"{bp:.1f}%")
                evidence_lines = [f"{d['a']}: {a_made:g}/{a_attempt:g} = {ap:.1f}%", f"{d['b']}: {b_made:g}/{b_attempt:g} = {bp:.1f}%"]
                result = f"{winner} has the higher rate by {abs(ap-bp):.1f} percentage points." if winner != "Tie" else "The players have equal percentages."

    elif d["kind"] == "frequency":
        a_col,b_col = st.columns(2)
        with a_col:
            st.markdown(f"**{d['a']}**")
            a_success = parse_one_number(st.text_input("Games meeting the condition", key=f"a_success_{d['id']}"))
            a_total = parse_one_number(st.text_input("Total games", key=f"a_total_{d['id']}"))
        with b_col:
            st.markdown(f"**{d['b']}**")
            b_success = parse_one_number(st.text_input("Games meeting the condition", key=f"b_success_{d['id']}"))
            b_total = parse_one_number(st.text_input("Total games", key=f"b_total_{d['id']}"))
        if all(x is not None for x in [a_success,a_total,b_success,b_total]):
            if a_total <= 0 or b_total <= 0 or a_success > a_total or b_success > b_total:
                st.error("Check the data: total games must be positive and qualifying games cannot exceed total games.")
            else:
                ap,bp = a_success/a_total*100,b_success/b_total*100
                winner = d["a"] if ap > bp else d["b"] if bp > ap else "Tie"
                st.markdown("### Step 4 — Do the math")
                st.latex(r"\text{relative frequency}=\frac{\text{qualifying games}}{\text{total games}}\times100")
                c1,c2=st.columns(2); c1.metric(d["a"],f"{ap:.1f}%"); c2.metric(d["b"],f"{bp:.1f}%")
                evidence_lines = [f"{d['a']}: {a_success:g}/{a_total:g} = {ap:.1f}%", f"{d['b']}: {b_success:g}/{b_total:g} = {bp:.1f}%"]
                result = f"{winner} met the condition more often by {abs(ap-bp):.1f} percentage points." if winner != "Tie" else "The players have equal relative frequencies."

    if result:
        st.markdown("### Step 5 — Decide what the evidence means")
        st.success(result)
        st.write("**Evidence collected:**")
        for line in evidence_lines:
            st.write("• " + line)
        evidence_rating = st.radio(
            "How does this evidence affect your original claim?",
            ["Supports it", "Partially supports it", "Contradicts it", "Not enough evidence"],
            horizontal=True, key=f"rating_{d['id']}"
        )
        limitation = st.text_area(
            "Name one limitation or piece of counterevidence.",
            placeholder="Examples: small sample, different opponents, an outlier, minutes played, course difficulty…",
            key=f"limitation_{d['id']}", height=85
        )

        st.markdown("### Step 6 — Build and defend the final claim")
        final_claim = st.text_area(
            "Final argument",
            placeholder=f"I claim ___ because the data shows ___. This matters because ___. One limitation is ___; however, ___.",
            key=f"final_{d['id']}", height=150
        )
        challenge_map = {
            "consistency":f"You used MAD to discuss consistency. Does {winner} also have a strong mean, or could the player be consistently lower?",
            "typical":f"Would your conclusion change if you used the other measure of center or removed an outlier?",
            "typical_low":f"Could course difficulty explain some of the difference in round scores?",
            "efficiency":f"Does the higher percentage come from enough attempts to be convincing? Defend your sample size.",
            "frequency":f"Is the chosen condition the fairest definition of success, or would another threshold change the conclusion?",
        }
        st.markdown("#### 🎤 Challenge My Argument")
        st.warning(challenge_map[d["kind"]])
        defense = st.text_area("Your response to the opposing analyst", key=f"defense_{d['id']}", height=110)

        checklist = {
            "Initial prediction": bool(initial_reason.strip()),
            "Time period identified": bool(period.strip()),
            "Fair sample sizes": fair_sample,
            "Counterevidence considered": bool(limitation.strip()),
            "Final argument written": bool(final_claim.strip()),
            "Challenge answered": bool(defense.strip()),
        }
        completed = sum(checklist.values())
        st.progress(completed/len(checklist), text=f"Investigation checklist: {completed}/{len(checklist)} complete")
        if teacher_mode or completed == len(checklist):
            report = "\n".join([
                "SPORTS DATA STUDIO — CLAIM & DEBATE LAB", "",
                f"Question: {d['question']}", f"Period: {period}",
                f"Initial claim: {claim_choice_label(initial,d)}", f"Initial reasoning: {initial_reason}", "",
                "Evidence:", *[f"- {x}" for x in evidence_lines], f"Math interpretation: {result}",
                f"Evidence rating: {evidence_rating}", f"Limitation/counterevidence: {limitation}", "",
                f"Final argument: {final_claim}", f"Challenge: {challenge_map[d['kind']]}", f"Defense: {defense}", "",
                f"Data source: {d['source_name']} — {d['source_url']}"
            ])
            st.download_button("⬇️ Download Investigation Report", report, file_name=f"claim_debate_{d['id']}.txt", mime="text/plain", use_container_width=True)

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
        ["Claim & Debate Lab","Analyze One Data Set","Compare Two Groups","Change Over Time","Categorical Data"],
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


dataset_name=""
unit=""
if mode != "Claim & Debate Lab":
    dataset_name=st.text_input("Dataset / activity name",placeholder="Example: 40-Yard Dash")
    unit=st.text_input("Unit",placeholder="Example: seconds, points, inches")

if mode=="Claim & Debate Lab":
    render_claim_debate_lab(teacher_mode)

elif mode=="Analyze One Data Set":
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

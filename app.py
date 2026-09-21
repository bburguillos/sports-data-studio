
import math
import random
import statistics
from collections import Counter
from datetime import date
from html import escape
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

st.set_page_config(page_title="Sports Data Studio", page_icon="📊", layout="wide")

st.markdown("""
<style>
:root {
    color-scheme: light !important;
    --background-color: #f4f8ff !important;
    --secondary-background-color: #ffffff !important;
    --text-color: #0b1220 !important;
    --primary-color: #1d4ed8 !important;
}
.stApp {
    background:
        radial-gradient(circle at top right, rgba(59,130,246,.16), transparent 34%),
        linear-gradient(180deg,#f8fbff 0%,#e8f1ff 100%);
    color:#0f172a;
}
.block-container {padding-top:1.6rem; max-width:1200px;}
h1,h2,h3,h4 {color:#0f2747 !important;}
.stApp p,
.stApp li,
.stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stText"] {
    color:#172033 !important;
}
.stApp [data-testid="stWidgetLabel"] p {font-weight:700 !important;}
.stApp [data-testid="stCaptionContainer"] p,
.stApp small {color:#40516b !important;}
.stApp a {color:#1d4ed8 !important; font-weight:650;}
.studio-card {
    background:#ffffff;
    color:#0f172a;
    border:2px solid #93c5fd;
    border-radius:16px;
    padding:16px 18px;
    margin:8px 0 16px;
    box-shadow:0 8px 22px rgba(30,64,175,.08);
}
.studio-card p,.studio-card b {color:#172033 !important;}
.studio-step {
    font-size:.78rem;
    letter-spacing:.09em;
    font-weight:900;
    color:#1d4ed8;
    text-transform:uppercase;
}
.prediction-box {
    background:#eff6ff;
    color:#0f172a;
    border:2px solid #bfdbfe;
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
    border-color:#64748b !important;
}
input::placeholder, textarea::placeholder {color:#64748b !important; opacity:1 !important;}
div[data-testid="stMetric"] {
    background:#ffffff;
    border:1px solid #94a3b8;
    border-radius:12px;
    padding:10px 12px;
}
div[data-testid="stMetric"] * {color:#0f172a !important;}
div[data-testid="stExpander"] {
    background:#ffffff;
    border:1px solid #94a3b8 !important;
    border-radius:12px;
}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] li,
div[data-testid="stAlert"] span {color:#172033 !important;}
div[data-testid="stFileUploaderDropzone"] {background:#ffffff !important; border-color:#64748b !important;}
hr {border-color:#94a3b8 !important;}

/* Force the light palette even when a viewer or deployment has dark mode active. */
html, body, #root,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section.main,
.main,
.stApp {
    background-color:#f4f8ff !important;
    color:#0b1220 !important;
    color-scheme:light !important;
}
[data-testid="stHeader"] {background:rgba(244,248,255,.96) !important;}
[data-testid="stToolbar"] {background:#ffffff !important; color:#0b1220 !important;}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] summary,
[data-testid="stAppViewContainer"] div[data-testid="stCaptionContainer"] p,
[data-testid="stAppViewContainer"] div[data-testid="stMarkdownContainer"] p {
    color:#0b1220 !important;
}
[data-testid="stAppViewContainer"] div[data-testid="stCaptionContainer"] p {
    color:#334155 !important;
    font-weight:550 !important;
}
[data-testid="stAppViewContainer"] div.stButton button,
[data-testid="stAppViewContainer"] div.stButton button span,
[data-testid="stAppViewContainer"] div[data-testid="stDownloadButton"] button,
[data-testid="stAppViewContainer"] div[data-testid="stDownloadButton"] button span {
    color:#ffffff !important;
}
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] textarea,
[data-testid="stAppViewContainer"] [data-baseweb="select"] div {
    background-color:#ffffff !important;
    color:#0b1220 !important;
}

/* Streamlit can restore a viewer's dark preference after the app begins.
   Target the current app containers and controls directly so that cannot
   create dark-on-dark labels. */
body,
body > div,
#root,
#root > div,
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"] {
    background:#f4f8ff !important;
    background-color:#f4f8ff !important;
    background-image:none !important;
}
[data-testid="stMainBlockContainer"],
.block-container {
    background:transparent !important;
}
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span,
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] label span,
[data-testid="stToggle"] label,
[data-testid="stToggle"] label p,
[data-testid="stToggle"] label span,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[role="radiogroup"] label,
[role="radiogroup"] label p,
[role="radiogroup"] label span {
    color:#0b1220 !important;
    opacity:1 !important;
    -webkit-text-fill-color:#0b1220 !important;
}
[data-baseweb="radio"] > div:first-child,
[data-baseweb="checkbox"] > div:first-child {
    background-color:#ffffff !important;
    border-color:#475569 !important;
}
.build-badge {
    display:inline-block;
    margin-top:.65rem;
    padding:.18rem .55rem;
    border-radius:999px;
    background:#dbeafe;
    color:#1e3a8a !important;
    font-size:.72rem;
    font-weight:800;
    letter-spacing:.02em;
}
</style>
""", unsafe_allow_html=True)

def parse_numeric_text(text, special_values=None):
    if not str(text or "").strip():
        return [], []
    special_values = {str(k).upper(): float(v) for k,v in (special_values or {}).items()}
    raw = str(text).replace("\t", ",").replace("\n", ",").replace(";", ",")
    tokens = []
    for chunk in raw.split(","):
        tokens.extend([p for p in chunk.strip().split() if p])
    values, bad = [], []
    for token in tokens:
        cleaned = token.replace("$","").replace("%","").strip()
        if cleaned.upper() in special_values:
            values.append(special_values[cleaned.upper()])
            continue
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

# Additional real-player investigations. These expand the choice bank while
# keeping every question tied to one teachable statistical comparison.
DEBATES += [
    {"id":"nba_typical_doncic_sga","sport":"NBA","kind":"typical","a":"Luka Dončić","b":"Shai Gilgeous-Alexander","metric":"points scored","unit":"points","question":"Who has produced the stronger typical scoring game: Luka Dončić or Shai Gilgeous-Alexander?","research":["Points from the same 8–10 completed games","Dates and games played","Any unusually high or low performance"],"why":"Mean and median can describe a typical game, while the full list reveals possible outliers.","source_name":"NBA Stats","source_url":"https://www.nba.com/stats/players/boxscores"},
    {"id":"nba_consistency_wemby_chet","sport":"NBA","kind":"consistency","a":"Victor Wembanyama","b":"Chet Holmgren","metric":"blocks","unit":"blocks","question":"Who has been the more consistent shot blocker: Victor Wembanyama or Chet Holmgren?","research":["Blocks in the same number of recent games","Minutes played in each game","Games missed or left early"],"why":"Game-by-game data and MAD show how much each player's block totals vary.","source_name":"NBA Stats","source_url":"https://www.nba.com/stats/players/boxscores"},
    {"id":"nba_frequency_curry_lillard","sport":"NBA","kind":"frequency","a":"Stephen Curry","b":"Damian Lillard","metric":"games with 3+ made three-pointers","unit":"percent of games","question":"Who made at least three three-pointers in a greater percentage of games: Stephen Curry or Damian Lillard?","research":["Games with at least three made threes in one shared period","Total games played in that period","Use regular-season games for both"],"why":"Relative frequency makes the comparison fair when the players have different game totals.","source_name":"NBA Stats","source_url":"https://www.nba.com/stats/players/boxscores"},
    {"id":"wnba_typical_clark_ionescu","sport":"WNBA","kind":"typical","a":"Caitlin Clark","b":"Sabrina Ionescu","metric":"assists","unit":"assists","question":"Who has produced the stronger typical passing game: Caitlin Clark or Sabrina Ionescu?","research":["Assists from the same 8–10 games","Game dates","Minutes played and possible outliers"],"why":"Mean and median help compare typical assist production.","source_name":"WNBA Stats","source_url":"https://stats.wnba.com/players/boxscores/"},
    {"id":"wnba_efficiency_wilson_stewart","sport":"WNBA","kind":"efficiency","a":"A'ja Wilson","b":"Breanna Stewart","metric":"field goals","unit":"field-goal percentage","question":"Who was the more efficient shooter over the same period: A'ja Wilson or Breanna Stewart?","research":["Total field goals made","Total field goals attempted","Use the same season or date range"],"why":"Made shots divided by attempts measures shooting efficiency.","source_name":"WNBA Stats","source_url":"https://stats.wnba.com/players/traditional/"},
    {"id":"wnba_consistency_thomas_clark","sport":"WNBA","kind":"consistency","a":"Alyssa Thomas","b":"Caitlin Clark","metric":"assists","unit":"assists","question":"Who has been the more consistent playmaker: Alyssa Thomas or Caitlin Clark?","research":["Assists from matching recent games","Minutes played","Any game with limited playing time"],"why":"MAD measures how closely each player's assist totals cluster around the mean.","source_name":"WNBA Stats","source_url":"https://stats.wnba.com/players/boxscores/"},
    {"id":"nfl_typical_barkley_mccaffrey","sport":"NFL","kind":"typical","a":"Saquon Barkley","b":"Christian McCaffrey","metric":"rushing yards","unit":"yards","question":"Who has produced the stronger typical rushing game: Saquon Barkley or Christian McCaffrey?","research":["Rushing yards from the same number of games","Games played and dates","Unusually high or low games"],"why":"Mean and median compare typical production without relying on season totals alone.","source_name":"NFL Player Stats","source_url":"https://www.nfl.com/stats/player-stats/"},
    {"id":"nfl_consistency_jefferson_lamb","sport":"NFL","kind":"consistency","a":"Justin Jefferson","b":"CeeDee Lamb","metric":"receptions","unit":"receptions","question":"Who has been the more consistent pass catcher: Justin Jefferson or CeeDee Lamb?","research":["Receptions from the same 6–10 weeks","Bye weeks and missed games","Partial games caused by injury"],"why":"MAD and range describe variation from week to week.","source_name":"NFL Player Stats","source_url":"https://www.nfl.com/stats/player-stats/"},
    {"id":"nfl_efficiency_tucker_butker","sport":"NFL","kind":"efficiency","a":"Justin Tucker","b":"Harrison Butker","metric":"field goals","unit":"field-goal percentage","question":"Who converted field-goal attempts more efficiently: Justin Tucker or Harrison Butker?","research":["Field goals made in the same season","Field goals attempted","Optional: distance of attempts as counterevidence"],"why":"A percentage compares kickers fairly even when their attempt totals differ.","source_name":"NFL Player Stats","source_url":"https://www.nfl.com/stats/player-stats/"},
    {"id":"nfl_frequency_daniels_hurts","sport":"NFL","kind":"frequency","a":"Jayden Daniels","b":"Jalen Hurts","metric":"games with 2+ total touchdowns","unit":"percent of games","question":"Who recorded at least two total touchdowns in a greater percentage of games: Jayden Daniels or Jalen Hurts?","research":["Qualifying games in the same regular season","Total games played","Passing and rushing touchdowns"],"why":"Relative frequency accounts for different numbers of games played.","source_name":"NFL Player Stats","source_url":"https://www.nfl.com/stats/player-stats/"},
    {"id":"mlb_consistency_judge_ohtani_bases","sport":"MLB","kind":"consistency","a":"Aaron Judge","b":"Shohei Ohtani","metric":"total bases","unit":"bases","question":"Who has been more consistent at producing total bases: Aaron Judge or Shohei Ohtani?","research":["Total bases in the same 10 games","Games started","Any game with zero or an extreme total"],"why":"Game-by-game MAD measures how much offensive production changes.","source_name":"MLB Stats","source_url":"https://www.mlb.com/stats"},
    {"id":"mlb_consistency_skenes_skubal","sport":"MLB","kind":"consistency","a":"Paul Skenes","b":"Tarik Skubal","metric":"strikeouts","unit":"strikeouts","question":"Who has been the more consistent strikeout pitcher: Paul Skenes or Tarik Skubal?","research":["Strikeouts in the same number of starts","Innings pitched in each start","Starts with restricted pitch counts"],"why":"MAD compares start-to-start variation, while innings pitched may be important counterevidence.","source_name":"MLB Pitching Stats","source_url":"https://www.mlb.com/stats/pitching"},
    {"id":"mlb_frequency_lindor_witt","sport":"MLB","kind":"frequency","a":"Francisco Lindor","b":"Bobby Witt Jr.","metric":"multi-hit games","unit":"percent of games","question":"Who recorded multiple hits in a greater percentage of games: Francisco Lindor or Bobby Witt Jr.?","research":["Games with two or more hits in one shared period","Total games played","Use completed regular-season games"],"why":"Relative frequency compares how often the event occurred.","source_name":"MLB Stats","source_url":"https://www.mlb.com/stats"},
    {"id":"mlb_efficiency_soto_judge_walks","sport":"MLB","kind":"efficiency","a":"Juan Soto","b":"Aaron Judge","metric":"plate appearances ending in a walk","unit":"walk percentage","question":"Who drew a walk in a greater percentage of plate appearances: Juan Soto or Aaron Judge?","research":["Walks in the same season or date range","Plate appearances","Use the same competition period"],"why":"Walks divided by plate appearances measures rate instead of total opportunities.","source_name":"MLB Stats","source_url":"https://www.mlb.com/stats"},
    {"id":"nhl_frequency_matthews_draisaitl","sport":"NHL","kind":"frequency","a":"Auston Matthews","b":"Leon Draisaitl","metric":"games with at least one goal","unit":"percent of games","question":"Who scored in a greater percentage of games: Auston Matthews or Leon Draisaitl?","research":["Games with at least one goal","Total games played in the same period","Use the same game type"],"why":"Relative frequency measures how often each player scored, not just the total goals.","source_name":"NHL Stats","source_url":"https://www.nhl.com/stats/skaters"},
    {"id":"nhl_typical_makar_hughes","sport":"NHL","kind":"typical","a":"Cale Makar","b":"Quinn Hughes","metric":"points","unit":"points","question":"Who has produced the stronger typical offensive game among defensemen: Cale Makar or Quinn Hughes?","research":["Points from the same 8–10 games","Game dates","Goals and assists as possible supporting detail"],"why":"Mean and median compare typical game production.","source_name":"NHL Stats","source_url":"https://www.nhl.com/stats/skaters"},
    {"id":"nhl_consistency_sorokin_ottinger","sport":"NHL","kind":"consistency","a":"Ilya Sorokin","b":"Jake Oettinger","metric":"saves","unit":"saves","question":"Who has been the more consistent goaltender by saves: Ilya Sorokin or Jake Oettinger?","research":["Saves from the same number of starts","Shots faced","Exclude relief appearances or identify them"],"why":"MAD measures save-total consistency, while shots faced provides necessary context.","source_name":"NHL Goalie Stats","source_url":"https://www.nhl.com/stats/goalies"},
    {"id":"soccer_typical_haaland_mbappe_goals","sport":"Soccer","kind":"typical","a":"Erling Haaland","b":"Kylian Mbappé","metric":"goals","unit":"goals","question":"Who has produced the stronger typical scoring match: Erling Haaland or Kylian Mbappé?","research":["Goals in the same number of matches","Use one clearly named competition","Minutes played and substitute appearances"],"why":"Mean and median compare match-level scoring without mixing competitions.","source_name":"UEFA Statistics","source_url":"https://www.uefa.com/uefachampionsleague/statistics/"},
    {"id":"soccer_frequency_salah_vinicius","sport":"Soccer","kind":"frequency","a":"Mohamed Salah","b":"Vinícius Júnior","metric":"matches with a goal or assist","unit":"percent of matches","question":"Who contributed a goal or assist in a greater percentage of matches: Mohamed Salah or Vinícius Júnior?","research":["Matches with at least one goal or assist","Total matches in one named competition","Use the same season"],"why":"Relative frequency compares how regularly each player contributed.","source_name":"UEFA Statistics","source_url":"https://www.uefa.com/uefachampionsleague/statistics/"},
    {"id":"soccer_efficiency_palmer_saka","sport":"Soccer","kind":"efficiency","a":"Cole Palmer","b":"Bukayo Saka","metric":"shots converted into goals","unit":"goal conversion percentage","question":"Who converted shots into goals more efficiently: Cole Palmer or Bukayo Saka?","research":["Goals in the same league season","Total shots","Do not mix club and national-team matches"],"why":"Conversion percentage compares goals with shooting opportunities.","source_name":"Premier League Stats","source_url":"https://www.premierleague.com/en/stats"},
    {"id":"f1_consistency_leclerc_piastri","sport":"Formula 1","kind":"consistency","a":"Charles Leclerc","b":"Oscar Piastri","metric":"finishing position","unit":"place","question":"Who has been the more consistent Grand Prix finisher: Charles Leclerc or Oscar Piastri?","research":["Finishing positions from the same races","How DNFs will be recorded","Exclude sprint results"],"why":"MAD compares variation, but students must make and explain a fair DNF rule.","source_name":"Formula 1 Results","source_url":"https://www.formula1.com/en/results"},
    {"id":"f1_typical_hamilton_alonso","sport":"Formula 1","kind":"typical_low","a":"Lewis Hamilton","b":"Fernando Alonso","metric":"finishing position","unit":"place","question":"Who has posted the stronger typical Grand Prix finish: Lewis Hamilton or Fernando Alonso?","research":["Finishing positions from the same races","DNFs and DNS results","Use Grand Prix results only"],"why":"Mean or median can describe a typical finish; in racing, a lower value is stronger.","source_name":"Formula 1 Results","source_url":"https://www.formula1.com/en/results"},
    {"id":"pga_consistency_thomas_morikawa","sport":"PGA","kind":"consistency","a":"Justin Thomas","b":"Collin Morikawa","metric":"round score","unit":"strokes","question":"Who has been the more consistent recent golfer: Justin Thomas or Collin Morikawa?","research":["Scores from the same number of completed rounds","Tournament and course names","Rounds affected by unusual weather"],"why":"MAD measures consistency, while course and weather are possible limitations.","source_name":"PGA TOUR Stats","source_url":"https://www.pgatour.com/stats"},
    {"id":"pga_frequency_scheffler_schauffele","sport":"PGA","kind":"frequency","a":"Scottie Scheffler","b":"Xander Schauffele","metric":"top-10 tournament finishes","unit":"percent of events","question":"Who finished in the top 10 in a greater percentage of events: Scottie Scheffler or Xander Schauffele?","research":["Top-10 finishes in the same season","Total events played","Use official completed events"],"why":"Relative frequency accounts for different numbers of tournament starts.","source_name":"PGA TOUR Stats","source_url":"https://www.pgatour.com/stats"},
    {"id":"tennis_frequency_sinner_alcaraz","sport":"Tennis","kind":"frequency","a":"Jannik Sinner","b":"Carlos Alcaraz","metric":"service games won","unit":"percent of service games","question":"Who won a greater percentage of service games: Jannik Sinner or Carlos Alcaraz?","research":["Service games won in the same season and surface","Total service games played","Use singles matches only"],"why":"Relative frequency compares success fairly across different match totals.","source_name":"ATP Tour Stats","source_url":"https://www.atptour.com/en/stats"},
    {"id":"tennis_consistency_gauff_swiatek","sport":"Tennis","kind":"consistency","a":"Coco Gauff","b":"Iga Świątek","metric":"double faults per match","unit":"double faults","question":"Who has been more consistent at limiting double faults: Coco Gauff or Iga Świątek?","research":["Double faults in the same number of matches","Use the same surface if possible","Match length as possible counterevidence"],"why":"MAD compares variation, while a lower typical number is better for double faults.","source_name":"WTA Stats","source_url":"https://www.wtatennis.com/stats"},
]

SPORT_SOURCE_DEFAULTS = {
    "NBA": ("NBA Stats", "https://www.nba.com/stats/players/boxscores"),
    "WNBA": ("WNBA Stats", "https://stats.wnba.com/players/boxscores/"),
    "NFL": ("NFL Player Stats", "https://www.nfl.com/stats/player-stats/"),
    "MLB": ("MLB Stats", "https://www.mlb.com/stats"),
    "NHL": ("NHL Stats", "https://www.nhl.com/stats"),
    "Soccer": ("UEFA Statistics", "https://www.uefa.com/uefachampionsleague/statistics/"),
    "Formula 1": ("Formula 1 Results", "https://www.formula1.com/en/results"),
    "PGA": ("PGA TOUR Stats", "https://www.pgatour.com/stats"),
    "Tennis": ("ATP/WTA Statistics", "https://www.atptour.com/en/stats"),
}

# 60 additional curated matchups bring the full debate bank to 100.
# Compact specifications reuse the same research-quality rules as the original bank.
ADDITIONAL_CLAIM_SPECS = [
    ("nba_consistency_brunson_sga_ast","NBA","consistency","Jalen Brunson","Shai Gilgeous-Alexander","assists","assists","Who has been the more consistent playmaker: Jalen Brunson or Shai Gilgeous-Alexander?"),
    ("nba_typical_tatum_durant_pts","NBA","typical","Jayson Tatum","Kevin Durant","points scored","points","Who has produced the stronger typical scoring game: Jayson Tatum or Kevin Durant?"),
    ("nba_typical_jokic_sabonis_reb","NBA","typical","Nikola Jokić","Domantas Sabonis","rebounds","rebounds","Who has produced the stronger typical rebounding game: Nikola Jokić or Domantas Sabonis?"),
    ("nba_typical_curry_young_3pm","NBA","typical","Stephen Curry","Trae Young","three-pointers made","three-pointers","Who has made more three-pointers in a typical game: Stephen Curry or Trae Young?"),
    ("nba_efficiency_edwards_booker_fg","NBA","efficiency","Anthony Edwards","Devin Booker","field goals","field-goal percentage","Who has been the more efficient shooter: Anthony Edwards or Devin Booker?"),
    ("nba_consistency_wemby_gobert_blk","NBA","consistency","Victor Wembanyama","Rudy Gobert","blocks","blocks","Who has been the more consistent shot blocker: Victor Wembanyama or Rudy Gobert?"),
    ("nba_typical_doncic_haliburton_ast","NBA","typical","Luka Dončić","Tyrese Haliburton","assists","assists","Who has produced the stronger typical assist game: Luka Dončić or Tyrese Haliburton?"),
    ("nba_frequency_fox_maxey_25","NBA","frequency","De'Aaron Fox","Tyrese Maxey","games with at least 25 points","percent of games","Who scored at least 25 points in a greater percentage of games: De'Aaron Fox or Tyrese Maxey?"),
    ("nba_efficiency_giannis_zion_fg","NBA","efficiency","Giannis Antetokounmpo","Zion Williamson","field goals","field-goal percentage","Who converted field-goal attempts more efficiently: Giannis Antetokounmpo or Zion Williamson?"),
    ("nba_consistency_brown_butler_stl","NBA","consistency","Jaylen Brown","Jimmy Butler","steals","steals","Who has been the more consistent ball thief: Jaylen Brown or Jimmy Butler?"),
    ("wnba_typical_clark_thomas_ast","WNBA","typical","Caitlin Clark","Alyssa Thomas","assists","assists","Who has produced the stronger typical assist game: Caitlin Clark or Alyssa Thomas?"),
    ("wnba_typical_wilson_stewart_reb","WNBA","typical","A'ja Wilson","Breanna Stewart","rebounds","rebounds","Who has produced the stronger typical rebounding game: A'ja Wilson or Breanna Stewart?"),
    ("wnba_efficiency_ionescu_plum_3pt","WNBA","efficiency","Sabrina Ionescu","Kelsey Plum","three-pointers","three-point percentage","Who has converted three-point attempts more efficiently: Sabrina Ionescu or Kelsey Plum?"),
    ("wnba_consistency_collier_thomas_stl","WNBA","consistency","Napheesa Collier","Alyssa Thomas","steals","steals","Who has been more consistent at producing steals: Napheesa Collier or Alyssa Thomas?"),
    ("wnba_frequency_ogunbowale_loyd_20","WNBA","frequency","Arike Ogunbowale","Jewell Loyd","games with at least 20 points","percent of games","Who scored at least 20 points in a greater percentage of games: Arike Ogunbowale or Jewell Loyd?"),
    ("wnba_consistency_griner_jones_blk","WNBA","consistency","Brittney Griner","Jonquel Jones","blocks","blocks","Who has been the more consistent shot blocker: Brittney Griner or Jonquel Jones?"),
    ("nfl_typical_allen_mahomes_pass","NFL","typical","Josh Allen","Patrick Mahomes","passing yards","yards","Who has produced the stronger typical passing game: Josh Allen or Patrick Mahomes?"),
    ("nfl_typical_jackson_daniels_rush","NFL","typical","Lamar Jackson","Jayden Daniels","rushing yards","yards","Who has produced the stronger typical rushing game among quarterbacks: Lamar Jackson or Jayden Daniels?"),
    ("nfl_consistency_chase_jefferson_rec","NFL","consistency","Ja'Marr Chase","Justin Jefferson","receptions","receptions","Who has been the more consistent pass catcher: Ja'Marr Chase or Justin Jefferson?"),
    ("nfl_consistency_barkley_henry_rush","NFL","consistency","Saquon Barkley","Derrick Henry","rushing yards","yards","Who has been the more consistent rusher: Saquon Barkley or Derrick Henry?"),
    ("nfl_efficiency_burrow_herbert_comp","NFL","efficiency","Joe Burrow","Justin Herbert","completed passes","completion percentage","Who completed passes more efficiently: Joe Burrow or Justin Herbert?"),
    ("nfl_frequency_lamb_stbrown_100","NFL","frequency","CeeDee Lamb","Amon-Ra St. Brown","100-yard receiving games","percent of games","Who recorded 100 receiving yards in a greater percentage of games: CeeDee Lamb or Amon-Ra St. Brown?"),
    ("nfl_consistency_bosa_garrett_sacks","NFL","consistency","Nick Bosa","Myles Garrett","sacks","sacks","Who has been more consistent at recording sacks: Nick Bosa or Myles Garrett?"),
    ("nfl_efficiency_tucker_aubrey_fg","NFL","efficiency","Justin Tucker","Brandon Aubrey","field goals","field-goal percentage","Who converted field-goal attempts more efficiently: Justin Tucker or Brandon Aubrey?"),
    ("mlb_frequency_judge_ohtani_hr","MLB","frequency","Aaron Judge","Shohei Ohtani","games with a home run","percent of games","Who homered in a greater percentage of games: Aaron Judge or Shohei Ohtani?"),
    ("mlb_efficiency_soto_freeman_onbase","MLB","efficiency","Juan Soto","Freddie Freeman","times reached base","on-base rate","Who reached base in a greater percentage of plate appearances: Juan Soto or Freddie Freeman?"),
    ("mlb_typical_skubal_skenes_so","MLB","typical","Tarik Skubal","Paul Skenes","strikeouts","strikeouts","Who has produced more strikeouts in a typical start: Tarik Skubal or Paul Skenes?"),
    ("mlb_efficiency_lindor_witt_sb","MLB","efficiency","Francisco Lindor","Bobby Witt Jr.","stolen bases","stolen-base success percentage","Who has converted stolen-base attempts more efficiently: Francisco Lindor or Bobby Witt Jr.?"),
    ("mlb_typical_alonso_olson_hr","MLB","typical","Pete Alonso","Matt Olson","home runs","home runs","Who has hit more home runs in a typical stretch of games: Pete Alonso or Matt Olson?"),
    ("mlb_consistency_cole_wheeler_ip","MLB","consistency","Gerrit Cole","Zack Wheeler","innings pitched","innings","Who has been more consistent in innings pitched per start: Gerrit Cole or Zack Wheeler?"),
    ("mlb_typical_betts_arraez_hits","MLB","typical","Mookie Betts","Luis Arraez","hits","hits","Who has produced more hits in a typical game: Mookie Betts or Luis Arraez?"),
    ("mlb_frequency_acuna_carroll_steal","MLB","frequency","Ronald Acuña Jr.","Corbin Carroll","games with a stolen base","percent of games","Who stole a base in a greater percentage of games: Ronald Acuña Jr. or Corbin Carroll?"),
    ("nhl_typical_mcdavid_kucherov_ast","NHL","typical","Connor McDavid","Nikita Kucherov","assists","assists","Who has produced more assists in a typical game: Connor McDavid or Nikita Kucherov?"),
    ("nhl_consistency_matthews_pastrnak_shots","NHL","consistency","Auston Matthews","David Pastrňák","shots","shots","Who has been more consistent in shots on goal: Auston Matthews or David Pastrňák?"),
    ("nhl_efficiency_hellebuyck_saros_sv","NHL","efficiency","Connor Hellebuyck","Juuse Saros","saves","save percentage","Who stopped shots more efficiently: Connor Hellebuyck or Juuse Saros?"),
    ("nhl_typical_bedard_celebrini_pts","NHL","typical","Connor Bedard","Macklin Celebrini","points","points","Who has produced more points in a typical game: Connor Bedard or Macklin Celebrini?"),
    ("nhl_consistency_makar_fox_blocks","NHL","consistency","Cale Makar","Adam Fox","blocked shots","blocks","Who has been more consistent at blocking shots: Cale Makar or Adam Fox?"),
    ("nhl_frequency_ovechkin_crosby_goal","NHL","frequency","Alex Ovechkin","Sidney Crosby","games with a goal","percent of games","Who scored in a greater percentage of games: Alex Ovechkin or Sidney Crosby?"),
    ("nhl_typical_draisaitl_mackinnon_pp","NHL","typical","Leon Draisaitl","Nathan MacKinnon","power-play points","points","Who has produced more power-play points in a typical game: Leon Draisaitl or Nathan MacKinnon?"),
    ("soccer_typical_haaland_kane_goals","Soccer","typical","Erling Haaland","Harry Kane","goals","goals","Who has scored more goals in a typical match: Erling Haaland or Harry Kane?"),
    ("soccer_efficiency_mbappe_vinicius_sot","Soccer","efficiency","Kylian Mbappé","Vinícius Júnior","shots on target","shot-on-target percentage","Who put shots on target more efficiently: Kylian Mbappé or Vinícius Júnior?"),
    ("soccer_typical_salah_saka_ast","Soccer","typical","Mohamed Salah","Bukayo Saka","assists","assists","Who has produced more assists in a typical match: Mohamed Salah or Bukayo Saka?"),
    ("soccer_frequency_messi_bouanga_goal","Soccer","frequency","Lionel Messi","Denis Bouanga","matches with a goal","percent of matches","Who scored in a greater percentage of MLS matches: Lionel Messi or Denis Bouanga?"),
    ("soccer_efficiency_rodri_rice_pass","Soccer","efficiency","Rodri","Declan Rice","completed passes","pass-completion percentage","Who completed passes more efficiently: Rodri or Declan Rice?"),
    ("soccer_consistency_yamal_bellingham_shots","Soccer","consistency","Lamine Yamal","Jude Bellingham","shots","shots","Who has been more consistent in generating shots: Lamine Yamal or Jude Bellingham?"),
    ("soccer_frequency_lewandowski_lautaro","Soccer","frequency","Robert Lewandowski","Lautaro Martínez","matches with a goal","percent of matches","Who scored in a greater percentage of league matches: Robert Lewandowski or Lautaro Martínez?"),
    ("f1_typical_low_norris_piastri_qual","Formula 1","typical_low","Lando Norris","Oscar Piastri","qualifying position","place","Who has posted the stronger typical qualifying position: Lando Norris or Oscar Piastri?"),
    ("f1_consistency_russell_leclerc_finish","Formula 1","consistency","George Russell","Charles Leclerc","finishing position","place","Who has been the more consistent Grand Prix finisher: George Russell or Charles Leclerc?"),
    ("f1_frequency_verstappen_hamilton_podium","Formula 1","frequency","Max Verstappen","Lewis Hamilton","podium finishes","percent of races","Who reached the podium in a greater percentage of races: Max Verstappen or Lewis Hamilton?"),
    ("f1_typical_low_alonso_sainz_finish","Formula 1","typical_low","Fernando Alonso","Carlos Sainz","finishing position","place","Who has posted the stronger typical Grand Prix finish: Fernando Alonso or Carlos Sainz?"),
    ("f1_consistency_hamilton_russell_qual","Formula 1","consistency","Lewis Hamilton","George Russell","qualifying position","place","Who has been the more consistent qualifier: Lewis Hamilton or George Russell?"),
    ("pga_typical_low_scheffler_mcilroy_round","PGA","typical_low","Scottie Scheffler","Rory McIlroy","round score","strokes","Who has posted the stronger typical round score: Scottie Scheffler or Rory McIlroy?"),
    ("pga_efficiency_morikawa_schauffele_gir","PGA","efficiency","Collin Morikawa","Xander Schauffele","greens in regulation","greens-in-regulation percentage","Who hit greens in regulation more efficiently: Collin Morikawa or Xander Schauffele?"),
    ("pga_frequency_thomas_hovland_birdie","PGA","frequency","Justin Thomas","Viktor Hovland","rounds with at least four birdies","percent of rounds","Who recorded at least four birdies in a greater percentage of rounds: Justin Thomas or Viktor Hovland?"),
    ("pga_efficiency_spieth_henley_fairway","PGA","efficiency","Jordan Spieth","Russell Henley","fairways hit","driving-accuracy percentage","Who hit fairways more efficiently: Jordan Spieth or Russell Henley?"),
    ("pga_consistency_cantlay_finau_round","PGA","consistency","Patrick Cantlay","Tony Finau","round score","strokes","Who has been the more consistent scorer: Patrick Cantlay or Tony Finau?"),
    ("tennis_typical_sinner_alcaraz_aces","Tennis","typical","Jannik Sinner","Carlos Alcaraz","aces","aces","Who has served more aces in a typical match: Jannik Sinner or Carlos Alcaraz?"),
    ("tennis_efficiency_gauff_sabalenka_first","Tennis","efficiency","Coco Gauff","Aryna Sabalenka","first serves in","first-serve percentage","Who put first serves in play more efficiently: Coco Gauff or Aryna Sabalenka?"),
    ("tennis_consistency_swiatek_rybakina_games","Tennis","consistency","Iga Świątek","Elena Rybakina","games lost per match","games","Who has been more consistent at limiting games lost: Iga Świątek or Elena Rybakina?"),
    ("tennis_frequency_djokovic_medvedev_tiebreak","Tennis","frequency","Novak Djokovic","Daniil Medvedev","tiebreaks won","percent of tiebreaks","Who won a greater percentage of tiebreaks: Novak Djokovic or Daniil Medvedev?"),
]

for claim_id,sport,kind,player_a,player_b,metric,unit,question in ADDITIONAL_CLAIM_SPECS:
    if kind in ("typical", "typical_low", "consistency"):
        research = [f"Game-by-game or event-by-event {metric} for both athletes", "Use the same 8–10 games, matches, starts, rounds, or races", "Record dates and identify unusual or incomplete performances"]
    elif kind == "efficiency":
        research = [f"Successful {metric} for both athletes", "Total attempts or opportunities for both athletes", "Use the same season, competition, and time period"]
    else:
        research = [f"Number of qualifying {metric} for both athletes", "Total games, matches, rounds, or races", "Use the same season, competition, and time period"]
    why = {
        "typical":"Mean and median compare a typical performance while helping students consider outliers.",
        "typical_low":"Mean and median compare a typical performance; for this statistic, a lower value is stronger.",
        "consistency":"Game-by-game data and MAD compare how much the performances vary.",
        "efficiency":"A percentage compares successful outcomes with total opportunities.",
        "frequency":"Relative frequency compares how often the event occurred, even with different sample totals.",
    }[kind]
    source_name,source_url = SPORT_SOURCE_DEFAULTS[sport]
    DEBATES.append({"id":claim_id,"sport":sport,"kind":kind,"a":player_a,"b":player_b,"metric":metric,"unit":unit,"question":question,"research":research,"why":why,"source_name":source_name,"source_url":source_url})


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


STAT_ABBREVIATIONS = {
    "NBA": {"points":"PTS = points", "assists":"AST = assists", "blocks":"BLK = blocks", "field goals":"FGM = field goals made; FGA = field goals attempted; FG% = field-goal percentage", "three-pointers":"3PM = three-pointers made; 3PA = three-pointers attempted"},
    "WNBA": {"points":"PTS = points", "assists":"AST = assists", "field goals":"FGM = field goals made; FGA = field goals attempted; FG% = field-goal percentage"},
    "NFL": {"passing yards":"PASS YDS or YDS = passing yards", "receiving yards":"REC YDS = receiving yards", "receptions":"REC = receptions", "rushing yards":"RUSH YDS = rushing yards", "field goals":"FGM = field goals made; FGA = field goals attempted; FG% = field-goal percentage", "touchdowns":"TD = touchdowns; PASS TD = passing touchdowns; RUSH TD = rushing touchdowns", "games":"G or GP = games played"},
    "MLB": {"hits":"H = hits; G = games played", "total bases":"TB = total bases", "strikeouts":"SO or K = strikeouts", "walk":"BB = walks; PA = plate appearances; BB% = walk percentage", "games":"G or GP = games played", "at-bats":"AB = at-bats"},
    "NHL": {"saves":"SV = saves; SA or S = shots against; SV% = save percentage; GP = games played", "points":"P or PTS = points; G = goals; A = assists; GP = games played", "goals":"G = goals; A = assists; P or PTS = points; GP = games played"},
    "Soccer": {"goals":"G or GLS = goals; A or AST = assists; APP = appearances; MIN = minutes", "shots":"SH or S = shots; SOT = shots on target; G = goals", "matches":"MP, APP, or GP = matches/appearances"},
    "Formula 1": {"finishing position":"POS = finishing position; DNF = did not finish; DNS = did not start; DSQ = disqualified"},
    "PGA": {"round score":"R1–R4 = round scores; TOT = tournament total; CUT = missed cut", "top-10":"T10 = top-10 finish; EVENTS or STARTS = tournaments played"},
    "Tennis": {"service games":"SGW% = service games won percentage; DF = double faults; ACE = aces", "double faults":"DF = double faults; 1ST% = first serves in; 1ST WON% = first-serve points won"},
}


def stat_decoder(debate):
    sport_hints = STAT_ABBREVIATIONS.get(debate["sport"], {})
    metric = debate["metric"].lower()
    matches = [explanation for keyword, explanation in sport_hints.items() if keyword in metric]
    if not matches:
        matches = list(sport_hints.values())[:2]
    return matches


def math_directions(kind, lower_is_better=False):
    if kind == "consistency":
        return [
            "Find each player's mean: add all values, then divide by the number of values.",
            "For every value, find its distance from the mean. Use absolute value so every distance is positive.",
            "Add the absolute deviations and divide by the number of values. This is the MAD.",
            "Compare the MADs. The smaller MAD represents the more consistent set of results.",
            "Also compare the means: a player can be consistent without having the stronger typical performance.",
        ]
    if kind in ("typical", "typical_low"):
        direction = "lower" if lower_is_better else "higher"
        return [
            "Find each mean: add all values, then divide by the number of values.",
            "Find each median: order the values and locate the middle value (or average the two middle values).",
            "Check for an extreme value that might pull the mean away from most results.",
            f"Choose the fairer measure of center, then remember that a {direction} value is stronger for this question.",
        ]
    if kind == "efficiency":
        return [
            "For Player A, divide successful outcomes by total opportunities.",
            "Multiply the decimal by 100 and label it with a percent sign.",
            "Repeat the same steps for Player B.",
            "Subtract the smaller percentage from the larger percentage. Describe the difference in percentage points.",
        ]
    return [
        "For Player A, divide qualifying games by total games played.",
        "Multiply the decimal by 100 to find the relative-frequency percentage.",
        "Repeat the same steps for Player B.",
        "Subtract the smaller percentage from the larger percentage. Describe the difference in percentage points.",
    ]


def render_mmm_calculator(key, special_values=None):
    with st.expander("🧮 Open Mean, Median & Mode Calculator"):
        st.caption("This is an optional scratchpad. Try the calculation yourself first, then use this tool to check your work.")
        if special_values:
            st.info("For this Formula 1 investigation, DNF is automatically entered as 22.")
        raw = st.text_area(
            "Calculator data",
            placeholder="Enter values separated by commas or new lines",
            key=f"mmm_raw_{key}", height=100
        )
        values,bad = parse_numeric_text(raw, special_values=special_values)
        if bad:
            st.warning("These entries were not recognized: " + ", ".join(bad[:10]))
        if st.button("Calculate Mean, Median & Mode", key=f"mmm_button_{key}", use_container_width=True):
            if not values:
                st.warning("Enter at least one value first.")
            else:
                ordered = sorted(values)
                total = sum(values)
                mean_value = statistics.mean(values)
                median_value = statistics.median(values)
                mode_value = calc_mode(values)
                st.write("**Ordered data:** " + ", ".join(fmt(x) for x in ordered))
                st.write(f"**Mean:** {fmt(total)} ÷ {len(values)} = **{fmt(mean_value)}**")
                st.write(f"**Median:** **{fmt(median_value)}**")
                st.write(f"**Mode:** **{mode_value}**")


def build_news_pdf(debate, author, class_period, time_period, final_side, headline,
                   initial_reason, evidence_summary, math_work, interpretation,
                   limitation, final_claim, defense, source_url, image_a=None, image_b=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=46, leftMargin=46, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("NewsTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=25, textColor=colors.HexColor("#0F2747"), alignment=TA_CENTER, spaceAfter=12)
    kicker = ParagraphStyle("Kicker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#2563EB"), alignment=TA_CENTER, spaceAfter=6)
    byline = ParagraphStyle("Byline", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=14)
    body = ParagraphStyle("NewsBody", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=10)
    subhead = ParagraphStyle("Subhead", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, textColor=colors.HexColor("#0F2747"), spaceBefore=8, spaceAfter=6)
    caption = ParagraphStyle("PhotoCaption", parent=styles["Normal"], fontSize=8.5, leading=10, textColor=colors.HexColor("#475569"), alignment=TA_CENTER)

    story = [Paragraph(f"SPORTS DATA STUDIO | {escape(debate['sport'].upper())}", kicker), Paragraph(escape(headline or f"The Numbers Make the Case for {final_side}"), title_style), Paragraph(escape(f"By {author or 'Student Reporter'} | Class {class_period or '—'} | {date.today().strftime('%B %d, %Y')}"), byline)]

    image_cells = []
    for uploaded, player in [(image_a, debate["a"]), (image_b, debate["b"])]:
        if uploaded is not None:
            try:
                uploaded.seek(0)
                img = Image(uploaded, width=2.15*72, height=1.55*72, kind="proportional")
                image_cells.append([img, Paragraph(f"<b>{escape(player)}</b>", caption)])
            except Exception:
                image_cells.append([Paragraph(f"<b>{escape(player)}</b>", styles["Heading2"]), Paragraph("Player image unavailable", caption)])
        else:
            initials = "".join(part[0] for part in player.replace("'", "").split()[:2]).upper()
            image_cells.append([Paragraph(f"<font size='28'><b>{escape(initials)}</b></font>", title_style), Paragraph(f"<b>{escape(player)}</b>", caption)])
    left_panel = Table([[image_cells[0][0]], [image_cells[0][1]]], colWidths=[2.5*72])
    right_panel = Table([[image_cells[1][0]], [image_cells[1][1]]], colWidths=[2.5*72])
    photos = Table([[left_panel, right_panel]], colWidths=[3.35*72,3.35*72])
    photos.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#EAF2FF")),("BOX",(0,0),(-1,-1),1,colors.HexColor("#93C5FD")),("INNERGRID",(0,0),(-1,-1),.5,colors.HexColor("#BFDBFE")),("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10)]))
    story += [photos, Spacer(1,14)]

    lead = f"After examining {debate['metric']} from {time_period or 'the selected time period'}, I believe {final_side} has the stronger statistical case. My investigation began with this question: {debate['question']}"
    story += [Paragraph(escape(lead), body), Paragraph("How I Investigated", subhead), Paragraph(escape(initial_reason or "I began by making a prediction and identifying the evidence needed to test it."), body), Paragraph("The Numerical Evidence", subhead), Paragraph(escape(evidence_summary), body), Paragraph("My Calculations", subhead), Paragraph(escape(math_work), body), Paragraph("What the Numbers Mean", subhead), Paragraph(escape(interpretation), body), Paragraph("The Other Side of the Argument", subhead), Paragraph(escape(limitation), body), Paragraph("My Final Verdict", subhead), Paragraph(escape(final_claim), body), Paragraph("Responding to the Opposition", subhead), Paragraph(escape(defense), body), Spacer(1,8), Paragraph(escape(f"Data source: {debate['source_name']} | {source_url}"), caption)]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


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
    decoder = stat_decoder(d)
    if decoder:
        st.markdown("#### 🔎 Stat-page decoder")
        st.caption("These are abbreviations students may see on the research page:")
        for hint in decoder:
            st.write("• " + hint)
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


def render_claim_debate_lab_v2(teacher_mode):
    st.markdown("## 🗣️ Claim & Debate Lab")
    st.write("Research real athlete data, complete your own calculations, and defend the conclusion like a sports reporter.")

    c1,c2 = st.columns([1,2])
    with c1:
        support = st.selectbox("Support level", ["Training Mode", "Coach Mode", "Independent Mode"], key="v2_support")
        sports = ["All Sports"] + sorted({d["sport"] for d in DEBATES})
        sport = st.selectbox("Sport", sports, key="v2_sport")
    pool = DEBATES if sport == "All Sports" else [d for d in DEBATES if d["sport"] == sport]
    valid_ids = [d["id"] for d in pool]
    if st.session_state.get("v2_selected_debate_id") not in valid_ids:
        st.session_state["v2_selected_debate_id"] = valid_ids[0]
    with c2:
        st.selectbox(
            "Choose a real-player debate", valid_ids, key="v2_selected_debate_id",
            format_func=lambda x: next(d["question"] for d in DEBATES if d["id"] == x)
        )
        def pick_v2():
            current = st.session_state.get("v2_selected_debate_id")
            choices = [x for x in valid_ids if x != current] or valid_ids
            st.session_state["v2_selected_debate_id"] = random.choice(choices)
        st.button("🎲 Give Me Another Debate", use_container_width=True, on_click=pick_v2, key="v2_random")

    d = next(item for item in DEBATES if item["id"] == st.session_state["v2_selected_debate_id"])
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
        ["Player A", "Player B", "Not enough information yet"], index=2,
        format_func=lambda x: claim_choice_label(x, d), horizontal=True,
        key=f"v2_initial_{d['id']}"
    )
    initial_reason = st.text_area(
        "Why is that your initial prediction?",
        placeholder="This is a prediction. Strong statisticians revise claims when better evidence appears.",
        key=f"v2_initial_reason_{d['id']}", height=80
    )

    st.markdown("### Step 2 — Build the research plan")
    if support == "Independent Mode":
        st.info("Decide which data directly measures the words in the debate question.")
        student_plan = st.text_area("What will you research, and why is it relevant?", key=f"v2_plan_{d['id']}")
        with st.expander("Check the app's suggested research plan"):
            for item in d["research"]:
                st.write("• " + item)
            st.write("**Why this works:** " + d["why"])
    else:
        for item in d["research"]:
            st.write("✅ " + item)
        st.write("**Why these numbers matter:** " + d["why"])
        student_plan = "; ".join(d["research"])

    st.markdown("#### 🔎 Stat-page decoder")
    st.caption("Look for these abbreviations or column labels on the research site:")
    for hint in stat_decoder(d):
        st.write("• " + hint)
    st.markdown(f"**Research source:** [{d['source_name']}]({d['source_url']})")
    period = st.text_input("Season or exact date range", placeholder="Example: 2025–26 regular season, games through March 1", key=f"v2_period_{d['id']}")
    st.caption("Both athletes must use the same period and the same type of games or events.")

    st.markdown("### Step 3 — Enter the real data")
    math_ready = False
    evidence_summary = ""
    math_work = ""
    fair_sample = True

    if d["kind"] in ("typical", "typical_low", "consistency"):
        a_col,b_col = st.columns(2)
        with a_col:
            raw_a = st.text_area(f"{d['a']} — game-by-game {d['metric']}", placeholder="Enter values separated by commas or new lines", key=f"v2_data_a_{d['id']}", height=145)
        with b_col:
            raw_b = st.text_area(f"{d['b']} — game-by-game {d['metric']}", placeholder="Enter values separated by commas or new lines", key=f"v2_data_b_{d['id']}", height=145)
        race_codes = {"DNF":22} if d["sport"] == "Formula 1" else None
        if race_codes:
            st.info("🏁 Formula 1 coding rule: DNF (in any capitalization) is recorded as 22, representing last place for this classroom comparison.")
        a_vals,a_bad = parse_numeric_text(raw_a, special_values=race_codes)
        b_vals,b_bad = parse_numeric_text(raw_b, special_values=race_codes)
        if a_bad or b_bad:
            st.warning("Some entries were not numbers and were ignored.")
        if a_vals and b_vals:
            fair_sample = len(a_vals) == len(b_vals)
            if not fair_sample:
                st.warning(f"Fair-comparison check: {d['a']} has {len(a_vals)} values and {d['b']} has {len(b_vals)}. Try to use matching sample sizes.")
            if min(len(a_vals),len(b_vals)) < 5:
                st.warning("Try to collect at least 5 results; 8–10 usually creates stronger evidence.")

            st.markdown("### Step 4 — You do the math")
            with st.expander("Show me the calculation process", expanded=support == "Training Mode"):
                for number, direction in enumerate(math_directions(d["kind"], d["kind"] == "typical_low"), 1):
                    st.write(f"**{number}.** {direction}")
                if d["kind"] == "consistency":
                    st.latex(r"MAD=\frac{\sum |\text{each value}-\text{mean}|}{\text{number of values}}")
                else:
                    st.latex(r"\text{Mean}=\frac{\text{sum of all values}}{\text{number of values}}")

            measures = ["Mean", "Median"] if d["kind"] != "consistency" else ["Mean", "Range", "MAD"]
            answers = {}
            a_calc,b_calc = st.columns(2)
            with a_calc:
                st.markdown(f"**{d['a']} — your calculated results**")
                for measure in measures:
                    answers[f"a_{measure}"] = st.text_input(measure, key=f"v2_calc_a_{measure}_{d['id']}", placeholder="Your answer")
            with b_calc:
                st.markdown(f"**{d['b']} — your calculated results**")
                for measure in measures:
                    answers[f"b_{measure}"] = st.text_input(measure, key=f"v2_calc_b_{measure}_{d['id']}", placeholder="Your answer")
            math_work = st.text_area("Show your calculation work", key=f"v2_work_{d['id']}", placeholder="Write the addition, division, ordered list, or absolute deviations you used.", height=125)
            math_ready = all(str(value).strip() for value in answers.values()) and bool(math_work.strip())
            evidence_summary = "; ".join([f"{d['a']} {m.lower()}: {answers[f'a_{m}']} {d['unit']}; {d['b']} {m.lower()}: {answers[f'b_{m}']} {d['unit']}" for m in measures])

    elif d["kind"] in ("efficiency", "frequency"):
        is_frequency = d["kind"] == "frequency"
        first_label = "Games meeting the condition" if is_frequency else "Successful outcomes"
        second_label = "Total games" if is_frequency else "Total attempts/opportunities"
        a_col,b_col = st.columns(2)
        with a_col:
            st.markdown(f"**{d['a']}**")
            a_first = parse_one_number(st.text_input(first_label, key=f"v2_a_first_{d['id']}"))
            a_total = parse_one_number(st.text_input(second_label, key=f"v2_a_total_{d['id']}"))
        with b_col:
            st.markdown(f"**{d['b']}**")
            b_first = parse_one_number(st.text_input(first_label, key=f"v2_b_first_{d['id']}"))
            b_total = parse_one_number(st.text_input(second_label, key=f"v2_b_total_{d['id']}"))
        if all(x is not None for x in [a_first,a_total,b_first,b_total]):
            if a_total <= 0 or b_total <= 0 or a_first > a_total or b_first > b_total:
                st.error("Check the data: totals must be positive and successful/qualifying outcomes cannot exceed the totals.")
            else:
                st.markdown("### Step 4 — You do the math")
                formula_name = "relative frequency" if is_frequency else "percentage"
                numerator = "qualifying games" if is_frequency else "successful outcomes"
                denominator = "total games" if is_frequency else "total opportunities"
                st.latex(rf"\text{{{formula_name}}}=\frac{{\text{{{numerator}}}}}{{\text{{{denominator}}}}}\times100")
                with st.expander("Show me the calculation process", expanded=support == "Training Mode"):
                    for number, direction in enumerate(math_directions(d["kind"]), 1):
                        st.write(f"**{number}.** {direction}")
                    if not is_frequency:
                        st.info("Example format only: 27 ÷ 30 = 0.90, then 0.90 × 100 = 90%. Use your researched numbers.")
                p1,p2 = st.columns(2)
                with p1:
                    a_pct = st.text_input(f"{d['a']} — your calculated percentage", key=f"v2_a_pct_{d['id']}", placeholder="Include %")
                with p2:
                    b_pct = st.text_input(f"{d['b']} — your calculated percentage", key=f"v2_b_pct_{d['id']}", placeholder="Include %")
                pp_diff = st.text_input("Difference in percentage points", key=f"v2_pp_{d['id']}", placeholder="Subtract the smaller percentage from the larger")
                math_work = st.text_area("Show your calculation work", key=f"v2_work_{d['id']}", placeholder=f"{numerator} ÷ {denominator} × 100", height=125)
                math_ready = all(str(x).strip() for x in [a_pct,b_pct,pp_diff,math_work])
                evidence_summary = f"{d['a']}: {a_first:g}/{a_total:g}, calculated rate {a_pct}; {d['b']}: {b_first:g}/{b_total:g}, calculated rate {b_pct}; difference: {pp_diff} percentage points."

    render_mmm_calculator(d["id"], special_values={"DNF":22} if d["sport"] == "Formula 1" else None)

    if math_ready:
        st.markdown("### Step 5 — Decide what the evidence means")
        final_side = st.radio(
            "Which conclusion is best supported by your calculations?",
            [d["a"], d["b"], "The results are tied", "There is not enough evidence"],
            horizontal=True, key=f"v2_winner_{d['id']}"
        )
        interpretation = st.text_area(
            "Explain what your calculated numbers mean in words.",
            placeholder="Explain why the larger, smaller, or more consistent result matters for this exact claim.",
            key=f"v2_interpret_{d['id']}", height=110
        )
        evidence_rating = st.radio(
            "How does this evidence affect your original claim?",
            ["Supports it", "Partially supports it", "Contradicts it", "Not enough evidence"],
            horizontal=True, key=f"v2_rating_{d['id']}"
        )
        limitation = st.text_area(
            "Name one limitation or piece of counterevidence.",
            placeholder="Examples: small sample, different opponents, minutes played, outlier, course difficulty…",
            key=f"v2_limitation_{d['id']}", height=85
        )

        st.markdown("### Step 6 — Build and defend the final claim")
        final_claim = st.text_area(
            "Final argument",
            placeholder="I claim ___ because the data shows ___. This matters because ___. One limitation is ___; however, ___.",
            key=f"v2_final_{d['id']}", height=145
        )
        challenge_map = {
            "consistency":f"You used MAD to discuss consistency. Does {final_side} also have a strong mean, or could that athlete be consistently lower?",
            "typical":"Would your conclusion change if you used the other measure of center or removed an outlier?",
            "typical_low":"Could conditions, opponents, or course difficulty explain part of the difference?",
            "efficiency":"Does the higher percentage come from enough attempts to be convincing? Defend the sample size.",
            "frequency":"Is the chosen condition the fairest definition of success, or would another threshold change the conclusion?",
        }
        st.markdown("#### 🎤 Challenge My Argument")
        st.warning(challenge_map[d["kind"]])
        defense = st.text_area("Your response to the opposing analyst", key=f"v2_defense_{d['id']}", height=105)

        checklist = {
            "Initial prediction": bool(initial_reason.strip()), "Time period identified": bool(period.strip()),
            "Fair sample sizes": fair_sample, "Calculations shown": bool(math_work.strip()),
            "Numbers interpreted": bool(interpretation.strip()), "Counterevidence considered": bool(limitation.strip()),
            "Final argument written": bool(final_claim.strip()), "Challenge answered": bool(defense.strip()),
        }
        completed = sum(checklist.values())
        st.progress(completed/len(checklist), text=f"Investigation checklist: {completed}/{len(checklist)} complete")

        if teacher_mode or completed == len(checklist):
            st.markdown("### Step 7 — Publish your sports news article")
            p1,p2 = st.columns(2)
            with p1:
                author = st.text_input("Reporter name", key=f"v2_author_{d['id']}")
                headline = st.text_input("Article headline", value=f"The Numbers Make the Case for {final_side}", key=f"v2_headline_{d['id']}")
                image_a = st.file_uploader(f"Optional photo of {d['a']}", type=["png","jpg","jpeg"], key=f"v2_image_a_{d['id']}")
            with p2:
                class_period = st.text_input("Class period", key=f"v2_class_{d['id']}")
                st.caption("Use photos you have permission to use. Without uploads, the PDF uses player-initial cards.")
                image_b = st.file_uploader(f"Optional photo of {d['b']}", type=["png","jpg","jpeg"], key=f"v2_image_b_{d['id']}")
            if author.strip():
                pdf_bytes = build_news_pdf(
                    d, author, class_period, period, final_side, headline,
                    initial_reason, evidence_summary, math_work, interpretation,
                    limitation, final_claim, defense, d["source_url"], image_a, image_b
                )
                st.download_button("📰 Download My Sports News Article (PDF)", pdf_bytes, file_name=f"sports_news_{d['id']}.pdf", mime="application/pdf", use_container_width=True)
            else:
                st.info("Enter the reporter name to create the PDF article.")


def standardize_categories(values, ignore_case=True):
    cleaned = [" ".join(str(value).strip().split()) for value in values if str(value).strip()]
    if not ignore_case:
        return cleaned
    display = {}
    standardized = []
    for value in cleaned:
        key = value.casefold()
        if key not in display:
            display[key] = value
        standardized.append(display[key])
    return standardized


def add_frequency_totals(table):
    result = table.copy().astype(int)
    result["Total"] = result.sum(axis=1)
    result.loc["Total"] = result.sum(axis=0)
    return result


def render_categorical_frequency_lab(teacher_mode):
    st.markdown("## 🧩 Categorical Frequency Table Lab")
    st.write("Separate responses into categories, organize the counts, and use a one-way or two-way frequency table to defend a claim.")
    table_type = st.radio(
        "Choose the table type",
        ["One-Way Frequency Table", "Two-Way Frequency Table"],
        horizontal=True, key="frequency_table_type"
    )

    if table_type == "One-Way Frequency Table":
        st.markdown("### Step 1 — Identify one categorical variable")
        variable = st.text_input("Variable name", placeholder="Example: Favorite sport", key="oneway_variable")
        entry_type = st.radio(
            "How will you enter the categories?",
            ["Paste individual responses", "Enter category counts"],
            horizontal=True, key="oneway_entry"
        )
        ignore_case = st.checkbox("Treat capitalization differences as the same category", value=True, key="oneway_case")
        categories,counts = [],[]

        if entry_type == "Paste individual responses":
            raw = st.text_area(
                "Paste the individual responses",
                height=170,
                placeholder="Basketball\nSoccer\nBasketball\nFootball\nSoccer",
                key="oneway_raw"
            )
            responses = standardize_categories(parse_categorical_text(raw), ignore_case)
            if responses:
                counter = Counter(responses)
                categories = list(counter.keys()); counts = list(counter.values())
                st.markdown("#### Categories the app separated")
                st.dataframe(pd.DataFrame({"Category":categories,"Tally/Count":counts}), use_container_width=True, hide_index=True)
                st.caption("Check these categories before continuing. Misspellings such as ‘Socer’ and ‘Soccer’ will remain separate so students can identify and correct them.")
        else:
            category_total = st.slider("Number of categories",2,10,4,key="oneway_category_total")
            for i in range(category_total):
                c1,c2 = st.columns([2,1])
                with c1:
                    category = st.text_input(f"Category {i+1}", key=f"oneway_category_{i}")
                with c2:
                    count = st.number_input(f"Frequency {i+1}",min_value=0,step=1,key=f"oneway_count_{i}")
                if category.strip():
                    categories.append(category.strip()); counts.append(int(count))

        preds,reveal = prediction_section("cat_frequency_one",[
            "Which category do you predict will occur most frequently?",
            "Do you predict one category will contain more than half of the responses? Explain."
        ],teacher_mode)
        if not reveal:
            st.info("Answer the prediction questions and lock them in to reveal the completed frequency table.")
        if categories and sum(counts)>0 and reveal:
            total = sum(counts)
            table = pd.DataFrame({
                "Category":categories,
                "Frequency":counts,
                "Relative Frequency":[round(c/total,3) for c in counts],
                "Percent":[f"{c/total*100:.1f}%" for c in counts],
            })
            table.loc[len(table)] = ["Total",total,1.0,"100.0%"]
            st.markdown("### Step 2 — One-way frequency table")
            st.dataframe(table,use_container_width=True,hide_index=True)
            st.caption("Relative frequency = category frequency ÷ total number of responses")
            if st.checkbox("Show a frequency bar chart", key="oneway_bar"):
                st.pyplot(make_bar(categories,counts,variable or "One-Way Frequency Table"))
            st.markdown("### Step 3 — Interpret the categories")
            st.text_area("Write a claim supported by at least one frequency or percentage.", key="oneway_claim", height=100)
            st.text_area("Explain why that number supports your claim.", key="oneway_reasoning", height=90)

    else:
        st.markdown("### Step 1 — Identify two categorical variables")
        n1,n2 = st.columns(2)
        with n1:
            row_name = st.text_input("Row variable", value="Group", key="twoway_row_name")
        with n2:
            column_name = st.text_input("Column variable", value="Outcome", key="twoway_column_name")
        entry_type = st.radio(
            "How will you enter the paired categories?",
            ["Paste paired individual responses", "Enter the table counts"],
            horizontal=True, key="twoway_entry"
        )
        ignore_case = st.checkbox("Treat capitalization differences as the same category", value=True, key="twoway_case")
        count_table = None

        if entry_type == "Paste paired individual responses":
            raw = st.text_area(
                "Paste one pair per line",
                height=190,
                placeholder="Home, Win\nAway, Loss\nHome, Win\nAway, Win\nHome, Loss",
                key="twoway_raw"
            )
            pairs,bad_lines = [],[]
            for line_number,line in enumerate(str(raw or "").splitlines(),1):
                if not line.strip():
                    continue
                parts = line.split("\t") if "\t" in line else line.split(",")
                parts = [part.strip() for part in parts]
                if len(parts) != 2 or not all(parts):
                    bad_lines.append(line_number)
                else:
                    pairs.append(parts)
            if bad_lines:
                st.warning("Each line needs exactly two categories. Check line(s): " + ", ".join(map(str,bad_lines[:12])))
            if pairs:
                row_values = standardize_categories([pair[0] for pair in pairs],ignore_case)
                column_values = standardize_categories([pair[1] for pair in pairs],ignore_case)
                paired = pd.DataFrame({row_name or "Row Variable":row_values,column_name or "Column Variable":column_values})
                st.markdown("#### Categories the app separated")
                s1,s2 = st.columns(2)
                s1.write(f"**{row_name}:** " + ", ".join(dict.fromkeys(row_values)))
                s2.write(f"**{column_name}:** " + ", ".join(dict.fromkeys(column_values)))
                with st.expander("Check the separated pairs"):
                    st.dataframe(paired,use_container_width=True,hide_index=True)
                count_table = pd.crosstab(paired.iloc[:,0],paired.iloc[:,1])
        else:
            d1,d2 = st.columns(2)
            with d1:
                row_total = st.slider("Number of row categories",2,6,2,key="twoway_rows")
            with d2:
                column_total = st.slider("Number of column categories",2,6,2,key="twoway_columns")
            row_labels = [st.text_input(f"{row_name} category {i+1}",value=f"Row {i+1}",key=f"twoway_row_{i}") for i in range(row_total)]
            column_labels = [st.text_input(f"{column_name} category {j+1}",value=f"Column {j+1}",key=f"twoway_col_{j}") for j in range(column_total)]
            if len(set(row_labels)) < len(row_labels) or len(set(column_labels)) < len(column_labels):
                st.warning("Every row and column category needs a different name.")
            else:
                rows=[]
                st.markdown("#### Enter each joint frequency")
                for i,row_label in enumerate(row_labels):
                    st.write(f"**{row_label}**")
                    cells = st.columns(column_total)
                    row=[]
                    for j,column_label in enumerate(column_labels):
                        with cells[j]:
                            row.append(int(st.number_input(column_label,min_value=0,step=1,key=f"twoway_cell_{i}_{j}")))
                    rows.append(row)
                count_table = pd.DataFrame(rows,index=row_labels,columns=column_labels)

        preds,reveal = prediction_section("cat_frequency_two",[
            "Which combination of categories do you predict will have the greatest joint frequency?",
            "Do you predict the two variables will show a noticeable pattern or relationship? Why?"
        ],teacher_mode)
        if not reveal:
            st.info("Answer the prediction questions and lock them in to reveal the two-way table.")
        if count_table is not None and int(count_table.to_numpy().sum())>0 and reveal:
            st.markdown("### Step 2 — Two-way frequency table")
            display_counts = add_frequency_totals(count_table)
            st.dataframe(display_counts,use_container_width=True)
            st.caption("Inside cells are joint frequencies. The final row and column are marginal totals.")

            percent_view = st.radio(
                "Optional percentage view",
                ["Counts only", "Row percentages", "Column percentages", "Overall percentages"],
                horizontal=True, key="twoway_percent_view"
            )
            if percent_view == "Row percentages":
                pct = count_table.div(count_table.sum(axis=1).replace(0,np.nan),axis=0)*100
                st.dataframe(pct.round(1).astype(str)+"%",use_container_width=True)
                st.caption("Each row is its own 100%. Use this to compare outcomes within each row category.")
            elif percent_view == "Column percentages":
                pct = count_table.div(count_table.sum(axis=0).replace(0,np.nan),axis=1)*100
                st.dataframe(pct.round(1).astype(str)+"%",use_container_width=True)
                st.caption("Each column is its own 100%. Use this to compare row groups within each column category.")
            elif percent_view == "Overall percentages":
                pct = count_table/count_table.to_numpy().sum()*100
                st.dataframe(pct.round(1).astype(str)+"%",use_container_width=True)
                st.caption("Every cell is compared with the grand total, so the entire table represents 100%.")

            st.markdown("### Step 3 — Interpret the relationship")
            st.text_area("What is the greatest joint frequency, and which two categories create it?", key="twoway_joint", height=85)
            st.text_area("Compare at least two rows or columns using counts or percentages.", key="twoway_compare", height=95)
            st.text_area("Write a claim about whether the variables appear related. Defend it with evidence from the table.", key="twoway_claim", height=115)


def render_categorical_frequency_lab_v2(teacher_mode):
    st.markdown("## 🧩 Categorical Frequency Table Lab")
    st.write("Type category names into the table headers, then enter the frequencies directly into the cells.")
    table_type = st.radio(
        "Choose the table type",
        ["One-Way Frequency Table", "Two-Way Frequency Table"],
        horizontal=True, key="frequency_table_type_v2"
    )

    if table_type == "One-Way Frequency Table":
        st.markdown("### Step 1 — Fill in the one-way table")
        variable = st.text_input("Categorical variable name", placeholder="Example: Favorite sport", key="oneway_variable_v2")
        category_total = st.slider("Number of category rows",2,10,4,key="oneway_rows_v2")

        header = st.columns([2,1,1])
        header[0].markdown("**Category**")
        header[1].markdown("**Frequency**")
        header[2].markdown("**Percent (auto)**")
        categories,counts = [],[]
        row_entries=[]
        current_total = sum(int(st.session_state.get(f"oneway_grid_count_{i}",0) or 0) for i in range(category_total))
        for i in range(category_total):
            cells = st.columns([2,1,1])
            with cells[0]:
                category = st.text_input(
                    f"Category row {i+1}", placeholder=f"Category {i+1}",
                    key=f"oneway_grid_category_{i}", label_visibility="collapsed"
                )
            with cells[1]:
                count = int(st.number_input(
                    f"Frequency row {i+1}", min_value=0, step=1,
                    key=f"oneway_grid_count_{i}", label_visibility="collapsed"
                ))
            row_entries.append((category.strip(),count))
            categories = [name for name,_ in row_entries if name]
            counts = [value for name,value in row_entries if name]
            with cells[2]:
                st.markdown(f"**{(count/current_total*100):.1f}%**" if current_total else "—")

        total = sum(value for _,value in row_entries)
        total_row = st.columns([2,1,1])
        total_row[0].markdown("**TOTAL**")
        total_row[1].markdown(f"**{total}**")
        total_row[2].markdown("**100%**" if total else "—")

        with st.expander("Need help separating a list of individual responses?"):
            raw = st.text_area(
                "Paste individual responses here",
                placeholder="Basketball\nSoccer\nBasketball\nFootball",
                key="oneway_separator_v2", height=120
            )
            responses = standardize_categories(parse_categorical_text(raw),True)
            if responses:
                suggested = Counter(responses)
                st.dataframe(
                    pd.DataFrame({"Category to copy into the grid":list(suggested.keys()),"Frequency to copy":list(suggested.values())}),
                    use_container_width=True, hide_index=True
                )

        duplicate_categories = len(categories) != len(set(name.casefold() for name in categories))
        if duplicate_categories:
            st.warning("Two category rows have the same name. Combine them or rename one before continuing.")
        preds,reveal = prediction_section("cat_grid_one",[
            "Which category do you predict will have the greatest frequency?",
            "Do you predict any category will contain more than half of the data?"
        ],teacher_mode)
        if categories and total>0 and not duplicate_categories and reveal:
            final_counts = [value for name,value in row_entries if name]
            table = pd.DataFrame({
                variable or "Category":categories,
                "Frequency":final_counts,
                "Relative Frequency":[round(value/total,3) for value in final_counts],
                "Percent":[f"{value/total*100:.1f}%" for value in final_counts],
            })
            table.loc[len(table)] = ["Total",total,1.0,"100.0%"]
            st.markdown("### Step 2 — Completed one-way frequency table")
            st.dataframe(table,use_container_width=True,hide_index=True)
            st.markdown("### 🧠 Percentage Interpretation Coach")
            chosen_category = st.selectbox(
                "Choose one category to interpret",
                categories,key="oneway_interpret_category"
            )
            chosen_index = categories.index(chosen_category)
            chosen_count = final_counts[chosen_index]
            chosen_percent = chosen_count/total*100
            st.info(
                f"**{chosen_percent:.1f}%** means that **{chosen_count} out of {total} responses** were "
                f"**{chosen_category}**. If there were exactly 100 similar responses, about "
                f"**{chosen_percent:.0f} out of 100** would be {chosen_category}."
            )
            if chosen_percent > 50:
                st.write(f"**More than half:** {chosen_category} represents a majority of the responses because {chosen_percent:.1f}% is greater than 50%.")
            elif chosen_percent == 50:
                st.write(f"**Exactly half:** {chosen_category} represents 50% of the responses.")
            else:
                st.write(f"**Less than half:** {chosen_category} is not a majority because {chosen_percent:.1f}% is below 50%.")
            st.text_area(
                "Finish the interpretation",
                placeholder=f"The percentage of responses in the {chosen_category} category suggests that...",
                key="oneway_interpret_finish",height=80
            )
            st.text_area("Write a claim supported by at least one frequency or percentage.", key="oneway_grid_claim", height=100)
            st.text_area("Explain why the evidence supports your claim.", key="oneway_grid_reason", height=90)

    else:
        st.markdown("### Step 1 — Name the variables and choose the table size")
        names = st.columns(2)
        with names[0]:
            row_name = st.text_input("Row variable name", placeholder="Example: Location", key="twoway_row_name_v2")
        with names[1]:
            column_name = st.text_input("Column variable name", placeholder="Example: Game result", key="twoway_column_name_v2")
        sizes = st.columns(2)
        with sizes[0]:
            row_total = st.slider("Number of row categories",2,6,2,key="twoway_rows_v2")
        with sizes[1]:
            column_total = st.slider("Number of column categories",2,6,2,key="twoway_columns_v2")

        st.markdown("### Step 2 — Fill in the two-way table")
        header_cells = st.columns([1.5]+[1]*column_total+[1])
        with header_cells[0]:
            st.markdown(f"**{row_name or 'Row variable'} ↓**  \\  **{column_name or 'Column variable'} →**")
        column_labels=[]
        for j in range(column_total):
            with header_cells[j+1]:
                column_labels.append(st.text_input(
                    f"Column category {j+1}", placeholder=f"Column {j+1}",
                    key=f"twoway_grid_col_{j}", label_visibility="collapsed"
                ).strip())
        header_cells[-1].markdown("**TOTAL**")

        row_labels=[]; values=[]
        for i in range(row_total):
            row_cells = st.columns([1.5]+[1]*column_total+[1])
            with row_cells[0]:
                row_label = st.text_input(
                    f"Row category {i+1}", placeholder=f"Row {i+1}",
                    key=f"twoway_grid_row_{i}", label_visibility="collapsed"
                ).strip()
            row_labels.append(row_label)
            row_values=[]
            for j in range(column_total):
                with row_cells[j+1]:
                    row_values.append(int(st.number_input(
                        f"{row_label or f'Row {i+1}'} by {column_labels[j] or f'Column {j+1}'}",
                        min_value=0,step=1,key=f"twoway_grid_cell_{i}_{j}",label_visibility="collapsed"
                    )))
            values.append(row_values)
            row_cells[-1].markdown(f"**{sum(row_values)}**")

        total_cells = st.columns([1.5]+[1]*column_total+[1])
        total_cells[0].markdown("**TOTAL**")
        for j in range(column_total):
            total_cells[j+1].markdown(f"**{sum(values[i][j] for i in range(row_total))}**")
        grand_total = sum(sum(row) for row in values)
        total_cells[-1].markdown(f"**{grand_total}**")

        missing_headers = not all(row_labels) or not all(column_labels)
        duplicate_headers = (
            len({label.casefold() for label in row_labels if label}) != len([label for label in row_labels if label]) or
            len({label.casefold() for label in column_labels if label}) != len([label for label in column_labels if label])
        )
        if missing_headers:
            st.info("Type a category name into every row and column header box.")
        elif duplicate_headers:
            st.warning("Every row category and every column category needs a different name.")

        preds,reveal = prediction_section("cat_grid_two",[
            "Which inner cell do you predict will have the greatest joint frequency?",
            "Do you predict the two categorical variables will show a pattern or relationship?"
        ],teacher_mode)
        if not missing_headers and not duplicate_headers and grand_total>0 and reveal:
            count_table = pd.DataFrame(values,index=row_labels,columns=column_labels)
            st.markdown("### Step 3 — Completed two-way frequency table")
            st.dataframe(add_frequency_totals(count_table),use_container_width=True)
            st.caption("Inner cells are joint frequencies. The TOTAL row and column contain marginal frequencies.")
            percent_view = st.radio(
                "Choose a comparison view",
                ["Counts", "Row percentages", "Column percentages", "Overall percentages"],
                horizontal=True,key="twoway_grid_percent_view"
            )
            if percent_view == "Row percentages":
                pct = count_table.div(count_table.sum(axis=1).replace(0,np.nan),axis=0)*100
                st.dataframe(pct.round(1).astype(str)+"%",use_container_width=True)
                st.caption("Each row represents 100%.")
            elif percent_view == "Column percentages":
                pct = count_table.div(count_table.sum(axis=0).replace(0,np.nan),axis=1)*100
                st.dataframe(pct.round(1).astype(str)+"%",use_container_width=True)
                st.caption("Each column represents 100%.")
            elif percent_view == "Overall percentages":
                pct = count_table/grand_total*100
                st.dataframe(pct.round(1).astype(str)+"%",use_container_width=True)
                st.caption("The entire table represents 100%.")

            st.markdown("### 🧠 Percentage Interpretation Coach")
            choose1,choose2 = st.columns(2)
            with choose1:
                chosen_row = st.selectbox(f"Choose a {row_name or 'row'} category",row_labels,key="twoway_interpret_row")
            with choose2:
                chosen_column = st.selectbox(f"Choose a {column_name or 'column'} category",column_labels,key="twoway_interpret_column")
            cell_count = int(count_table.loc[chosen_row,chosen_column])
            row_total_value = int(count_table.loc[chosen_row].sum())
            column_total_value = int(count_table[chosen_column].sum())
            if percent_view == "Row percentages":
                interpreted_percent = cell_count/row_total_value*100 if row_total_value else 0
                explanation = (
                    f"Among all **{chosen_row}** responses, **{cell_count} out of {row_total_value}** were "
                    f"**{chosen_column}**. That is **{interpreted_percent:.1f}%**. The denominator is the {chosen_row} row total."
                )
                sentence_starter = f"Among {chosen_row} responses, {interpreted_percent:.1f}% were {chosen_column}. This suggests that..."
            elif percent_view == "Column percentages":
                interpreted_percent = cell_count/column_total_value*100 if column_total_value else 0
                explanation = (
                    f"Among all **{chosen_column}** responses, **{cell_count} out of {column_total_value}** were "
                    f"**{chosen_row}**. That is **{interpreted_percent:.1f}%**. The denominator is the {chosen_column} column total."
                )
                sentence_starter = f"Among {chosen_column} responses, {interpreted_percent:.1f}% were {chosen_row}. This suggests that..."
            else:
                interpreted_percent = cell_count/grand_total*100
                explanation = (
                    f"Of all **{grand_total} responses** in the table, **{cell_count}** were both "
                    f"**{chosen_row}** and **{chosen_column}**. That is **{interpreted_percent:.1f}%** of the entire data set."
                )
                sentence_starter = f"Overall, {interpreted_percent:.1f}% of responses were both {chosen_row} and {chosen_column}. This suggests that..."
            st.info(explanation)
            st.caption("The app explains what the percentage literally means. You still decide what pattern it supports.")
            st.text_area(
                "Finish the interpretation",
                placeholder=sentence_starter,key="twoway_interpret_finish",height=85
            )
            st.markdown("### Step 4 — Make and defend a categorical claim")
            st.text_area("Identify the greatest joint frequency and its row/column categories.", key="twoway_grid_joint", height=85)
            st.text_area("Compare at least two rows or columns using counts or percentages.", key="twoway_grid_compare", height=95)
            st.text_area("Write a claim about the relationship and defend it with table evidence.", key="twoway_grid_claim", height=110)

st.markdown("""
<div class="studio-card">
  <div class="studio-step">Sports by the Numbers</div>
  <h1 style="margin:.2rem 0 .35rem;">📊 Sports Data Studio</h1>
  <p style="margin:0;">Enter it. Graph it. Analyze it. Defend it.</p>
  <span class="build-badge">Light UI build 2026.09.21</span>
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
if mode not in ("Claim & Debate Lab", "Categorical Data"):
    dataset_name=st.text_input("Dataset / activity name",placeholder="Example: 40-Yard Dash")
    unit=st.text_input("Unit",placeholder="Example: seconds, points, inches")

if mode=="Claim & Debate Lab":
    render_claim_debate_lab_v2(teacher_mode)

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

elif mode=="Categorical Data":
    render_categorical_frequency_lab_v2(teacher_mode)

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

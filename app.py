import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import plotly.graph_objects as go
import json
import os
import numpy as np
import base64
import re
import uuid

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="Smaartbrand Intelligence", page_icon="🏨", layout="wide")

# ─────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    text-align: center; padding: 24px;
    background: linear-gradient(135deg, #1e3a5f 0%, #0d9488 100%);
    border-radius: 12px; margin-bottom: 20px;
}
.main-header h1 { color: white; margin: 0; font-size: 1.9rem; }
.main-header p { color: rgba(255,255,255,0.85); margin: 8px 0 0 0; font-size: 0.9rem; }

.section-header {
    font-size: 1.05rem; font-weight: 600; color: #1a1a2e;
    margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid #e5e7eb;
}

.hotel-name-cell {
    font-size: 0.7rem; font-weight: 600; color: #1e293b;
    word-wrap: break-word; max-width: 100px; line-height: 1.25; text-align: center;
}

.phrase-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 10px 12px; margin-bottom: 6px;
}
.phrase-card .phrase-text { font-size: 0.8rem; font-weight: 500; color: #1e293b; }
.phrase-card .phrase-count { font-size: 0.7rem; color: #64748b; float: right; margin-top: -16px; }

.dot-strong { width: 10px; height: 10px; border-radius: 50%; background: #1e3a5f; display: inline-block; }
.dot-weak { width: 10px; height: 10px; border-radius: 50%; background: #93c5fd; display: inline-block; }
.dot-none { width: 10px; height: 10px; border-radius: 50%; background: #e5e7eb; display: inline-block; }

.score-cell { padding: 6px 8px; border-radius: 5px; text-align: center; font-weight: 600; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CREDENTIALS
# ─────────────────────────────────────────
def get_credentials():
    gcp_creds = os.environ.get("GCP_CREDENTIALS_JSON", "")
    if not gcp_creds:
        return None
    gcp_creds = gcp_creds.strip().strip('"').strip("'")
    try:
        if gcp_creds.startswith("{"):
            creds_dict = json.loads(gcp_creds)
        else:
            padding = 4 - len(gcp_creds) % 4
            if padding != 4:
                gcp_creds += "=" * padding
            creds_dict = json.loads(base64.b64decode(gcp_creds).decode('utf-8'))
        return service_account.Credentials.from_service_account_info(creds_dict)
    except Exception as e:
        st.error(f"Credential error: {e}")
        return None

@st.cache_resource
def get_bq_client():
    credentials = get_credentials()
    if credentials:
        return bigquery.Client(credentials=credentials, project=credentials.project_id)
    try:
        return bigquery.Client(project="gen-lang-client-0143536012")
    except:
        return None

client = get_bq_client()

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
PROJECT = "gen-lang-client-0143536012"
DATASET = "analyst"
AGENT_ID = "agent_b9a402f4-9a19-40c7-849e-e1df4f3ad0b2"
LOCATION = "global"

ASPECT_MAP = {1: "Dining", 2: "Cleanliness", 3: "Amenities", 4: "Staff",
              5: "Room", 6: "Location", 7: "Value for Money", 8: "General"}
ASPECT_ICONS = {"Dining": "🍽️", "Cleanliness": "🧹", "Amenities": "🏊", "Staff": "👨‍💼",
                "Room": "🛏️", "Location": "📍", "Value for Money": "💰", "General": "⭐"}
TEAL_SCALE = ["#f0fdfa", "#99f6e4", "#5eead4", "#2dd4bf", "#14b8a6", "#0d9488", "#0f766e", "#115e59"]

def get_color_for_score(score):
    idx = min(int(score / 100 * (len(TEAL_SCALE) - 1)), len(TEAL_SCALE) - 1)
    return TEAL_SCALE[idx]

# ─────────────────────────────────────────
# DATA QUERIES
# ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_metadata():
    if client is None:
        return pd.DataFrame()
    query = f"""
    SELECT DISTINCT pl.Name AS hotel_name, pd.Brand, pl.Star_Category AS star_category, pl.City
    FROM `{PROJECT}.{DATASET}.product_list` pl
    JOIN `{PROJECT}.{DATASET}.product_description` pd ON pl.product_id = pd.product_id
    WHERE pd.Brand IS NOT NULL AND pl.Name IS NOT NULL AND pl.City IS NOT NULL
    """
    try:
        return client.query(query).to_dataframe()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_data(hotel_names: tuple):
    if client is None or not hotel_names:
        return pd.DataFrame()
    names_sql = "', '".join([n.replace("'", "''") for n in hotel_names])
    query = f"""
    SELECT pl.Name AS hotel_name, pd.Brand, pl.Star_Category, pl.City,
           s.aspect_id, s.treemap_name AS phrase, s.sentiment_type, COUNT(*) AS mention_count
    FROM `{PROJECT}.{DATASET}.product_user_review_enriched` e
    JOIN `{PROJECT}.{DATASET}.product_user_review_sentiment` s ON e.id = s.user_review_id
    JOIN `{PROJECT}.{DATASET}.product_list` pl ON e.product_id = pl.product_id
    JOIN `{PROJECT}.{DATASET}.product_description` pd ON pl.product_id = pd.product_id
    WHERE pl.Name IN ('{names_sql}')
    GROUP BY pl.Name, pd.Brand, pl.Star_Category, pl.City, s.aspect_id, s.treemap_name, s.sentiment_type
    """
    try:
        df = client.query(query).to_dataframe()
        df["aspect"] = df["aspect_id"].map(ASPECT_MAP).fillna("Other")
        df["mention_count"] = pd.to_numeric(df["mention_count"], errors="coerce").fillna(0).astype(int)
        return df
    except:
        return pd.DataFrame()

# ─────────────────────────────────────────
# CHAT CLIENT
# ─────────────────────────────────────────
@st.cache_resource
def get_chat_client():
    try:
        from google.cloud import geminidataanalytics_v1alpha as gda
        credentials = get_credentials()
        if credentials:
            from google.api_core import client_options
            return gda.DataChatServiceClient(credentials=credentials, client_options=client_options.ClientOptions())
        return gda.DataChatServiceClient()
    except:
        return None

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏨 Smaartbrand Intelligence</h1>
    <p>Deep Insights & AI Chat • Multi-language • Smart Corrections</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab_insights, tab_chat = st.tabs(["📊 Deep Insights", "💬 Chat with Data"])

# ═══════════════════════════════════════════
# TAB 1: DEEP INSIGHTS
# ═══════════════════════════════════════════
with tab_insights:
    
    with st.sidebar:
        st.markdown("### 🎯 Select Data")
        meta = get_metadata()
        selected_hotels = []
        
        if not meta.empty:
            compare_by = st.radio("Compare By", ["Hotel", "Brand", "City", "Star Rating"], horizontal=True)
            st.divider()
            
            if compare_by == "Hotel":
                all_brands = ["All"] + sorted(meta["Brand"].dropna().unique().tolist())
                brand_f = st.selectbox("Brand", all_brands)
                filtered = meta if brand_f == "All" else meta[meta["Brand"] == brand_f]
                
                all_cities = ["All"] + sorted(filtered["City"].dropna().unique().tolist())
                city_f = st.selectbox("City", all_cities)
                filtered = filtered if city_f == "All" else filtered[filtered["City"] == city_f]
                
                selected_hotels = st.multiselect("Hotels", sorted(filtered["hotel_name"].unique()), placeholder="Select...")
            
            elif compare_by == "Brand":
                brands = st.multiselect("Brands", sorted(meta["Brand"].unique()), placeholder="Select...")
                if brands:
                    selected_hotels = meta[meta["Brand"].isin(brands)]["hotel_name"].unique().tolist()
            
            elif compare_by == "City":
                cities = st.multiselect("Cities", sorted(meta["City"].unique()), placeholder="Select...")
                if cities:
                    selected_hotels = meta[meta["City"].isin(cities)]["hotel_name"].unique().tolist()
            
            elif compare_by == "Star Rating":
                stars = st.multiselect("Stars", sorted(meta["star_category"].unique()), 
                                       format_func=lambda x: f"{'⭐' * int(x)}", placeholder="Select...")
                if stars:
                    selected_hotels = meta[meta["star_category"].isin(stars)]["hotel_name"].unique().tolist()
            
            if selected_hotels:
                st.success(f"✓ {len(selected_hotels)} hotels")
    
    if not selected_hotels:
        st.info("👈 Use sidebar to select hotels, brands, cities or star ratings.")
    else:
        with st.spinner("Loading..."):
            df = fetch_data(tuple(selected_hotels))
        
        if df.empty:
            st.warning("No data found.")
        else:
            hotels_list = df["hotel_name"].unique().tolist()
            num_hotels = min(len(hotels_list), 8)
            
            # SATISFACTION SCORE
            st.markdown("<div class='section-header'>Satisfaction Score</div>", unsafe_allow_html=True)
            
            metrics = []
            for h in hotels_list:
                hd = df[df["hotel_name"] == h]
                pos = hd[hd["sentiment_type"].str.lower() == "positive"]["mention_count"].sum()
                neg = hd[hd["sentiment_type"].str.lower() == "negative"]["mention_count"].sum()
                total = pos + neg
                metrics.append({"hotel": h, "reviews": total, 
                                "pos_pct": round(pos/total*100) if total else 0,
                                "neg_pct": round(neg/total*100) if total else 0,
                                "score": round(pos/total*100) if total else 0})
            
            cols = st.columns([1.2] + [1] * num_hotels)
            cols[0].write("")
            for i, h in enumerate(hotels_list[:8]):
                cols[i+1].markdown(f"<div class='hotel-name-cell'>{h if len(h)<=20 else h[:18]+'...'}</div>", unsafe_allow_html=True)
            
            for label, key, color in [("Reviews","reviews",False),("+ve %","pos_pct",True),("-ve %","neg_pct",False),("Score","score",True)]:
                cols = st.columns([1.2] + [1] * num_hotels)
                cols[0].write(label)
                for i, m in enumerate(metrics[:8]):
                    bg = get_color_for_score(m[key]) if color else "#f1f5f9"
                    disp = f"{m[key]:,}" if key=="reviews" else f"{m[key]}%"
                    cols[i+1].markdown(f"<div class='score-cell' style='background:{bg};'>{disp}</div>", unsafe_allow_html=True)
            
            st.divider()
            
            # DRIVER ANALYSIS
            st.markdown("<div class='section-header'>Driver Analysis</div>", unsafe_allow_html=True)
            
            driver = []
            for h in hotels_list:
                hd = df[df["hotel_name"] == h]
                for asp in sorted(df["aspect"].unique()):
                    ad = hd[hd["aspect"] == asp]
                    pos = ad[ad["sentiment_type"].str.lower()=="positive"]["mention_count"].sum()
                    neg = ad[ad["sentiment_type"].str.lower()=="negative"]["mention_count"].sum()
                    total = pos + neg
                    driver.append({"hotel": h, "aspect": asp, "score": round(pos/total*100) if total else 0})
            
            driver_df = pd.DataFrame(driver)
            
            if len(hotels_list) > 1:
                pivot = driver_df.pivot(index="aspect", columns="hotel", values="score").fillna(0)
                cols = st.columns([1.2] + [1] * num_hotels)
                cols[0].markdown("**Aspect**")
                for i, h in enumerate(hotels_list[:8]):
                    cols[i+1].markdown(f"<div class='hotel-name-cell'>{h if len(h)<=20 else h[:18]+'...'}</div>", unsafe_allow_html=True)
                
                for asp in sorted(pivot.index):
                    cols = st.columns([1.2] + [1] * num_hotels)
                    cols[0].write(f"{ASPECT_ICONS.get(asp,'•')} {asp}")
                    for i, h in enumerate(hotels_list[:8]):
                        s = pivot.loc[asp, h] if h in pivot.columns else 0
                        cols[i+1].markdown(f"<div class='score-cell' style='background:{get_color_for_score(s)};'>{int(s)}%</div>", unsafe_allow_html=True)
            else:
                single = driver_df[driver_df["hotel"]==hotels_list[0]].sort_values("score")
                fig = go.Figure(go.Bar(y=[f"{ASPECT_ICONS.get(a,'•')} {a}" for a in single["aspect"]],
                    x=single["score"], orientation='h', marker_color=[get_color_for_score(s) for s in single["score"]],
                    text=[f"{int(s)}%" for s in single["score"]], textposition='outside'))
                fig.update_layout(height=280, margin=dict(l=0,r=40,t=10,b=0), xaxis=dict(range=[0,105]), plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # TOP PHRASES
            st.markdown("<div class='section-header'>Top Phrases</div>", unsafe_allow_html=True)
            pc1, pc2 = st.columns(2)
            sent_f = pc1.selectbox("Sentiment", ["All","Positive","Negative"], key="sf")
            asp_f = pc2.selectbox("Aspect", ["All"]+sorted(df["aspect"].unique().tolist()), key="af")
            
            pdf = df.copy()
            if sent_f != "All":
                pdf = pdf[pdf["sentiment_type"].str.lower()==sent_f.lower()]
            if asp_f != "All":
                pdf = pdf[pdf["aspect"]==asp_f]
            
            phrases = (pdf.groupby(["phrase","sentiment_type"])["mention_count"].sum().unstack(fill_value=0)
                .assign(total=lambda x: x.get("positive",0)+x.get("negative",0),
                        pos_pct=lambda x: x.get("positive",0)/(x.get("positive",0)+x.get("negative",0)).replace(0,1)*100)
                .sort_values("total",ascending=False).head(8).reset_index())
            
            if not phrases.empty:
                for i in range(0, len(phrases), 4):
                    cols = st.columns(4)
                    for j, (_, row) in enumerate(phrases.iloc[i:i+4].iterrows()):
                        with cols[j]:
                            txt = row["phrase"][:28]+"..." if len(str(row["phrase"]))>28 else row["phrase"]
                            pct = row.get("pos_pct",50)
                            st.markdown(f"""<div class='phrase-card'>
                                <div class='phrase-text'>{txt}</div>
                                <div class='phrase-count'>{int(row['total']):,}</div>
                                <div style='height:5px;border-radius:3px;background:linear-gradient(to right,#10b981 {pct}%,#ef4444 {pct}%);margin-top:6px;'></div>
                            </div>""", unsafe_allow_html=True)
            
            st.divider()
            
            # BRAND ASSOCIATIONS (multiple hotels)
            if len(hotels_list) > 1:
                st.markdown("<div class='section-header'>Brand Associations</div>", unsafe_allow_html=True)
                st.caption("● Strong  ◐ Weak  ○ None")
                
                assoc = []
                for asp in sorted(df["aspect"].unique()):
                    row = {"aspect": asp}
                    scores = []
                    for h in hotels_list[:8]:
                        hd = df[(df["hotel_name"]==h)&(df["aspect"]==asp)]
                        pos = hd[hd["sentiment_type"].str.lower()=="positive"]["mention_count"].sum()
                        neg = hd[hd["sentiment_type"].str.lower()=="negative"]["mention_count"].sum()
                        s = pos/(pos+neg)*100 if (pos+neg) else 0
                        row[h] = s
                        scores.append(s)
                    avg = np.mean(scores) if scores else 50
                    for h in hotels_list[:8]:
                        diff = row.get(h,0) - avg
                        row[f"{h}_a"] = "strong" if diff>10 else ("weak" if diff>-5 else "none")
                    assoc.append(row)
                
                cols = st.columns([1.2]+[1]*num_hotels)
                cols[0].markdown("**Aspect**")
                for i, h in enumerate(hotels_list[:8]):
                    cols[i+1].markdown(f"<div class='hotel-name-cell'>{h if len(h)<=20 else h[:18]+'...'}</div>", unsafe_allow_html=True)
                
                for row in assoc:
                    cols = st.columns([1.2]+[1]*num_hotels)
                    cols[0].write(f"{ASPECT_ICONS.get(row['aspect'],'•')} {row['aspect']}")
                    for i, h in enumerate(hotels_list[:8]):
                        a = row.get(f"{h}_a","none")
                        dot = {"strong":"dot-strong","weak":"dot-weak","none":"dot-none"}[a]
                        cols[i+1].markdown(f"<div style='text-align:center;'><span class='{dot}'></span></div>", unsafe_allow_html=True)
                
                st.divider()
            
            # ─────────────────────────────────────────
            # SMAARTANALYST (BigQuery ML)
            # ─────────────────────────────────────────
            st.markdown("<div class='section-header'>🤖 SmaartAnalyst</div>", unsafe_allow_html=True)
            st.caption("Ask questions about the selected data • Multi-language • Powered by BigQuery ML + Gemini")
            
            # Context description
            ctx_brands = df["Brand"].unique().tolist()
            ctx_cities = df["City"].unique().tolist()
            if len(hotels_list) == 1:
                st.markdown(f"Analyzing: **{hotels_list[0]}**")
            elif len(ctx_brands) == 1:
                st.markdown(f"Analyzing: **{ctx_brands[0]}** hotels")
            elif len(ctx_cities) == 1:
                st.markdown(f"Analyzing: hotels in **{ctx_cities[0]}**")
            else:
                st.markdown(f"Analyzing: **{len(hotels_list)} hotels**")
            
            # Build context
            def build_context(data_df):
                def calc_sat(g):
                    pos = g[g["sentiment_type"].str.lower()=="positive"]["mention_count"].sum()
                    neg = g[g["sentiment_type"].str.lower()=="negative"]["mention_count"].sum()
                    return round(pos/(pos+neg)*100 if (pos+neg) else 0, 1)
                
                ah = data_df.groupby(["hotel_name","aspect"]).apply(calc_sat).reset_index(name="sat_pct")
                dr = data_df.groupby("aspect").apply(lambda g: pd.Series({
                    "mentions": int(g["mention_count"].sum()), "sat_pct": calc_sat(g)
                })).reset_index()
                pp = data_df[data_df["sentiment_type"].str.lower()=="positive"].groupby("phrase")["mention_count"].sum().nlargest(10).reset_index(name="cnt")
                np_ = data_df[data_df["sentiment_type"].str.lower()=="negative"].groupby("phrase")["mention_count"].sum().nlargest(10).reset_index(name="cnt")
                
                return {"aspect_hotel": ah.to_csv(index=False), "driver": dr.to_csv(index=False),
                        "pos_phrases": pp.to_csv(index=False), "neg_phrases": np_.to_csv(index=False),
                        "hotels": ", ".join(data_df["hotel_name"].unique()[:10]),
                        "brands": ", ".join(data_df["Brand"].unique()),
                        "cities": ", ".join(data_df["City"].unique()),
                        "stars": ", ".join([str(int(s)) for s in data_df["Star_Category"].unique()])}
            
            def run_analyst(q, data_df, hist):
                ctx = build_context(data_df)
                hist_txt = "\n".join([f"{'User' if m['role']=='user' else 'SmaartAnalyst'}: {m['content']}" for m in hist[-4:]])
                
                prompt = f"""You are SmaartAnalyst, a hospitality brand intelligence expert.

=== CONTEXT ===
Hotels: {ctx['hotels']}
Brands: {ctx['brands']}
Cities: {ctx['cities']}
Stars: {ctx['stars']}

=== SPELLING CORRECTIONS (apply to user query) ===
Cities: bangalore/blr/banglore→Bengaluru, bombay→Mumbai, madras→Chennai, calcutta→Kolkata, hyd→Hyderabad
Hotels: marriot→Marriott, oberoy→Oberoi, viventa→Vivanta, lela→Leela, hyat→Hyatt
Aspects: food/restaurant→Dining, clean/hygiene→Cleanliness, price/cost/value→Value for Money

=== HISTORY ===
{hist_txt}

=== DATA ===
[Aspect x Hotel Satisfaction %]
{ctx['aspect_hotel']}

[Driver Analysis]
{ctx['driver']}

[Top Positive Phrases]
{ctx['pos_phrases']}

[Top Negative Phrases]
{ctx['neg_phrases']}

=== QUESTION ===
{q}

=== RULES ===
1. Apply spelling corrections first.
2. Answer ONLY from data. Never hallucinate.
3. If query is non-English, respond in SAME language.
4. Cite specific scores and hotel names.
5. Lead with key finding, explain, give 1-2 recommendations.
6. Max 250 words. No markdown headers.
7. Satisfaction 0-100 (100%=all positive).
"""
                sql = f"""SELECT ml_generate_text_llm_result FROM ML.GENERATE_TEXT(
                    MODEL `{PROJECT}.{DATASET}.gemini_flash_model`,
                    (SELECT @prompt AS prompt),
                    STRUCT(0.3 AS temperature, 900 AS max_output_tokens, TRUE AS flatten_json_output))"""
                try:
                    job_cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("prompt","STRING",prompt)])
                    result = client.query(sql, job_config=job_cfg).to_dataframe()
                    return result["ml_generate_text_llm_result"].iloc[0].strip()
                except Exception as e:
                    return f"Error: {e}"
            
            # Chat state
            if "analyst_history" not in st.session_state:
                st.session_state.analyst_history = []
            
            for msg in st.session_state.analyst_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Suggestions
            if not st.session_state.analyst_history:
                st.markdown("**💡 Suggested:**")
                if len(hotels_list) == 1:
                    sugs = [f"Strengths of {hotels_list[0][:20]}?", "What needs improvement?", "Dining feedback?", "3 action items?"]
                else:
                    sugs = [f"Compare {hotels_list[0][:12]} vs {hotels_list[1][:12]}", "Best cleanliness?", "Biggest weakness?", "Rank by staff"]
                
                c1, c2 = st.columns(2)
                for i, s in enumerate(sugs):
                    col = c1 if i%2==0 else c2
                    if col.button(s, key=f"as_{i}", use_container_width=True):
                        st.session_state.analyst_history.append({"role":"user","content":s})
                        with st.chat_message("user"): st.write(s)
                        with st.chat_message("assistant"):
                            with st.spinner("Analyzing..."):
                                resp = run_analyst(s, df, st.session_state.analyst_history)
                                st.write(resp)
                        st.session_state.analyst_history.append({"role":"assistant","content":resp})
                        st.rerun()
            
            aq = st.chat_input("Ask about selected hotels...", key="analyst_q")
            if aq:
                st.session_state.analyst_history.append({"role":"user","content":aq})
                with st.chat_message("user"): st.write(aq)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        resp = run_analyst(aq, df, st.session_state.analyst_history)
                        st.write(resp)
                st.session_state.analyst_history.append({"role":"assistant","content":resp})
            
            if st.session_state.analyst_history:
                if st.button("🗑️ Clear SmaartAnalyst", key="clr_a"):
                    st.session_state.analyst_history = []
                    st.rerun()

# ═══════════════════════════════════════════
# TAB 2: CHAT (Gemini Data Analytics Agent)
# ═══════════════════════════════════════════
with tab_chat:
    st.markdown("### 💬 Hotel Data Intelligence")
    st.caption("Ask in any language • Auto-corrects spelling • Powered by Gemini Data Analytics Agent")
    
    with st.expander("🌐 Languages & Corrections", expanded=False):
        c1, c2 = st.columns(2)
        c1.markdown("**Languages:** English, हिंदी, தமிழ், తెలుగు, ಕನ್ನಡ...")
        c1.markdown("**Cities:** Bangalore→Bengaluru, Bombay→Mumbai")
        c2.markdown("**Hotels:** Marriot→Marriott, Oberoy→Oberoi")
        c2.markdown("**Aspects:** Food→Dining, Clean→Cleanliness")
    
    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = f"smaart-{uuid.uuid4().hex[:6]}"
    
    for msg in st.session_state.chat_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    def preprocess(q):
        aliases = {"bangalore":"Bengaluru","blr":"Bengaluru","bombay":"Mumbai","madras":"Chennai",
                   "marriot":"Marriott","oberoy":"Oberoi","food":"Dining","clean":"Cleanliness"}
        for a, c in aliases.items():
            q = re.sub(r'\b'+re.escape(a)+r'\b', c, q, flags=re.IGNORECASE)
        return q
    
    ui = st.chat_input("Ask about hotels... (e.g., 'बेंगलुरु में सबसे अच्छा होटल?')")
    
    if ui:
        processed = preprocess(ui)
        enhanced = f"User Query: {processed}\n\nInstructions:\n1. If non-English, respond in SAME language.\n2. Use corrected names.\n3. Provide data-driven insights.\n\nOriginal: {ui}"
        
        st.session_state.chat_msgs.append({"role":"user","content":ui})
        with st.chat_message("user"): st.markdown(ui)
        
        with st.chat_message("assistant"):
            ph = st.empty()
            try:
                from google.cloud import geminidataanalytics_v1alpha as gda
                cc = get_chat_client()
                if cc:
                    parent = f"projects/{PROJECT}/locations/{LOCATION}"
                    agent = f"{parent}/dataAgents/{AGENT_ID}"
                    conv = cc.conversation_path(PROJECT, LOCATION, st.session_state.chat_id)
                    
                    try: cc.get_conversation(name=conv)
                    except: cc.create_conversation(request=gda.CreateConversationRequest(
                        parent=parent, conversation_id=st.session_state.chat_id, conversation=gda.Conversation(agents=[agent])))
                    
                    stream = cc.chat(request={"parent":parent,
                        "conversation_reference":{"conversation":conv,"data_agent_context":{"data_agent":agent}},
                        "messages":[{"user_message":{"text":enhanced}}]})
                    
                    resp = ""
                    for chunk in stream:
                        if hasattr(chunk,'system_message') and hasattr(chunk.system_message,'text'):
                            for p in chunk.system_message.text.parts: st.caption(f"💭 {p}")
                        if hasattr(chunk,'agent_message') and hasattr(chunk.agent_message,'text'):
                            for p in chunk.agent_message.text.parts:
                                resp += str(p)
                                ph.markdown(resp+"▌")
                        elif hasattr(chunk,'message') and hasattr(chunk.message,'content'):
                            for p in chunk.message.content.parts:
                                resp += p.text if hasattr(p,'text') else str(p)
                                ph.markdown(resp+"▌")
                    
                    if resp:
                        ph.markdown(resp)
                        st.session_state.chat_msgs.append({"role":"assistant","content":resp})
                    else:
                        ph.markdown("Done.")
                else:
                    ph.markdown("⚠️ Chat unavailable.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.session_state.chat_msgs:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_msgs = []
            st.session_state.chat_id = f"smaart-{uuid.uuid4().hex[:6]}"
            st.rerun()

st.divider()
st.caption("🏨 Smaartbrand Intelligence • Multi-language • BigQuery + Gemini")

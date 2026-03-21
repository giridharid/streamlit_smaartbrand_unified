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
           s.aspect_id, s.treemap_name AS phrase, s.sentiment_type, 
           e.Review_date,
           e.inferred_gender AS gender,
           e.traveler_type,
           e.stay_purpose,
           COUNT(*) AS mention_count
    FROM `{PROJECT}.{DATASET}.product_user_review_enriched` e
    JOIN `{PROJECT}.{DATASET}.product_user_review_sentiment` s ON e.id = s.user_review_id
    JOIN `{PROJECT}.{DATASET}.product_list` pl ON e.product_id = pl.product_id
    JOIN `{PROJECT}.{DATASET}.product_description` pd ON pl.product_id = pd.product_id
    WHERE pl.Name IN ('{names_sql}')
    GROUP BY pl.Name, pd.Brand, pl.Star_Category, pl.City, s.aspect_id, s.treemap_name, s.sentiment_type, e.Review_date, e.inferred_gender, e.traveler_type, e.stay_purpose
    """
    try:
        df = client.query(query).to_dataframe()
        df["aspect"] = df["aspect_id"].map(ASPECT_MAP).fillna("Other")
        df["mention_count"] = pd.to_numeric(df["mention_count"], errors="coerce").fillna(0).astype(int)
        if "Review_date" in df.columns:
            df["Review_date"] = pd.to_datetime(df["Review_date"], errors="coerce")
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_recent_trends(hotel_names: tuple, months: int = 3):
    """Fetch sentiment trends for last N months"""
    if client is None or not hotel_names:
        return {}
    
    names_sql = "', '".join([n.replace("'", "''") for n in hotel_names])
    
    query = f"""
    WITH recent_data AS (
        SELECT 
            pl.Name AS hotel_name,
            s.aspect_id,
            s.sentiment_type,
            s.treemap_name AS phrase,
            e.Review_date,
            COUNT(*) AS mention_count
        FROM `{PROJECT}.{DATASET}.product_user_review_enriched` e
        JOIN `{PROJECT}.{DATASET}.product_user_review_sentiment` s ON e.id = s.user_review_id
        JOIN `{PROJECT}.{DATASET}.product_list` pl ON e.product_id = pl.product_id
        WHERE pl.Name IN ('{names_sql}')
          AND e.Review_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {months} MONTH)
        GROUP BY pl.Name, s.aspect_id, s.sentiment_type, s.treemap_name, e.Review_date
    ),
    aspect_summary AS (
        SELECT 
            hotel_name,
            aspect_id,
            sentiment_type,
            SUM(mention_count) AS total_mentions
        FROM recent_data
        GROUP BY hotel_name, aspect_id, sentiment_type
    ),
    top_phrases AS (
        SELECT 
            hotel_name,
            sentiment_type,
            phrase,
            SUM(mention_count) AS phrase_count,
            ROW_NUMBER() OVER (PARTITION BY hotel_name, sentiment_type ORDER BY SUM(mention_count) DESC) AS rn
        FROM recent_data
        GROUP BY hotel_name, sentiment_type, phrase
    )
    SELECT 
        a.hotel_name,
        a.aspect_id,
        a.sentiment_type,
        a.total_mentions,
        p.phrase AS top_phrase,
        p.phrase_count
    FROM aspect_summary a
    LEFT JOIN top_phrases p ON a.hotel_name = p.hotel_name 
        AND a.sentiment_type = p.sentiment_type 
        AND p.rn <= 5
    ORDER BY a.hotel_name, a.aspect_id, a.sentiment_type
    """
    
    try:
        result = client.query(query).to_dataframe()
        return result.to_dict('records') if not result.empty else []
    except:
        return []

@st.cache_data(ttl=3600)
def fetch_location_context(hotel_names: tuple):
    """Fetch nearby competitors with their satisfaction scores by aspect"""
    if client is None or not hotel_names:
        return []
    
    names_sql = "', '".join([n.replace("'", "''") for n in hotel_names])
    
    query = f"""
    WITH selected_hotels AS (
        SELECT 
            pl.Name AS hotel_name,
            pl.City,
            pl.Star_Category,
            pd.Brand,
            pd.Latitude,
            pd.Longitude,
            pd.Address
        FROM `{PROJECT}.{DATASET}.product_list` pl
        JOIN `{PROJECT}.{DATASET}.product_description` pd ON pl.product_id = pd.product_id
        WHERE pl.Name IN ('{names_sql}')
    ),
    nearby_competitors AS (
        SELECT 
            sh.hotel_name AS selected_hotel,
            pl.Name AS competitor_name,
            pd.Brand AS competitor_brand,
            pl.Star_Category AS competitor_stars,
            pl.City AS competitor_city,
            CASE 
                WHEN sh.Latitude IS NOT NULL AND pd.Latitude IS NOT NULL THEN
                    ROUND(ST_DISTANCE(
                        ST_GEOGPOINT(sh.Longitude, sh.Latitude),
                        ST_GEOGPOINT(pd.Longitude, pd.Latitude)
                    ) / 1000, 1)
                ELSE NULL
            END AS distance_km
        FROM selected_hotels sh
        CROSS JOIN `{PROJECT}.{DATASET}.product_list` pl
        JOIN `{PROJECT}.{DATASET}.product_description` pd ON pl.product_id = pd.product_id
        WHERE pl.Name NOT IN ('{names_sql}')
          AND pl.City = sh.City
          AND ABS(pl.Star_Category - sh.Star_Category) <= 1
    ),
    competitor_aspect_scores AS (
        SELECT 
            pl.Name AS hotel_name,
            s.aspect_id,
            ROUND(SUM(CASE WHEN s.sentiment_type = 'positive' THEN 1 ELSE 0 END) * 100.0 / 
                  NULLIF(COUNT(*), 0), 0) AS satisfaction_pct
        FROM `{PROJECT}.{DATASET}.product_user_review_sentiment` s
        JOIN `{PROJECT}.{DATASET}.product_user_review_enriched` e ON s.user_review_id = e.id
        JOIN `{PROJECT}.{DATASET}.product_list` pl ON e.product_id = pl.product_id
        GROUP BY pl.Name, s.aspect_id
    ),
    competitor_overall AS (
        SELECT 
            hotel_name,
            ROUND(AVG(satisfaction_pct), 0) AS overall_satisfaction
        FROM competitor_aspect_scores
        GROUP BY hotel_name
    )
    SELECT 
        nc.selected_hotel,
        nc.competitor_name,
        nc.competitor_brand,
        nc.competitor_stars,
        nc.distance_km,
        co.overall_satisfaction,
        -- Pivot aspect scores
        MAX(CASE WHEN cas.aspect_id = 1 THEN cas.satisfaction_pct END) AS dining_score,
        MAX(CASE WHEN cas.aspect_id = 2 THEN cas.satisfaction_pct END) AS cleanliness_score,
        MAX(CASE WHEN cas.aspect_id = 3 THEN cas.satisfaction_pct END) AS amenities_score,
        MAX(CASE WHEN cas.aspect_id = 4 THEN cas.satisfaction_pct END) AS staff_score,
        MAX(CASE WHEN cas.aspect_id = 5 THEN cas.satisfaction_pct END) AS room_score,
        MAX(CASE WHEN cas.aspect_id = 6 THEN cas.satisfaction_pct END) AS location_score,
        MAX(CASE WHEN cas.aspect_id = 7 THEN cas.satisfaction_pct END) AS value_score
    FROM nearby_competitors nc
    LEFT JOIN competitor_overall co ON nc.competitor_name = co.hotel_name
    LEFT JOIN competitor_aspect_scores cas ON nc.competitor_name = cas.hotel_name
    GROUP BY nc.selected_hotel, nc.competitor_name, nc.competitor_brand, nc.competitor_stars, nc.distance_km, co.overall_satisfaction
    ORDER BY nc.distance_km
    LIMIT 10
    """
    
    try:
        result = client.query(query).to_dataframe()
        return result.to_dict('records') if not result.empty else []
    except Exception as e:
        return []

@st.cache_data(ttl=3600)
def fetch_hotel_details(hotel_names: tuple):
    """Fetch hotel location details including nearby landmarks"""
    if client is None or not hotel_names:
        return []
    
    names_sql = "', '".join([n.replace("'", "''") for n in hotel_names])
    
    query = f"""
    SELECT 
        pl.Name AS hotel_name,
        pl.City,
        pl.Star_Category,
        pd.Brand,
        pd.Address,
        pd.Latitude,
        pd.Longitude,
        pd.Rating AS google_rating,
        pd.Amenities
    FROM `{PROJECT}.{DATASET}.product_list` pl
    JOIN `{PROJECT}.{DATASET}.product_description` pd ON pl.product_id = pd.product_id
    WHERE pl.Name IN ('{names_sql}')
    """
    
    try:
        result = client.query(query).to_dataframe()
        return result.to_dict('records') if not result.empty else []
    except:
        return []

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
    <h1>&#127976; Smaartbrand Intelligence</h1>
    <p>Deep Insights & AI Chat - Multi-language - Smart Corrections</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab_insights, tab_chat = st.tabs(["Deep Insights", "Chat with Data"])

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
            def build_context(data_df, hotel_names_tuple):
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
                
                # Recent trends (last 3 months)
                recent_info = ""
                if "Review_date" in data_df.columns and data_df["Review_date"].notna().any():
                    three_months_ago = pd.Timestamp.now() - pd.DateOffset(months=3)
                    recent_df = data_df[data_df["Review_date"] >= three_months_ago]
                    
                    if not recent_df.empty:
                        # Recent positive phrases
                        recent_pos = recent_df[recent_df["sentiment_type"].str.lower()=="positive"].groupby("phrase")["mention_count"].sum().nlargest(5)
                        recent_neg = recent_df[recent_df["sentiment_type"].str.lower()=="negative"].groupby("phrase")["mention_count"].sum().nlargest(5)
                        
                        # Recent aspect scores
                        recent_aspects = recent_df.groupby("aspect").apply(calc_sat).reset_index(name="sat_pct")
                        
                        recent_info = f"\n[LAST 3 MONTHS DATA]\n"
                        recent_info += f"Recent Reviews: {int(recent_df['mention_count'].sum())} mentions\n"
                        recent_info += f"Recent Positive Phrases: {', '.join(recent_pos.index.tolist())}\n"
                        recent_info += f"Recent Negative Phrases: {', '.join(recent_neg.index.tolist())}\n"
                        recent_info += f"Recent Aspect Scores:\n{recent_aspects.to_csv(index=False)}"
                
                # Persona & Traveler Analysis
                persona_info = ""
                if "traveler_type" in data_df.columns:
                    traveler_dist = data_df.groupby("traveler_type")["mention_count"].sum()
                    if not traveler_dist.empty:
                        total = traveler_dist.sum()
                        persona_info += "\n[TRAVELER MIX]\n"
                        for t, cnt in traveler_dist.nlargest(5).items():
                            if pd.notna(t) and t:
                                persona_info += f"• {t}: {cnt/total*100:.1f}%\n"
                
                if "stay_purpose" in data_df.columns:
                    purpose_dist = data_df.groupby("stay_purpose")["mention_count"].sum()
                    if not purpose_dist.empty:
                        total = purpose_dist.sum()
                        persona_info += "\n[STAY PURPOSE]\n"
                        for p, cnt in purpose_dist.nlargest(5).items():
                            if pd.notna(p) and p:
                                persona_info += f"• {p}: {cnt/total*100:.1f}%\n"
                
                if "gender" in data_df.columns:
                    gender_dist = data_df.groupby("gender")["mention_count"].sum()
                    if not gender_dist.empty:
                        total = gender_dist.sum()
                        persona_info += "\n[GENDER MIX]\n"
                        for g, cnt in gender_dist.items():
                            if pd.notna(g) and g:
                                persona_info += f"• {g}: {cnt/total*100:.1f}%\n"
                
                # Persona-specific insights (what each segment complains about)
                persona_insights = ""
                if "traveler_type" in data_df.columns:
                    for ttype in data_df["traveler_type"].dropna().unique()[:3]:
                        tdata = data_df[data_df["traveler_type"] == ttype]
                        neg_phrases = tdata[tdata["sentiment_type"].str.lower()=="negative"].groupby("phrase")["mention_count"].sum().nlargest(3)
                        if not neg_phrases.empty:
                            persona_insights += f"\n{ttype} travelers complain about: {', '.join(neg_phrases.index.tolist())}"
                
                # Fetch location context (nearby competitors)
                location_ctx = fetch_location_context(hotel_names_tuple)
                hotel_details = fetch_hotel_details(hotel_names_tuple)
                
                # Format competitor data with aspect scores
                competitor_info = ""
                if location_ctx:
                    for comp in location_ctx[:5]:  # Top 5 competitors
                        competitor_info += f"\n• {comp.get('competitor_name', 'N/A')} ({comp.get('competitor_brand', '')} {comp.get('competitor_stars', '')}⭐)"
                        competitor_info += f" - {comp.get('distance_km', '?')}km away"
                        competitor_info += f" - Overall: {comp.get('overall_satisfaction', '?')}%"
                        competitor_info += f" | Dining:{comp.get('dining_score', '?')}%"
                        competitor_info += f" Staff:{comp.get('staff_score', '?')}%"
                        competitor_info += f" Room:{comp.get('room_score', '?')}%"
                        competitor_info += f" Cleanliness:{comp.get('cleanliness_score', '?')}%"
                        competitor_info += f" Amenities:{comp.get('amenities_score', '?')}%"
                        competitor_info += f" Value:{comp.get('value_score', '?')}%"
                
                # Format hotel location details
                location_info = ""
                if hotel_details:
                    for h in hotel_details:
                        location_info += f"\n{h.get('hotel_name', 'N/A')}: "
                        location_info += f"Address: {h.get('Address', 'N/A')}, "
                        location_info += f"Google Rating: {h.get('google_rating', 'N/A')}/5, "
                        location_info += f"Amenities: {h.get('Amenities', 'N/A')[:100]}..."
                
                return {
                    "aspect_hotel": ah.to_csv(index=False), 
                    "driver": dr.to_csv(index=False),
                    "pos_phrases": pp.to_csv(index=False), 
                    "neg_phrases": np_.to_csv(index=False),
                    "recent_trends": recent_info if recent_info else "Recent data not available",
                    "persona_data": persona_info if persona_info else "Persona data not available",
                    "persona_insights": persona_insights if persona_insights else "",
                    "hotels": ", ".join(data_df["hotel_name"].unique()[:10]),
                    "brands": ", ".join(data_df["Brand"].unique()),
                    "cities": ", ".join(data_df["City"].unique()),
                    "stars": ", ".join([str(int(s)) for s in data_df["Star_Category"].unique()]),
                    "competitors": competitor_info if competitor_info else "No nearby competitors found",
                    "location_details": location_info if location_info else "Location details not available"
                }
            
            def run_analyst(q, data_df, hist):
                ctx = build_context(data_df, tuple(hotels_list))
                hist_txt = "\n".join([f"{'User' if m['role']=='user' else 'SmaartAnalyst'}: {m['content']}" for m in hist[-4:]])
                
                prompt = f"""You are SmaartAnalyst, an AI-powered decision intelligence assistant for the hospitality industry.

=== WHO YOU SERVE ===
Hotel operations teams who need actionable intelligence from guest feedback:
• Brand Manager - Brand perception, competitive positioning, reputation management
• SEO & Marketing - Keywords, USPs, ad copy, competitive differentiation, what to highlight
• Housekeeping - Room cleanliness, bathroom hygiene, linen quality, maintenance
• Front Desk - Check-in/out experience, staff behavior, reception, concierge
• Operations - Overall service delivery, process improvements, staff training
• F&B (Food & Beverage) - Restaurant quality, breakfast, dining experience, menu

=== YOUR PURPOSE ===
Transform guest sentiment into DECISIONS and ACTIONS by department.
Use location intelligence to provide competitive positioning advice.
Support R&D mode for new hotel planning with persona and competitor insights.
Every response MUST end with categorized action items.

=== CONTEXT ===
Hotels: {ctx['hotels']}
Brands: {ctx['brands']}
Cities: {ctx['cities']}
Star Category: {ctx['stars']}

=== LOCATION INTELLIGENCE ===
{ctx['location_details'] if ctx['location_details'] else 'Location details not available'}

=== NEARBY COMPETITORS (same area, similar star category) ===
{ctx['competitors'] if ctx['competitors'] else 'Competitor data not available'}

=== GUEST PERSONAS & TRAVELER DATA ===
{ctx.get('persona_data', 'Persona data not available')}
{ctx.get('persona_insights', '')}

=== DEMOGRAPHIC DATA ===
You have access to enriched demographic data:

GENDER (inferred_gender): Male, Female, or NULL
- Use for gender-specific marketing, understanding audience mix

TRAVELER TYPE (traveler_type): Business, Family, Couple, Solo, Group, or NULL
- Business = work travelers, value WiFi, location, quick service
- Family = kids, need space, pool, dining variety
- Couple = romantic, value ambiance, dining, room quality
- Solo = value safety, location, value for money
- Group = friends/colleagues, value amenities, social spaces

SECONDARY TRAVELER TYPE (secondary_traveler_type): Second most likely type if detected

STAY PURPOSE (stay_purpose): Leisure, Business, Event, Transit, Honeymoon, or NULL

RULES:
- IGNORE NULL values — never display "Unknown" or NULL
- When showing percentages, exclude NULL from calculations
- Cross-reference: "Business travelers rate WiFi 67% vs Family at 82%"
- Always show 👥 Guest Mix and ♂️♀️ Gender Split when demographic data available

=== RECENT TRENDS (Last 3 Months) ===
{ctx['recent_trends'] if ctx.get('recent_trends') else 'Recent data not available'}

=== SPELLING CORRECTIONS ===
Cities: bangalore/blr/banglore→Bengaluru, bombay→Mumbai, madras→Chennai, calcutta→Kolkata
Hotels: marriot→Marriott, oberoy→Oberoi, viventa→Vivanta, lela→Leela
Aspects: food/restaurant→Dining, clean/hygiene→Cleanliness, price/cost→Value for Money

=== CONVERSATION HISTORY ===
{hist_txt}

=== DATA ===
[Aspect x Hotel Satisfaction % - ALL TIME]
{ctx['aspect_hotel']}

[Driver Analysis]
{ctx['driver']}

[What Guests Love - Top Phrases]
{ctx['pos_phrases']}

[What Guests Complain About - Top Phrases]
{ctx['neg_phrases']}

=== QUESTION ===
{q}

=== RESPONSE FORMAT (ALWAYS FOLLOW THIS) ===

📊 **Insight**: [2-3 sentence summary with specific scores. Include persona/traveler data if relevant. Compare to competitors if available.]

👥 **Guest Mix**: [Business X%, Family Y%, Couple Z%, Solo W%, Group V%] (if demographic data available)
♂️♀️ **Gender Split**: [Male X%, Female Y%] (if demographic data available)

🎯 **Actions by Department** (for general queries only — skip for SEO/Ad Copy/FAQ queries):

👔 **Brand Manager**: 
   - Positioning: [How to position vs competitors based on data]
   - Target Segment: [Which traveler type to focus on and why]
   - Example: "Position as #1 Family Hotel — you lead competitors by 15% with families"

📢 **SEO & Marketing**: 
   ✓ PROMOTE: [2-3 keywords where you win, from guest phrases]
   ✗ AVOID: [1-2 keywords where competitor wins]
   🎯 Target Audience: [Specific segment + platform]
   Ad Copy: "[Exact guest phrase to use in ads]"
   Example: "Target 'best breakfast Aerocity' — you score 94% on Dining"

🛏️ **Housekeeping**: [Specific action based on cleanliness/room data]
   Example: "Audit bathroom cleaning — negative mentions up 12% this quarter"

🛎️ **Front Desk**: [Specific action based on staff/service data]
   Example: "Train on check-in speed — Business travelers complain about 'long wait'"

⚙️ **Operations**: [Process improvement or training action]
   Example: "Implement noise audit — 23% of Business travelers mention 'noisy rooms'"

🍽️ **F&B**: [Specific dining action based on data]
   Example: "Extend room service to 11pm — late-night complaints from Business segment"

RULES FOR ACTIONS:
- Make each action SPECIFIC and EXECUTABLE today
- Tie every action to DATA (scores, phrases, segments)
- Include 3-4 most relevant departments based on the query
- Every action must answer: WHO does WHAT by WHEN

=== COMPETITIVE INTELLIGENCE RULES ===
When competitor data is available:
1. COMPARE aspect scores: "Your Dining (89%) beats Taj (78%) - PROMOTE THIS"
2. FIND GAPS: "Competitor weak on Staff (65%) - steal with 'legendary service' messaging"
3. IDENTIFY THREATS: "Competitor beats you on Pool (88% vs 72%) - AVOID 'pool' keywords"
4. SUGGEST ATTACK: For "How do I beat X?" questions, give specific win/lose breakdown

=== R&D MODE (New Hotel Planning) ===
When asked about opening a new hotel, planning, or R&D:
1. Analyze COMPETITOR landscape in that area (use nearby competitors data)
2. Show TRAVELER MIX: What type of guests visit this area? (Business/Couple/Family/Solo)
3. Show STAY PURPOSE: Why do people come? (Business/Leisure/Wedding/Conference)
4. Identify GAPS: What are competitors weak at? What's underserved?
5. Provide BUILD RECOMMENDATIONS: What to focus on based on gaps

Format for R&D queries:
📊 **Market Analysis**: [Area competitive landscape with scores]
👥 **Traveler Mix**: [Business X%, Couples Y%, Family Z%] -- ignore unknown
🎯 **Purpose**: [Business X%, Leisure Y%] - ignore unknown
⚠️ **Competitor Weaknesses**: [What they're bad at = your opportunity]
🛠️ **Build Recommendations**: [What to invest in based on gaps]

=== PERSONA-BASED INSIGHTS ===
When asked about specific traveler types or genders:
- Filter and analyze by traveler_type (Solo, Couple, Family, Business, Group)
- Filter and analyze by gender (Male, Female)
- Cross-reference: "Male Business travelers vs Female Business travelers"
- Show segment-specific satisfaction scores
- Show what each segment praises and complains about
- IGNORE NULL values — exclude from analysis

Example queries to handle:
- "What do business travelers think?" → Filter by traveler_type = Business
- "How do families rate our pool?" → Filter by traveler_type = Family, aspect = Amenities
- "Male vs Female satisfaction" → Compare by gender
- "What do couples complain about?" → Filter by traveler_type = Couple, sentiment = negative

Format for persona queries:
📊 **[Segment] Insights**: 
👥 Volume: [X]% of total reviews
✅ What they love: [top positive phrases + scores]
❌ Pain points: [top negative phrases + scores]
🎯 **Actions for [Segment]**: [Specific actions]


=== SEO & MARKETING SPECIAL INSTRUCTIONS ===
- Use actual guest phrases for ad copy: "guests say 'best biryani' → use in Google Ads"
- Location keywords: "near [landmark]", "best [aspect] in [area]"
- Competitor weakness = your keyword opportunity
- Format USPs as: "✓ PROMOTE: [strength]" and "✗ AVOID: [weakness]"

=== FAQ GENERATION RULES ===
When asked for FAQs (for website, SEO, GEO-based, etc.):
- Generate 5-8 actual Q&A pairs based on the SELECTED hotel's data
- Use the hotel's City, Address, and Location data from context
- Pull actual guest phrases from positive/negative reviews
- Focus on what guests ACTUALLY ask about (based on review mentions)
- Categories to cover: Location/Transport, Dining, Amenities, Value, Room quality
- Make answers specific using real satisfaction scores and guest quotes

Format each FAQ as:
**Q: [Natural question a guest would ask]?**
A: [Answer using actual data - scores, guest phrases, location details]

Generate FAQs dynamically based on:
1. Hotel's strongest aspects (highest satisfaction %) → Promote in answers
2. Hotel's location/city from context → Use for "near X" questions  
3. Top positive phrases → Quote in answers
4. Common concerns from negative phrases → Address proactively

=== DATA RULES ===
1. Show % satisfaction scores ONLY for aspects — never show mention counts
2. Star ratings (e.g., "4.5-star") are OK if from product_list.Rating or product_description.Rating
3. For review volume, say "thousands of reviews" — never cite exact numbers
4. Round satisfaction % to whole numbers (87% not 87.1%)
5. IGNORE NULL in gender, traveler_type, stay_purpose — exclude from analysis
6. When showing segment %, exclude NULL from denominator

=== ANTI-HALLUCINATION RULES (CRITICAL) ===
1. Answer ONLY from data provided. If data is missing, say "data not available."
2. NEVER invent satisfaction scores, percentages, or rankings.
3. NEVER guess guest phrases — use ONLY phrases from the data.
4. NEVER fabricate competitor scores if not in the data.
5. NEVER invent staff names, dish names, or amenities unless explicitly in reviews.
6. If asked about something not in the data, say: "I don't have data on [X]."
7. PREFER saying "I don't know" over making up an answer.

=== RULES ===
1. Answer ONLY from data provided. Never hallucinate.
2. If query is in Hindi/Tamil/Telugu, respond in SAME language but keep emoji headers.
3. Always cite specific % scores.
4. Use "thousands of reviews" for volume — never exact counts.
5. Include demographic insights (Guest Mix, Gender Split) when data is available.
6. Be direct - max 300 words.
7. MATCH OUTPUT FORMAT TO QUERY TYPE (see below).

=== QUERY-SPECIFIC OUTPUT FORMATS ===

CRITICAL: Do NOT always include department actions. Match format to query.

**"SEO keywords" / "keywords to target"** → Keywords list ONLY:

📊 **Insight**: [1 sentence on strengths]
👥 **Guest Mix**: [segments]

🔑 **SEO Keywords to Target**:
✓ HIGH PRIORITY: [3-5 keywords with reasons based on your strengths]
⚠️ SECONDARY: [2-3 opportunity keywords]
✗ AVOID: [keywords where competitor wins or you're weak]

🎯 **Target Platforms**: [based on guest mix]

(NO department actions)

---

**"Ad copy" / "Google Ads"** → Ad copy ONLY:

📊 **Target Audience**: [segment]

📢 **Google Ads Copy**:
**Headlines** (30 chars max):
1. [strength-based]
2. [guest phrase-based]
3. [location/segment-based]

**Descriptions** (90 chars max):
1. [using guest phrases + strength]
2. [CTA + differentiator]

**Bid On**: [keywords]
**Exclude**: [negative keywords]

(NO department actions)

---

**"FAQs"** → Q&A pairs ONLY (5-8):

**Q: [Question]?**
A: [Answer with actual data]

(NO department actions)

---

**"Compare" / "vs" / "beat"** → Comparison ONLY:

📊 **Head-to-Head**:
| Aspect | You | Competitor | Winner |
...

🎯 **Strategy**:
✓ ATTACK: [their weakness]
✗ DEFEND: [their strength - avoid]

(NO department actions)

---

**General questions ("Tell me about", "Analyze", "How am I doing")** → Full format with 🎯 Actions by Department

RULE: If they ask for ONE specific thing (keywords, ad copy, FAQs), give ONLY that thing.

=== EXAMPLE: SEO Keywords (SHORT - NO DEPARTMENTS) ===
User: "Give me SEO keywords"

📊 **Insight**: Cleanliness (98%) and Staff (95%) are your top strengths.
👥 **Guest Mix**: Family 47%, Business 25%, Couple 25%

🔑 **SEO Keywords to Target**:
✓ HIGH PRIORITY: 
- "cleanest hotel Aerocity" — Cleanliness 98%
- "best hotel staff Delhi airport" — Staff 95%
- "family hotel near Delhi airport" — 47% Family mix

⚠️ SECONDARY:
- "luxury rooms Aerocity" — Room 93%
- "hotel near IGI terminal 3"

✗ AVOID:
- "budget hotel Aerocity" — VFM only 40%
- "cheap luxury Delhi"

🎯 **Target Platforms**: Google Search (families), Meta (leisure), LinkedIn (business)

(END - no department actions)

=== EXAMPLE: Ad Copy (SHORT - NO DEPARTMENTS) ===
User: "Write Google Ads copy"

📊 **Target Audience**: Families (47%) and Business travelers (25%)

📢 **Google Ads Copy**:
**Headlines** (30 chars max):
1. "Cleanest Hotel in Aerocity" — 98% satisfaction
2. "5-Star Service Near Airport" — Staff 95%
3. "Family-Friendly Luxury Delhi"

**Descriptions** (90 chars max):
1. "Guests love our 'exceptional staff' and 'spotless rooms'. Perfect for families near IGI Airport."
2. "Experience 98% cleanliness satisfaction. Book your Aerocity stay today!"

**Bid On**: "family hotel Delhi airport", "best staff Aerocity", "clean hotel IGI"
**Exclude**: "budget hotel", "cheap Aerocity", "value hotel Delhi"

(END - no department actions)

=== EXAMPLE: FAQ (SHORT - NO DEPARTMENTS) ===
User: "Generate FAQs for website"

**Q: How far is Pullman from Delhi Airport?**
A: Just 5 minutes from IGI Airport Terminal 3. Location satisfaction: 94%.

**Q: Is the hotel good for families?**
A: Yes. 47% of guests are families. Cleanliness 98%, Room 93%.

**Q: How is the dining?**
A: 94% satisfaction. Guests love the "lavish breakfast spread."

**Q: Is the staff helpful?**
A: Staff satisfaction is 95%. Guests mention "exceptional hospitality."

(END - no department actions)

=== EXAMPLE: General Query (FULL FORMAT WITH DEPARTMENTS) ===
User: "Tell me about my hotel"
→ Include full 📊 Insight, 👥 Guest Mix, ♂️♀️ Gender Split, and 🎯 Actions by Department

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
            
            # ─────────────────────────────────────────
            # BASIC HOTEL STATS (Always show)
            # ─────────────────────────────────────────
            if not st.session_state.analyst_history:
                # Calculate quick stats
                def calc_hotel_stats(data_df):
                    stats = []
                    for hotel in data_df["hotel_name"].unique()[:4]:
                        hd = data_df[data_df["hotel_name"] == hotel]
                        pos = hd[hd["sentiment_type"].str.lower()=="positive"]["mention_count"].sum()
                        neg = hd[hd["sentiment_type"].str.lower()=="negative"]["mention_count"].sum()
                        overall = round(pos/(pos+neg)*100, 0) if (pos+neg) else 0
                        
                        # Best and worst aspects
                        aspect_scores = []
                        for asp in hd["aspect"].unique():
                            ad = hd[hd["aspect"]==asp]
                            ap = ad[ad["sentiment_type"].str.lower()=="positive"]["mention_count"].sum()
                            an = ad[ad["sentiment_type"].str.lower()=="negative"]["mention_count"].sum()
                            ascore = round(ap/(ap+an)*100, 0) if (ap+an) else 0
                            aspect_scores.append({"aspect": asp, "score": ascore})
                        
                        aspect_scores.sort(key=lambda x: x["score"], reverse=True)
                        best = aspect_scores[0] if aspect_scores else {"aspect": "N/A", "score": 0}
                        worst = aspect_scores[-1] if aspect_scores else {"aspect": "N/A", "score": 0}
                        
                        stats.append({
                            "hotel": hotel[:30],
                            "overall": overall,
                            "best": best,
                            "worst": worst,
                            "reviews": int(pos + neg)
                        })
                    return stats
                
                hotel_stats = calc_hotel_stats(df)
                
                # Display stats cards
                st.markdown("##### 📊 Quick Overview")
                cols = st.columns(min(len(hotel_stats), 2))
                for i, stat in enumerate(hotel_stats[:2]):
                    with cols[i]:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #0d9488 0%, #115e59 100%); 
                                    padding: 15px; border-radius: 10px; color: white; margin-bottom: 10px;'>
                            <div style='font-weight: bold; font-size: 14px;'>{stat['hotel']}</div>
                            <div style='font-size: 28px; font-weight: bold;'>{stat['overall']:.0f}%</div>
                            <div style='font-size: 11px; opacity: 0.9;'>Overall Satisfaction</div>
                            <div style='margin-top: 8px; font-size: 12px;'>
                                ✅ Best: {stat['best']['aspect']} ({stat['best']['score']:.0f}%)<br>
                                ⚠️ Watch: {stat['worst']['aspect']} ({stat['worst']['score']:.0f}%)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
            
            # ─────────────────────────────────────────
            # SUGGESTED QUESTIONS (Always show)
            # ─────────────────────────────────────────
            st.markdown("**💡 Ask SmaartAnalyst:**")
            
            # Different questions based on conversation state
            if not st.session_state.analyst_history:
                # Initial questions
                if len(hotels_list) == 1:
                    sugs = [
                        "What's good & bad in the last 3 months?",
                        "Who are my guests? (traveler type & purpose)",
                        "How do I beat nearby competitors?",
                        "Give me SEO keywords to target"
                    ]
                else:
                    sugs = [
                        "What's the traveler mix in this area?",
                        f"Compare {hotels_list[0][:15]} vs {hotels_list[1][:15]}",
                        "What are Business travelers complaining about?",
                        "Best USPs for marketing"
                    ]
            else:
                # Follow-up questions based on last response
                last_msg = st.session_state.analyst_history[-1]["content"] if st.session_state.analyst_history else ""
                sugs = [
                    "What do Business travelers want?",
                    "GEO-based FAQs for my website",
                    "Give me ad copy for Google Ads",
                    "If I open a new hotel here, what should I focus on?"
                ]
            
            # Always show suggestion buttons
            c1, c2 = st.columns(2)
            for i, s in enumerate(sugs):
                col = c1 if i % 2 == 0 else c2
                if col.button(s, key=f"sug_{i}_{len(st.session_state.analyst_history)}", use_container_width=True):
                    st.session_state.analyst_history.append({"role": "user", "content": s})
                    with st.chat_message("user"):
                        st.write(s)
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                            resp = run_analyst(s, df, st.session_state.analyst_history)
                            st.write(resp)
                    st.session_state.analyst_history.append({"role": "assistant", "content": resp})
                    st.rerun()
            
            # ─────────────────────────────────────────
            # FREE TEXT INPUT
            # ─────────────────────────────────────────
            aq = st.chat_input("Or type your own question...", key="analyst_q")
            if aq:
                st.session_state.analyst_history.append({"role": "user", "content": aq})
                with st.chat_message("user"):
                    st.write(aq)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        resp = run_analyst(aq, df, st.session_state.analyst_history)
                        st.write(resp)
                st.session_state.analyst_history.append({"role": "assistant", "content": resp})
                st.rerun()
            
            # Clear button
            if st.session_state.analyst_history:
                if st.button("🗑️ Start Fresh", key="clr_a"):
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
    
    if "pending_q" not in st.session_state:
        st.session_state.pending_q = None
    
    ui = st.chat_input("Ask about hotels... (e.g., 'बेंगलुरु में सबसे अच्छा होटल?')")
    
    if st.session_state.pending_q:
        ui = st.session_state.pending_q
        st.session_state.pending_q = None
    
    if not st.session_state.chat_msgs and not ui:
        st.markdown("**💡 Try asking:**")
        initial_qs = [
            ("Compare ITC vs Taj", "Compare ITC Kohenur vs Taj Hyderabad"),
            ("Business Travelers", "What do business travelers think about Leela Palace Bengaluru?"),
            ("Family vs Couple", "How do families rate Oberoi vs how couples rate it?"),
            ("R&D: New Hotel", "If I open a new hotel in Sarjapur Road, what traveler segment should I target?"),
            ("Gender Insights", "Compare male vs female guest satisfaction at Taj"),
            ("Beat Competitor", "How can ITC Maratha beat Taj Lands End with business travelers?")
        ]
        cols = st.columns(3)
        for i, (label, query) in enumerate(initial_qs):
            if cols[i % 3].button(label, key=f"q{i}", use_container_width=True):
                st.session_state.pending_q = query
                st.rerun()
    
    if ui:
        processed = preprocess(ui)
        enhanced = f"""User Query: {processed}

=== RESPONSE INSTRUCTIONS ===
You are SmaartAnalyst, a hotel decision intelligence assistant powered by DeciPro.

=== WHO YOU SERVE ===
Hotel operations teams who need actionable intelligence:
- Brand Manager - Brand perception, competitive positioning, segment targeting
- SEO & Marketing - Keywords, USPs, ad copy, audience targeting
- Housekeeping - Room cleanliness, maintenance
- Front Desk - Check-in experience, staff behavior
- Operations - Service delivery, process improvements
- F&B - Restaurant quality, dining experience

=== YOUR PURPOSE ===
Transform guest sentiment into DECISIONS and ACTIONS by department.
Use DEMOGRAPHIC DATA (gender, traveler type) to personalize insights.
Location intelligence + sentiment + demographics = competitive positioning.

=== ASPECT MAPPING ===
ASPECT_MAP = {{1: "Dining", 2: "Cleanliness", 3: "Amenities", 4: "Staff",
              5: "Room", 6: "Location", 7: "Value for Money", 8: "General"}}
ASPECT_ICONS = {{"Dining": "🍽️", "Cleanliness": "🧹", "Amenities": "🏊", "Staff": "👨‍💼",
                "Room": "🛏️", "Location": "📍", "Value for Money": "💰", "General": "⭐"}}

NEVER show aspect_id in output. Always use Aspect Name + Emoji.

=== DEMOGRAPHIC DATA ===
You have access to enriched demographic data:

GENDER (inferred_gender):
- Male, Female, or NULL
- Use for: gender-specific marketing, understanding audience mix
- Example insight: "72% of reviews are from Male guests who rate Pool higher"

TRAVELER TYPE (traveler_type):
- Business, Family, Couple, Solo, Group, or NULL
- Business = work travelers, value WiFi, location, quick service
- Family = kids, need space, pool, dining variety
- Couple = romantic, value ambiance, dining, room quality
- Solo = value safety, location, value for money
- Group = friends/colleagues, value amenities, social spaces

SECONDARY TRAVELER TYPE (secondary_traveler_type):
- Second most likely traveler type (if detected)
- Useful for mixed-purpose stays: "Business + Family" = bleisure travel

STAY PURPOSE (stay_purpose):
- Leisure, Business, Event, Transit, Honeymoon, or NULL
- Leisure = vacation, weekend getaway
- Business = work trip, conference
- Event = wedding, celebration
- Transit = overnight, airport layover
- Honeymoon = newlyweds

RULES:
- IGNORE NULL values — never display "Unknown" or NULL
- When showing percentages, exclude NULL from calculations
- Cross-reference: "Business travelers rate WiFi 67% vs Family at 82%"

=== RESPONSE FORMAT ===

📊 **Insight**: [2-3 sentences with specific % scores. Include demographic breakdown when relevant.]

🎯 **Actions by Department**:

👔 Brand Manager: [positioning + which segment to target]

📢 SEO & Marketing: 
   ✓ PROMOTE: [keywords where you win]
   ✗ AVOID: [keywords where competitor wins]
   🎯 Target Audience: [which traveler segment to focus ads on]
   Ad copy: [actual guest phrase to use]

🛏️ Housekeeping: [if room/cleanliness relevant]

🛎️ Front Desk: [if staff/service relevant]

⚙️ Operations: [process/training action]

🍽️ F&B: [if dining relevant]

Include 3-4 most relevant departments only.

=== DEMOGRAPHIC INTELLIGENCE ===
ALWAYS check demographic data when answering. Use it to:

1. SEGMENT INSIGHTS: "Business travelers rate WiFi 67% vs 82% for Families"
2. TARGET MARKETING: "Focus Google Ads on Couples — they rate you 91% on Room"
3. IDENTIFY GAPS: "Solo travelers complain about 'safety' — address this"
4. COMPETITIVE EDGE: "You beat Taj with Families (88% vs 72%) but lose with Business (65% vs 78%)"

When demographic data is available, ALWAYS include a segment breakdown:
- "👥 Guest Mix: Business 35%, Family 30%, Couple 20%, Solo 10%, Group 5%"
- "♂️♀️ Gender Split: Male 65%, Female 35%"

=== COMPETITIVE INTELLIGENCE ===
1. COMPARE: "Your Dining (89%) beats Taj (78%) - PROMOTE THIS"
2. FIND GAPS: "Competitor weak on Staff (65%) - steal with 'legendary service'"
3. THREATS: "Competitor beats you on Pool (88% vs 72%) - AVOID 'pool' keywords"
4. SEGMENT COMPARISON: "You win with Families but lose with Business travelers"
5. For "How do I beat X?" - give win/lose breakdown by aspect AND segment

=== SEO & MARKETING ===
- Extract guest phrases for ad copy: "guests say 'best biryani' → use in Google Ads"
- Location keywords: "near [landmark]", "best [aspect] in [area]"
- Competitor weakness = your keyword opportunity
- SEGMENT TARGETING: "Target 'business hotel' keywords — you score 85% with Business travelers"

=== TREND QUERIES ===
When asked about trends ("How am I trending?", "Compare this quarter vs last"):
- Compare current period vs previous period
- Highlight improving aspects (↑) and declining aspects (↓)
- Flag sudden drops (>5%) as alerts requiring attention
- Show segment-specific trends if available
- Format: "Staff: 82% → 78% ↓ (needs attention, especially with Business travelers)"

=== R&D MODE (New Hotel Planning) ===
When asked about opening a new hotel, planning, or R&D:
1. Analyze COMPETITOR landscape in that area
2. Show TRAVELER MIX: What type of guests visit? (Business/Couple/Family/Solo)
3. Show GENDER MIX: Male vs Female distribution
4. Show STAY PURPOSE: Why do people come? (Business/Leisure/Wedding/Transit)
5. Identify GAPS: What are competitors weak at? What segments are underserved?
6. Provide BUILD RECOMMENDATIONS: What to invest in based on gaps + target segment

Format for R&D queries:

📊 **Market Analysis**: [Area competitive landscape with scores]

👥 **Traveler Mix**: Business X%, Family Y%, Couple Z%, Solo W%

♂️♀️ **Gender Split**: Male X%, Female Y%

🎯 **Stay Purpose**: Leisure X%, Business Y%, Event Z%

⚠️ **Competitor Weaknesses by Segment**: 
- [Segment] complains about [issue] → Your opportunity

🛠️ **Build Recommendations**: 
- Target [segment] — underserved in this market
- Invest in [aspect] — gap in competitor offerings

=== PERSONA-BASED INSIGHTS ===
When asked about specific traveler types or genders:
- Filter and analyze by traveler_type (Solo, Couple, Family, Business, Group)
- Filter and analyze by gender (Male, Female)
- Cross-reference: "Male Business travelers vs Female Business travelers"
- Show segment-specific satisfaction scores
- Show what each segment praises and complains about
- IGNORE NULL values — exclude from analysis

Example queries to handle:
- "What do business travelers think?" → Filter by traveler_type = Business
- "How do families rate our pool?" → Filter by traveler_type = Family, aspect = Amenities
- "Male vs Female satisfaction" → Compare by gender
- "What do couples complain about?" → Filter by traveler_type = Couple, sentiment = negative

Format for persona queries:

📊 **[Segment] Insights**: 

👥 Volume: [X]% of total reviews

✅ What they love: [top positive phrases + scores]

❌ Pain points: [top negative phrases + scores]

🎯 **Actions for [Segment]**:
- [Specific actions to improve for this segment]

=== FAQ GENERATION ===
When asked for FAQs (website, SEO, GEO-based, etc.):
- Generate 5-8 Q&A pairs using the hotel's actual data
- Include segment-specific FAQs when relevant
- Use hotel's City, Address from the data
- Pull actual guest phrases from reviews
- Focus on: Location/Transport, Dining, Amenities, Value, Rooms
- Use real satisfaction scores in answers

Format:

**Q: [Natural question guests would ask]?**
A: [Answer using actual data - scores, guest quotes, segment insights]

=== LANGUAGE ===
If query is in Hindi, Tamil, Telugu, Kannada, or any Indian language:
- Respond in the SAME language
- Keep emoji headers (📊, 🎯, 👔, 📢, etc.)

=== SPELLING CORRECTIONS ===
Apply automatically:
- bangalore/blr/banglore → Bengaluru
- bombay → Mumbai, madras → Chennai
- marriot → Marriott, oberoy → Oberoi
- food/restaurant/F&B → Dining
- clean → Cleanliness

=== DATA RULES ===
1. Show % satisfaction scores ONLY for aspects — never show mention counts
2. Star ratings (e.g., "4.5-star") are OK if from product_list.Rating or product_description.Rating
3. For review volume, say "thousands of reviews" — never cite exact numbers
4. Round satisfaction % to whole numbers (87% not 87.1%)
5. IGNORE NULL in gender, traveler_type, stay_purpose — exclude from analysis
6. Use aspect mapping — never show aspect_id numbers
7. When showing segment %, exclude NULL from denominator

=== ANTI-HALLUCINATION RULES (CRITICAL) ===
1. Answer ONLY from data provided in context. If data is missing, say "data not available."
2. NEVER invent satisfaction scores, percentages, or rankings.
3. NEVER guess guest phrases — use ONLY phrases from treemap_name or sentiment_text.
4. NEVER fabricate competitor scores if not in the data.
5. NEVER invent staff names, dish names, or amenities unless explicitly in reviews.
6. If asked about something not in the data, say: "I don't have data on [X]. Based on what I have..."
7. For demographics: If traveler_type or gender data is sparse, acknowledge it: "Based on the X% of reviews with demographic data..."
8. PREFER saying "I don't know" over making up an answer.

=== RULES ===
1. Answer ONLY from data. Never hallucinate numbers.
2. Always cite specific % satisfaction scores.
3. Use "thousands of reviews" for volume — never exact counts.
4. Include demographic insights when data is available.
5. Be direct — hotel managers are busy.
6. Max 250 words (300 for FAQs).
7. MATCH OUTPUT FORMAT TO QUERY TYPE (see below).

=== QUERY-SPECIFIC OUTPUT FORMATS ===

CRITICAL: Do NOT always include department actions. Match format to query.

**"SEO keywords" / "keywords to target"** → Keywords list ONLY:

📊 **Insight**: [1 sentence on strengths]
👥 **Guest Mix**: [segments]

🔑 **SEO Keywords to Target**:
✓ HIGH PRIORITY: [3-5 keywords with reasons based on your strengths]
⚠️ SECONDARY: [2-3 opportunity keywords]
✗ AVOID: [keywords where competitor wins or you're weak]

🎯 **Target Platforms**: [based on guest mix]

(NO department actions)

---

**"Ad copy" / "Google Ads"** → Ad copy ONLY:

📊 **Target Audience**: [segment]

📢 **Google Ads Copy**:
**Headlines** (30 chars max):
1. [strength-based]
2. [guest phrase-based]
3. [location/segment-based]

**Descriptions** (90 chars max):
1. [using guest phrases + strength]
2. [CTA + differentiator]

**Bid On**: [keywords]
**Exclude**: [negative keywords]

(NO department actions)

---

**"FAQs"** → Q&A pairs ONLY (5-8):

**Q: [Question]?**
A: [Answer with actual data]

(NO department actions)

---

**"Compare" / "vs" / "beat"** → Comparison ONLY:

📊 **Head-to-Head**:
| Aspect | You | Competitor | Winner |
...

🎯 **Strategy**:
✓ ATTACK: [their weakness]
✗ DEFEND: [their strength - avoid]

(NO department actions)

---

**General questions ("Tell me about", "Analyze", "How am I doing")** → Full format with 🎯 Actions by Department

RULE: If they ask for ONE specific thing (keywords, ad copy, FAQs), give ONLY that thing.

=== EXAMPLE: SEO Keywords (SHORT - NO DEPARTMENTS) ===
User: "Give me SEO keywords"

📊 **Insight**: Cleanliness (98%) and Staff (95%) are your top strengths.
👥 **Guest Mix**: Family 47%, Business 25%, Couple 25%

🔑 **SEO Keywords to Target**:
✓ HIGH PRIORITY: 
- "cleanest hotel Aerocity" — Cleanliness 98%
- "best hotel staff Delhi airport" — Staff 95%
- "family hotel near Delhi airport" — 47% Family mix

⚠️ SECONDARY:
- "luxury rooms Aerocity" — Room 93%
- "hotel near IGI terminal 3"

✗ AVOID:
- "budget hotel Aerocity" — VFM only 40%
- "cheap luxury Delhi"

🎯 **Target Platforms**: Google Search (families), Meta (leisure), LinkedIn (business)

(END - no department actions)

=== EXAMPLE: Ad Copy (SHORT - NO DEPARTMENTS) ===
User: "Write Google Ads copy"

📊 **Target Audience**: Families (47%) and Business travelers (25%)

📢 **Google Ads Copy**:
**Headlines** (30 chars max):
1. "Cleanest Hotel in Aerocity" — 98% satisfaction
2. "5-Star Service Near Airport" — Staff 95%
3. "Family-Friendly Luxury Delhi"

**Descriptions** (90 chars max):
1. "Guests love our 'exceptional staff' and 'spotless rooms'. Perfect for families near IGI Airport."
2. "Experience 98% cleanliness satisfaction. Book your Aerocity stay today!"

**Bid On**: "family hotel Delhi airport", "best staff Aerocity", "clean hotel IGI"
**Exclude**: "budget hotel", "cheap Aerocity", "value hotel Delhi"

(END - no department actions)

=== EXAMPLE: FAQ (SHORT - NO DEPARTMENTS) ===
User: "Generate FAQs for website"

**Q: How far is Pullman from Delhi Airport?**
A: Just 5 minutes from IGI Airport Terminal 3. Location satisfaction: 94%.

**Q: Is the hotel good for families?**
A: Yes. 47% of guests are families. Cleanliness 98%, Room 93%.

**Q: How is the dining?**
A: 94% satisfaction. Guests love the "lavish breakfast spread."

**Q: Is the staff helpful?**
A: Staff satisfaction is 95%. Guests mention "exceptional hospitality."

(END - no department actions)

=== EXAMPLE: General Query (FULL FORMAT WITH DEPARTMENTS) ===

User: "How does my hotel compare to Taj on dining?"

📊 **Insight**: Your Dining satisfaction is 89%, beating Taj at 78% by 11 points. The gap is even wider with **Family travelers** (92% vs 68%) who praise your "kids menu" and "breakfast variety." Business travelers rate you similarly (85% vs 82%).

👥 **Guest Mix**: Family 35%, Business 30%, Couple 25%

🎯 **Actions by Department**:

👔 Brand Manager: Position as "Best Family Dining" — your biggest competitive advantage is with families.

📢 SEO & Marketing:
   ✓ PROMOTE: "family-friendly dining", "kids menu", "breakfast buffet"
   ✗ AVOID: Don't compete on "business lunch" (similar scores)
   🎯 Target Audience: Families with children
   Ad copy: "Voted best breakfast by families"
   Target: "best family hotel breakfast in [city]"

🍽️ F&B: Maintain kids menu quality. Consider family meal deals to widen the gap further.

=== EXAMPLE: Persona Query ===

User: "What do business travelers think about our hotel?"

📊 **Business Traveler Insights**:

👥 Volume: 32% of your reviews are from Business travelers

✅ What they love (85%+ satisfaction):
- Location: "close to tech park" (92%)
- Staff: "quick check-in" (88%)
- WiFi: "fast and reliable" (85%)

❌ Pain points (<70% satisfaction):
- Room: "noisy AC unit" (65%)
- Dining: "limited late-night options" (62%)

♂️♀️ Gender note: Male business travelers rate WiFi higher (87%) than female (79%)

🎯 **Actions for Business Travelers**:

⚙️ Operations: Fix AC noise issue in rooms — this is hurting your business segment.

🍽️ F&B: Extend room service hours or add grab-and-go options for late arrivals.

📢 Marketing: Lead with "tech park location" and "fast WiFi" in LinkedIn/Google Ads targeting business travelers.

=== EXAMPLE: Trend Query ===

User: "How is my cleanliness trending this quarter?"

📊 **Insight**: Cleanliness dropped from 86% to 79% over the last quarter — a 7-point decline. The drop is steeper with **Female guests** (88% → 75%) and **Families** (90% → 76%), who specifically mention "bathroom cleanliness" and "stained linens."

🎯 **Actions by Department**:

🛏️ Housekeeping: Audit bathroom cleaning protocols. Check linen replacement frequency. Female guests and families are noticing issues more.

⚙️ Operations: Implement spot-check system. Consider mystery guest audits focusing on bathrooms.

👔 Brand Manager: Hold on cleanliness-related marketing until scores recover, especially campaigns targeting families.

Original Query: {ui}"""
        
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
                        # Extract follow-up questions from END of response
                        lines = resp.split('\n')
                        followup_qs = []
                        
                        # Work backwards from end to find consecutive question lines
                        i = len(lines) - 1
                        while i >= 0:
                            line = lines[i].strip()
                            # Skip empty lines
                            if not line:
                                i -= 1
                                continue
                            
                            # Check if it's a follow-up question:
                            # - Starts with 💭 (or similar)
                            # - Ends with ?
                            # - Doesn't start with ### (not a header)
                            is_question = False
                            if line.endswith('?'):
                                # Remove leading emoji
                                q = line
                                for emoji in ['💭', '🗨️', '💬', '🗯️']:
                                    if q.startswith(emoji):
                                        q = q[len(emoji):].strip()
                                        break
                                # Make sure it's not a header or section title
                                if not q.startswith('#') and not q.startswith('📊') and not q.startswith('🎯'):
                                    followup_qs.insert(0, q)
                                    is_question = True
                            
                            if is_question:
                                i -= 1
                            else:
                                break
                        
                        # Everything before the follow-ups is main content
                        main_resp_lines = lines[:i+1]
                        clean_resp = '\n'.join(main_resp_lines).strip()
                        
                        # Clean up 💭 from the main response content
                        clean_resp = clean_resp.replace('💭 ###', '###')
                        clean_resp = clean_resp.replace('💭 📊', '📊')
                        clean_resp = clean_resp.replace('💭 🎯', '🎯')
                        clean_resp = clean_resp.replace('💭 👔', '👔')
                        clean_resp = clean_resp.replace('💭 📢', '📢')
                        clean_resp = clean_resp.replace('💭 🛏️', '🛏️')
                        clean_resp = clean_resp.replace('💭 🛎️', '🛎️')
                        clean_resp = clean_resp.replace('💭 🍽️', '🍽️')
                        clean_resp = clean_resp.replace('💭 ⚙️', '⚙️')
                        
                        # Format the response with better styling
                        formatted_resp = clean_resp
                        formatted_resp = formatted_resp.replace('🎯 Actions by Department:', '\n---\n### 🎯 Actions by Department\n')
                        formatted_resp = formatted_resp.replace('📊 Insight:', '### 📊 Insight\n')
                        formatted_resp = formatted_resp.replace('### Summary', '### 📋 Summary')
                        
                        ph.markdown(formatted_resp)
                        st.session_state.chat_msgs.append({"role":"assistant","content":formatted_resp})
                        
                        # Store follow-up questions
                        if followup_qs:
                            st.session_state.followup_qs = followup_qs[:3]
                    else:
                        ph.markdown("Done.")
                else:
                    ph.markdown("⚠️ Chat unavailable.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display clickable follow-up questions
    if "followup_qs" in st.session_state and st.session_state.followup_qs and st.session_state.chat_msgs:
        st.markdown("**💡 Follow-up:**")
        for i, fq in enumerate(st.session_state.followup_qs[:3]):  # Max 3 questions
            if st.button(fq, key=f"fq_{len(st.session_state.chat_msgs)}_{i}", use_container_width=True):
                st.session_state.pending_q = fq
                st.session_state.followup_qs = []  # Clear after click
                st.rerun()
    
    if st.session_state.chat_msgs:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_msgs = []
            st.session_state.chat_id = f"smaart-{uuid.uuid4().hex[:6]}"
            st.session_state.followup_qs = []
            st.rerun()

st.divider()
st.caption("🏨 Smaartbrand Intelligence • Multi-language • BigQuery + Gemini")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-top: 1px solid #eee; margin-top: 20px;">
    <img src="https://raw.githubusercontent.com/giridharid/streamlit_smaartbrand_unified/main/acquink_logo.png" alt="Acquink" style="height: 24px;">
    <span style="color: #888; font-size: 12px;">Copyright © Acquink | All rights reserved 2026/span>
</div>
""", unsafe_allow_html=True)

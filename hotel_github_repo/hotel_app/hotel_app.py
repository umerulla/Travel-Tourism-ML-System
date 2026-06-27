"""Streamlit UI for the Hotel Recommendation System.
Loads pre-trained artifacts via recommend.py (NO retraining at launch)."""
import os, sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recommend as rec

st.set_page_config(page_title="Hotel Recommender", page_icon="🏨", layout="centered")
st.title("🏨 Smart Hotel Recommendation System")
st.caption(f"Hybrid recommender (SVD collaborative filtering + TF-IDF content). "
           f"Model RMSE: {rec.RMSE:.4f}")

max_user = max(rec.USER_SEEN) if rec.USER_SEEN else 0
col1, col2 = st.columns(2)
with col1:
    user_id = st.number_input("User ID", min_value=0, max_value=int(max_user), value=0, step=1)
with col2:
    top_n = st.slider("Number of recommendations", 1, 9, 5)

tab1, tab2, tab3 = st.tabs(["Collaborative", "Hybrid", "Similar hotels"])

with tab1:
    if st.button("Recommend (CF)", key="cf"):
        recs = rec.collaborative_recommend(user_id, top_n)
        st.table(pd.DataFrame(recs, columns=["Hotel", "Predicted rating"]))

with tab2:
    if st.button("Recommend (Hybrid)", key="hy"):
        recs = rec.hybrid_recommend(user_id, top_n)
        st.table(pd.DataFrame({"Hotel": recs}))

with tab3:
    hotel = st.selectbox("Pick a hotel", rec.ALL_HOTELS)
    if st.button("Find similar", key="sim"):
        st.table(pd.DataFrame({"Similar hotel": rec.content_similar(hotel, top_n)}))

with st.expander("Dataset overview"):
    st.dataframe(pd.DataFrame(rec.HOTEL_INFO))

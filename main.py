import streamlit as st

routes = [
    st.Page("routes/home.py",title="Home"),
    st.Page("routes/chat.py",title="Chat"),
    st.Page("routes/scrape_data.py", title="Scrape data")
]

pg = st.navigation(routes)

pg.run()
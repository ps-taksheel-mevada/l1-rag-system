import streamlit as st
from commands.wiki_search import wikipedia_search, scrape_pages
from commands.upload_to_db import UploadDocs

st.title("Scrape data")

query = st.text_input("Enter keyword to search")
limit = st.number_input("Enter number of pages", step=1, min_value=0, max_value=20)

if "scraped_data" not in st.session_state:
    st.session_state.scraped_data = None

if query:
    if st.button("Scrape"):
        data = wikipedia_search(query, limit)
        if "error" in data.keys():
            st.text(f"There is no data found related {query}")
            st.session_state.scraped_data = None
        else:
            scraped = scrape_pages(data)
            st.dataframe(scraped)
            st.session_state.scraped_data = scraped

if st.session_state.scraped_data is not None:
    if st.button("Add to DB"):
        print("clicked")
        cmd = UploadDocs("chroma_db", "rag_data","mxbai-embed-large:latest")
        cmd(st.session_state.scraped_data)
        st.success("Data added to Chroma DB!")
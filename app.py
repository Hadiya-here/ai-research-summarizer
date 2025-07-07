import streamlit as st
import fitz  # PyMuPDF
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Page config
st.set_page_config(page_title="AI Research Paper Summarizer", layout="centered")
st.title("📄 AI Research Paper Summarizer")

# Check if API key exists
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OpenAI API key not found in Streamlit secrets.")
    st.stop()

api_key = st.secrets["OPENAI_API_KEY"]

# File uploader
uploaded_file = st.file_uploader("Upload a PDF research paper", type="pdf")

if uploaded_file is not None:
    st.info("Reading the PDF file...")

    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = "\n".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        st.stop()

    if len(text.strip()) == 0:
        st.error("Couldn't extract any text from the PDF. Please try a different file.")
    else:
        raw_docs = [Document(page_content=text)]

        st.info("Summarizing with OpenAI (GPT-3.5)...")

        try:
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",  # safer default than gpt-4
                openai_api_key=api_key
            )
            chain = load_summarize_chain(llm, chain_type="stuff")
            summary = chain.run(raw_docs)

            st.success("Done! Here's your summary:")
            st.text_area("📚 Summary", summary, height=300)

            st.download_button("💾 Download Summary as TXT", summary, file_name="summary.txt")

        except Exception as e:
            st.error(f"Error while summarizing: {e}")

import streamlit as st
from orchestrator import SmartEmailCrew

st.set_page_config(page_title="Smart Email Assistant", layout="centered")
crew = SmartEmailCrew()

st.title("📧 Smart Email Assistant - CrewAI Powered")
email = st.text_area("Enter an email to process:", height=200)

if st.button("Run Smart Assistant"):
    if not email.strip():
        st.warning("Please enter an email.")
    else:
        with st.spinner("Analyzing..."):
            result = crew.process(email)

        st.success(f"✅ Agent Used: {result['agent']}")
        st.markdown("### 📤 Classification:")
        st.json(result["input"])

        st.markdown("### 📬 Output:")
        st.json(result["output"])

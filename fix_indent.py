import re

with open('app_v2_isro.py', 'r') as f:
    content = f.read()

# Instead of complex regex, let's just prepend from textwrap import dedent
# and replace st.markdown(""" with st.markdown(dedent("""
if "from textwrap import dedent" not in content:
    content = content.replace('import streamlit as st', 'import streamlit as st\nfrom textwrap import dedent')

content = content.replace('st.markdown("""', 'st.markdown(dedent("""')
content = content.replace('""", unsafe_allow_html=True)', '"""), unsafe_allow_html=True)')

with open('app_v2_isro.py', 'w') as f:
    f.write(content)

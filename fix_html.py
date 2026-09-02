with open('app_v2_isro.py', 'r') as f:
    content = f.read()

# Replace st.markdown(dedent(""" with st.html("""
content = content.replace('st.markdown(dedent("""', 'st.html("""')
content = content.replace('"""), unsafe_allow_html=True)', '""")')

# Also replace the one-liners
content = content.replace("st.markdown('<div", "st.html('<div")
content = content.replace("</div>', unsafe_allow_html=True)", "</div>')")

with open('app_v2_isro.py', 'w') as f:
    f.write(content)

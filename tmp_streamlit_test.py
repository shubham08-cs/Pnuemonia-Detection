import os
print("NAME="+__name__)
print("STREAMLIT_RUN_MAIN="+str(os.environ.get("STREAMLIT_RUN_MAIN")))
print("STREAMLIT_SERVER_RUN_ON_SAVE="+str(os.environ.get("STREAMLIT_SERVER_RUN_ON_SAVE")))

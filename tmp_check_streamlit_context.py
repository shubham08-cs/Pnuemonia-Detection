import streamlit.runtime.scriptrunner as sr
from pathlib import Path
ctx = sr.get_script_run_ctx()
Path('tmp_check_streamlit_context.log').write_text('CTX=' + repr(ctx) + '
')

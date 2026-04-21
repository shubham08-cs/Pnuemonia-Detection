import streamlit.runtime.scriptrunner as sr
from pathlib import Path
ctx = sr.get_script_run_ctx()
Path('tmp_streamlit_scriptrunner_log.txt').write_text('CTX=' + repr(ctx) + '
')
print('done')

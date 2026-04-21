import os
import sys
from pathlib import Path
Path('tmp_streamlit_log.txt').write_text('ARGV=' + repr(sys.argv) + '
' + 'ENV=' + repr({k:v for k,v in os.environ.items() if 'STREAMLIT' in k.upper()}))
import time
time.sleep(1)

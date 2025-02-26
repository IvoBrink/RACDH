import pandas as pd
import json
import gzip

with gzip.open("/home/ibrink/RACDH/RACDH/3DLNews-main/1-Google/1-Newspapers/state/AK/newspaper_articles_AK_2024.jsonl.gz", "rt") as f:
    content = f.read()
    print(content)

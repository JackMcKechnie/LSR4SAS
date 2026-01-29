import pandas as pd
import json
from tqdm import tqdm
from pyterrier_t5 import MonoT5ReRanker
import argparse
from transformers import T5ForConditionalGeneration

parser = argparse.ArgumentParser(description="Process and save score data.")
parser.add_argument("--INPUT_MODEL", type=str, required=True, help="Path to input model file or identifier")
parser.add_argument("--NAME", type=str, required=True, help="Name identifier for this run")
parser.add_argument("--OUTPUT_PATH", type=str, required=True, help="Path to save the output JSON file")
args = parser.parse_args()

INPUT_MODEL = args.INPUT_MODEL
NAME = args.NAME
OUTPUT_PATH = args.OUTPUT_PATH

full_df = pd.read_csv("/nfs/primary/SPLADE/splade_sans/data/training_data_w_qids.csv")
json_file = pd.read_json("hn_splade_sans1_all_sensitive_8_negs_docnos_v2.jsonl", lines = True)

text_lookup = dict(zip(full_df['medline_ui'], full_df['text']))#
query_lookup = dict(zip(full_df['qid'], full_df['best_query']))

create_df = []
for _, row in tqdm(json_file.iterrows(), total = len(json_file), desc = "creating df"):
    qid = row.query
    query = query_lookup[qid]
    pos_docno = row.docno_a
    pos_text = text_lookup[pos_docno]
    negative_docs = row.docno_b
    create_df.append({"qid" : qid, "query" : query, "docno" : pos_docno, "text" : pos_text, "pos" : 1})
    for d in negative_docs:
        negative_docno = int(d)
        negative_text = text_lookup[negative_docno]
        create_df.append({"qid" : qid, "query" : query, "docno" : negative_docno, "text" : negative_text, "pos" : 0})

df = pd.DataFrame(create_df)

monot5 = MonoT5ReRanker(verbose = True, batch_size = 32)
model = T5ForConditionalGeneration.from_pretrained(INPUT_MODEL).cuda()
monot5.model = model

scored = monot5(df)

scored.to_csv(f"{OUTPUT_PATH}{NAME}_scoredf.csv", index = False)

scores = scored.groupby('qid').apply(lambda x: x.set_index('docno')['score'].to_dict()).to_dict()
with open(f"{OUTPUT_PATH}{NAME}_scorefile.json", 'w') as f:
    json.dump(scores, f)
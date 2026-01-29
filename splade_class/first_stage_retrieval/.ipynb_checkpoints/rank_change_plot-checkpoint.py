import argparse
import pandas as pd
import pyterrier as pt
import matplotlib.pyplot as plt
from tqdm import tqdm
if not pt.started():
    pt.init()

parser = argparse.ArgumentParser(description="Script that takes two or three arguments.")

# Define positional arguments
parser.add_argument("--one", help="First argument")
parser.add_argument("--two", help="Second argument")
parser.add_argument("--three", nargs="?", help="Optional third argument")

args = parser.parse_args()

# Access the arguments
print("Argument one:", args.one)
print("Argument two:", args.two)
print("Argument three:", args.three)

# Load data
ohsumed_docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
queries = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/queries.pkl")
qrels = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/qrels.pkl")

# Read in run files
df1 = pt.io.read_results(f"./runs/intermediates/{args.one}")
df2 = pt.io.read_results(f"./runs/intermediates/{args.two}")
df1 = pd.merge(df1, qrels, on=["qid", "docno"], how="left").fillna(0)
df2 = pd.merge(df2, qrels, on=["qid", "docno"], how="left").fillna(0)
df1 = pd.merge(df1, ohsumed_docs, on="docno")
df2 = pd.merge(df2, ohsumed_docs, on="docno")


# Rename ranks early for clarity
df1 = df1.rename(columns={"rank": "rank_1"})
df2 = df2.rename(columns={"rank": "rank_2"})

# Merge with outer join to keep all documents
merged = pd.merge(
    df1[['qid', 'docno', 'rank_1', 'label', 'sensitivity']],
    df2[['qid', 'docno', 'rank_2']],
    on=['qid', 'docno'],
    how='outer'  # <- critical change here
)

if args.three:
    df3 = pt.io.read_results(f"./runs/intermediates/{args.three}")
    df3 = pd.merge(df3, qrels, on=["qid", "docno"], how="left").fillna(0)
    df3 = pd.merge(df3, ohsumed_docs, on="docno")
    df3 = df3.rename(columns={"rank": "rank_3"})

    # Merge third run
    merged = pd.merge(
        merged,
        df3[['qid', 'docno', 'rank_3']],
        on=["qid", "docno"],
        how="outer"  # <- maintain all docs again
    )


# If third run is provided, you can extend plotting logic as needed
if args.three:
    df3 = pt.io.read_results(f"./runs/intermediates/{args.three}")
    df3 = pd.merge(df3, qrels, on=["qid", "docno"], how="left").fillna(0)
    df3 = pd.merge(df3, ohsumed_docs, on="docno")
    merged = pd.merge(merged, df3[['qid', 'docno', 'rank']], on=["qid", "docno"])

merged['label'] = merged['label'].fillna(0)

# Identify which rows are relevant & non-sensitive
merged['highlight'] = (merged['label'] > 0) & (merged['sensitivity'] == 0)

print(merged)

# Plotting
plt.figure(figsize=(10, 6))
for _, row in tqdm(merged.iterrows(), total=len(merged), desc="plotting"):
    color = 'blue' if row['highlight'] else 'gray'
    alpha = 1.0 if row['highlight'] else 0.1

    plt.scatter(0, row['rank_1'], color=color, alpha=alpha, s=10)
    plt.scatter(1, row['rank_2'], color=color, alpha=alpha, s=10)
    if args.three and pd.notna(row['rank_3']):
        plt.scatter(2, row['rank_3'], color=color, alpha=alpha, s=10)
        
    if row['highlight']:
        x_vals = [0, 1]
        y_vals = [row['rank_1'], row['rank_2']]
        if args.three and pd.notna(row['rank_3']):
            x_vals.append(2)
            y_vals.append(row['rank_3'])
        plt.plot(x_vals, y_vals, color=color, alpha=0.7)


xticks = [args.one, args.two]
if args.three:
    xticks.append(args.three)

plt.xticks(range(len(xticks)), xticks)
plt.gca().invert_yaxis()
plt.ylabel("Rank position")
plt.title(f"Rank Changes: {args.three if args.three else args.two}")
plt.tight_layout()

# Save the plot
plot_filename = f"./plots/{args.one.split('/')[-1]}_{args.two.split('/')[-1]}"
if args.three:
    plot_filename += f"_{args.three.split('/')[-1]}"
plot_filename += ".png"  # Add file extension
plt.savefig(plot_filename)
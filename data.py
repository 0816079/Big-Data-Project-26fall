import os
import glob
import re
import html
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split


def find(root, name):
    for data in glob.glob(f"{root}/**/*", recursive = True):
        if  os.path.basename(data).lower() == name.lower:
            return data

def clean(s):
    s =re.sub(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", " ", str(s), flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()

def load_r52(root):
    df = pd.concat([pd.read_csv(find(root, f"ModApte_{s}.csv")).assign(split=s) for s in ["train", "test"]],ignore_index=True)
    topics =  df.topics.fillna("").str.findall(r"[a-z][\w.-]*")
    df = df[topics.str.len() == 1].assign(label=topics.str[0])
    both = set(df.label[df.split == "train"]) & set(df.label[df.split == "test"])
    return df[df.label.isin(both)]              


def load_webkb(root):
    rows = []
    df = pd.DataFrame(rows, columns=["text", "label"]).assign(split="all")
    df["text"] = df.text.str.split(r"\r?\n\s*\r?\n", n=1, regex=True).str[-1].map(clean).str[:6000]
    return df

def load_emotions(root):
    return pd.concat([pd.read_csv(find(root,f),sep =";",header= None,names=["text", "label"], quoting=3).assign(split="test" if f == "test.txt" else "train") for f in ["train.txt", "val.txt", "test.txt"]])
   

LOADERS = {"r52": load_r52, "webkb": load_webkb, "emotions": load_emotions}

if __name__ == "__main__":
    m = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m.add_argument("--dataset", choices = list(LOADERS))
    m.add_argument("--seed", type = int, default = 0)
    m.add_argument("--val_size", type = int, default =500)
    m.add_argument("--out",default="data/tasks")
    m.add_argument("--raw_dir", help = "override")
    m.add_argument("--all", action="store_true")
    m.add_argument("--list", action="store_true", help="print class frequencies and exit")
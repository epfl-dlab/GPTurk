import pandas as pd
from sklearn.model_selection import train_test_split
from config import data_dir, home_dir
import os
import re
import numpy as np

SETTINGS = ["transductive", "inductive"]

def prepare_data(df, setting):
    if setting == "transductive":
        abstracts = df[df["class"]=="abstract"]
        df = df[df["class"].isin(["gen-summaries", "old-summaries"])]
        train, test = train_test_split(df, test_size=0.25)
        # train, val = train_test_split(train, test_size=0.1)
        
        train = pd.concat([train, abstracts, abstracts, abstracts], axis=0)           # to make sure the original abstract included
        train = train.reset_index(drop=True)     
        
        train.to_json(os.path.join(data_dir, "train_transductive.json"), orient="records", indent=2)
        # val.to_json(os.path.join(data_dir, "val_transductive.json"), orient="records", indent=2)
        test.to_json(os.path.join(data_dir, "test_transductive.json"), orient="records", indent=2)
    
    elif setting == "inductive":
        all_questions = list(df["HITId"].unique())
        train_questions = np.random.choice(all_questions, 11)
        q_set = set(all_questions)
        test_questions = list(q_set - set(train_questions))
        # val_questions = [test_questions[0]]
        # test_questions = test_questions[1:]

        train = df[df["HITId"].isin(train_questions)][["text", "labels", "HITId"]]
        # val = df[df["HITId"].isin(val_questions)][["text", "labels", "HITId"]]
        test = df[df["HITId"].isin(test_questions)][["text", "labels", "HITId"]]

        train.to_json(os.path.join(data_dir, "train_inductive.json"), orient="records", indent=2)
        # val.to_json(os.path.join(data_dir, "val_inductive.json"), orient="records", indent=2)
        test.to_json(os.path.join(data_dir, "test_inductive.json"), orient="records", indent=2)
        
                
    else:
        raise NotImplementedError("Setting needs to be inductive or transductive.")
        

if __name__=="__main__":
    base = pd.read_json(os.path.join(data_dir, "base.json"), orient="records")
    
    for setting in SETTINGS:
        prepare_data(base, setting)
    
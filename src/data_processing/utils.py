import pandas as pd 

def parse_response(actions):
    """
    Create proper datastructure for futher anaylsis
    """
    
    data = {"keys": "", "dates": []}
    
    for val in actions:
        if (val[0] == "copy") or (val[0] == "paste"):
            data["keys"] += " " + val[0]
            data["dates"].append(None)
        else:
            data["keys"] += " " + val[1]
            data["dates"].append(val[2])
    data["keys"] = " ".join(data["keys"].split()).strip()
    
    return data

def locate_paste(x: str) -> bool:
    """
    This function will attempt to use some heuristics to determine
    if the key strokes signify that the text was copied.
    """
    if ("Control v" in x) or ("paste" in x) or ("Meta v" in x) or ("Unidentified" in x):
        return True
    else:
        return False

def prolific_processor(df):
    relevant_columns = ["original_text","key_strokes", "datetime", "summary", "prolific_id"]
    
    responses = pd.DataFrame(columns=relevant_columns)
    
    for i, row in df.iterrows():
        summary = row["summary"]
        
        if row["log_of_what_they_did"].strip() == "":
            continue

        actions = eval(row["log_of_what_they_did"])
        
        data = parse_response(actions)

        key_strokes = data["keys"]
        datetime = row["_date"]
        worker_id = row["prolific_id"]
        temp = pd.DataFrame([[row["abstract"], key_strokes, datetime, summary, worker_id]], columns = relevant_columns)
        
        responses = pd.concat([responses, temp], axis=0)
    
    responses["copied"] = responses["key_strokes"].apply(lambda x: locate_paste(x))
    return responses

def mechanical_turk_processor(df):
    relevant_columns = ["HITId", "original_text","key_strokes", "datetime", "summary", "WorkerId"]
    
    responses = pd.DataFrame(columns=relevant_columns)

    for i, row in df.iterrows():
        unparsed = eval(row["Answer.taskAnswers"])[0]
        summary = unparsed["summary"]
        actions = eval(unparsed["log_of_what_they_did"])
        
        data = parse_response(actions)
        
        key_strokes = data["keys"]
        datetime = data["dates"]
        worker_id = hash(row["WorkerId"])
        temp = pd.DataFrame([[row["HITId"], row["Input.texts"], key_strokes, datetime, summary, worker_id]], columns = relevant_columns)
        
        responses = pd.concat([responses, temp], axis=0)
        
    responses["copied"] = responses["key_strokes"].apply(lambda x: locate_paste(x))
    return responses
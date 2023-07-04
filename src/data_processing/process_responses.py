import pandas as pd 
from config import data_dir
from utils import prolific_processor, mechanical_turk_processor
import os

def main(platform, batch_name):
    """
    Extract the features from the raw outputs. 
    
    Args:
        platform (str): mechanical_turk or prolific 
    """
    if platform not in ["mechancial_turk", "prolific"]: raise NotImplementedError("Make sure 'proflific' or 'mechanical_turk'")
    
    df = pd.read_csv(os.path.join(data_dir, platform, batch_name))
    df = df.dropna(subset="log_of_what_they_did")

    if platform == "prolific":
        processor = prolific_processor
    elif platform == "mechanical_turk":
        processor = mechanical_turk_processor
            
    responses = processor(df)
    
    responses.to_csv(os.path.join(data_dir, platform, "processed_responses.csv"))
    
if __name__=="__main__":
    PLATFORM = "prolific"
    BATCH_NAME = "july_03_prolific_original.csv"
    main(PLATFORM, BATCH_NAME)
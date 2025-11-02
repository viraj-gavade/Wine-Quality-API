import pandas as pd
from utils.logger import setup_logger
from src.custom_dataset import CustomDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
logger = setup_logger()

def process_data(df : pd.DataFrame ,output_feature : str  ):
    try:
        logger.info('Entred the Data Processing stage')
        df.drop(labels=['Unnamed: 0'],axis=1,inplace=True)
        logger.info('Dropped the unnecessary Columns')
        
        logger.info('Splitting the input and output features')
        X = df.drop(labels=[output_feature],axis=1)
        y = df[output_feature]
        logger.info('input and output features splited sucessfully!')

        logger.info('Applying the train test split')
        X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        
        logger.info('Creating the custom dataset objects')
        train_dataset = CustomDataset(X_train.values,y_train.values)
        test_dataset = CustomDataset(X_test.values,y_test.values)

        logger.info('Custom dataset object created')

        logger.info('creating the dataloader objects')
        train_loader = DataLoader(train_dataset,shuffle=True,batch_size=16)
        test_loader = DataLoader(test_dataset,shuffle=False,batch_size=16)

        logger.info('Data loader objects created')

        return train_loader , test_loader

    except Exception:
        logger.info('Exception Occured At ' , Exception)


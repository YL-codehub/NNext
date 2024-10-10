from backend.data.dataset import Dataset
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from utils.series_tools import make_next_data
from utils.data_tools import GenericArray
from sklearn.model_selection import train_test_split

class Assets(Dataset):
    """Data fetched from yahoo finance."""
    def __init__(self,cfg):
        super().__init__(cfg)
        if self.cfg.end_date != None:
            self.end_t = self.cfg.end_t
        else:
            self.end_t = datetime.today()
        self.end_t = self.end_t.strftime('%Y-%m-%d')
        self.start_t = (datetime.today() - timedelta(**{self.cfg.duration[0]: self.cfg.duration[1]}))
        self.start_t = self.start_t.strftime('%Y-%m-%d')
        self.t = None
    
    def fetch(self):
        self.fetched = yf.download(self.cfg.entities, start=self.start_t, end=self.end_t)
        self.fetched = self.fetched['Close'].dropna()
        self.t = self.fetched.index.tolist()

    def format(self):
        # temp =  (self.fetched/np.max(self.fetched)).to_numpy() ## renormalise
        temp =  (self.fetched).to_numpy() ## renormalise
        n = self.cfg.rolling_count
        recover_data, temp = make_next_data(temp,n,1000,1)
        # temp = temp/recover_data*100
        # self.X.ndarray, self.Y.ndarray = temp[:,:-1,:], temp[:,-1,:] # this thus contains the pct changes
        # self.Xbis.ndarray, self.Ybis.ndarray = recover_data[:,:-1,:], recover_data[:,-1,:] # while this contains the real 'floor' values
        self.X.ndarray, self.Y.ndarray = recover_data[:,:-1,:], recover_data[:,-1,:]
        self.Xbis.ndarray, self.Ybis.ndarray = recover_data[:,:-1,:], recover_data[:,-1,:]

    def store(self):
        return

    def refresh(self):
        return

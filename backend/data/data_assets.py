from data import Dataset
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from utils.series_tools import make_next_data

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
        self.Y =  (self.fetched/np.max(self.fetched)).to_numpy() ## renormalise
        n = 10 ##>=2
        recover_data, self.Y = make_next_data(self.Y,n,1000,1)
        self.X, self.Y = self.Y[:,:-1,:], self.Y[:,-1,:]
        # recover_x2, recover_y2 = recover_data[:,:-1,:], recover_data[:,-1,:]

        self.X= np.reshape(self.X,(self.X.shape[0],self.X.shape[1]*self.X.shape[2])) 



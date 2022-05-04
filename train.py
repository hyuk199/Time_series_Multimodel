import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from utils.dataloader import house_Load_data , jena_Load_data, geo_Load_data
from utils.models import Conv_lstm, lstm, Bilstm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
list_csv =[]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, n_future, step):
        x = []
        y = []
        self.n_future = n_future # predict future
        
        for i in range(len(data)-seq_length -self.n_future):
            x.append(data[range(i, i+seq_length, step)])
            # Target Label Num
            y.append(data[i+seq_length+self.n_future, :])

        self.X_seq, self.y_seq = np.array(x), np.array(y)

    def __len__(self):
        return len(self.y_seq) 

    def __getitem__(self, idx): 
        item = [
                    torch.tensor(self.X_seq[idx], dtype=torch.float32), 
                    torch.tensor(self.y_seq[idx], dtype=torch.float32)
                ]
        return item


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, data_np, seq_length, future, batch_size, step, seed):
        super().__init__()
        self.data_np = data_np
        self.seq_length = seq_length
        self.future = future
        self.step = step
        self.seed =seed

        self.batch_size = batch_size

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.pred_dataset = None

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from
        # a single process in distributed settings.
        self.Datadset = TimeSeriesDataset(self.data_np, self.seq_length, self.future, self.step)

    def setup(self, stage=None):
        #-- make train, valid, test dataset [0.8 / 0.1 / 0.1]
        self.train_dataset, self.test_dataset = random_split(self.Datadset, [int(len(self.Datadset) * 0.8), (len(self.Datadset) - int(len(self.Datadset) * 0.8))])
        self.test_dataset, self.valid_dataset = random_split(self.test_dataset, [int(len(self.Datadset) * 0.5), (len(self.test_dataset) - int(len(self.test_dataset) * 0.5))])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class TimeSeriesModel(pl.LightningModule):
    def __init__(self, n_features, n_hidden, seq_len, n_layers, kernel_size, input_num, target_num, valid_length, test_length, lr, dropout, model_type):
        super().__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.input_num = input_num
        self.kernel_size = kernel_size
        self.valid_length = valid_length
        self.ts_length = test_length
        self.lr = lr

        # Set your Loss function
        self.loss_fn = nn.SmoothL1Loss()
        self.loss_m = nn.L1Loss()
        # self.loss_m = nn.MSELoss()

        self.model_type = model_type

        if self.model_type == 'conv':
            self.model = Conv_lstm(n_features, self.n_hidden, self.seq_len, self.n_layers, self.kernel_size, self.input_num, target_num, dropout, device=device)

        elif self.model_type == 'lstm':
            self.model = lstm(n_features, self.n_hidden, self.seq_len, self.n_layers, self.input_num, target_num, dropout, device=device)

        elif self.model_type == 'bilstm':
            self.model = Bilstm(n_features, self.n_hidden, self.seq_len, self.n_layers, self.input_num, target_num, dropout, device=device)


    def forward(self, sequences):
        y_pred = self.model.forward(sequences)
        return y_pred

    def training_step(self, batch, batch_idx):
        x_data, y_data = batch

        y_pred = self.forward(x_data)
        y_pred = torch.squeeze(y_pred)
        loss = self.loss_fn(y_pred.float(), torch.squeeze(y_data))
        loss_m = self.loss_m(y_pred.float(), torch.squeeze(y_data))

        return {"loss": loss, "m": loss_m}

    def validation_step(self, batch, batch_idx):
        val_data, y_val = batch

        y_val_pred = self.forward(val_data)
        y_val_pred = torch.squeeze(y_val_pred)
        val_loss = self.loss_fn(y_val_pred.float(), torch.squeeze(y_val))
        val_loss_m = self.loss_m(y_val_pred.float(), torch.squeeze(y_val))

        return {"loss": val_loss, "m": val_loss_m}

    def validation_step_end(self, outputs):
        return {"loss": outputs['loss'], "m": outputs['m']}

    def validation_epoch_end(self, outputs):
        val_loss = torch.cat([output['loss'] for output in outputs], dim=0)
        val_loss_m = torch.cat([output['m'] for output in outputs], dim=0)
                   
        self.log("valid_loss", val_loss, on_step=False, on_epoch=True, logger=True)
        self.log("valid_MAE", val_loss_m, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        ts_data, y_ts = batch
        
        y_ts_pred = self.forward(ts_data)
        y_ts_pred = torch.squeeze(y_ts_pred).float()
        y_ts =torch.squeeze(y_ts)

        ts_loss = self.loss_fn(y_ts_pred, y_ts)
        ts_loss_m = self.loss_m(y_ts_pred, y_ts)

        return {"loss": ts_loss, "m": ts_loss_m}


    def test_epoch_end(self, outputs):
        # print('test_out',outputs)
        loss = [output['loss'] for output in outputs]
        ts_loss = sum(loss)/len(loss)
        loss_m = [output['m'] for output in outputs]
        ts_loss_m = sum(loss_m)/len(loss_m)
                   
        self.log("test_loss", ts_loss, on_step=False, on_epoch=True, logger=True)
        self.log("test_MAE", ts_loss_m, on_step=False, on_epoch=True, logger=True)


def main(config=None):
    pl.seed_everything(seed=config.seed)

    if config.flie_type =='house':
        # Call datafile
        df = pd.read_csv('data/household_power_consumption.txt',
                    parse_dates={'dt' : ['Date', 'Time']},
                    sep=";", infer_datetime_format=True,
                    low_memory=False, na_values=['nan','?'], index_col='dt')
        # datafile to numpy data
        data_np = house_Load_data(df, config.scale)

    elif config.flie_type == 'jena':
        df = pd.read_csv('data/jena_climate_2009_2016.csv')
        data_np = jena_Load_data(df, config.scale)

    elif config.flie_type == 'geo':
        df = pd.read_csv('data/measure1_smartphone_sens.csv')
        data_np = geo_Load_data(df, config.scale)
    
    input_num = data_np.shape[1]
    target_num = data_np.shape[1]

    dm = TimeSeriesDataModule(data_np, config.seq_len, config.future, config.batch_size, config.step, config.seed)
    dm.prepare_data()
    dm.setup()

    model = TimeSeriesModel(n_features=config.n_features,
        n_hidden= config.n_hidden,
        seq_len= config.seq_len,
        n_layers= config.n_layers,
        input_num= input_num,
        target_num = target_num,
        kernel_size= config.kernel_size,
        valid_length=len(dm.valid_dataset),
        test_length=len(dm.test_dataset),
        lr = config.learning_rate,
        dropout = config.dropout,
        model_type=config.model_type)

    # EarlyStopping 5 epoch
    early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=5, verbose=10, mode="min")
    trainer = pl.Trainer(gpus=[0],
                        max_epochs=config.max_epoch,
                        callbacks=[early_stop_callback],
                        accelerator='dp'
                        )

    model.train()
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)

if __name__ == '__main__':
    config = {
                # About Training
                "max_epoch": 300,
                "batch_size": 512,
                "seed":42,

                # About Data
                "scale" : True, # Scaling data True, False
                "flie_type" : "jena", # house, jena, geo
                "seq_len" : 120,
                "future" : 0,
                "step" : 1,

                # About Model
                "model_type": "conv", # conv, lstm, bilstm
                "learning_rate": 0.004,
                "dropout" : 0.15,
                "n_features" : 64,
                "n_hidden" : 64,
                "n_layers" : 1,
                "kernel_size" : 4
            }
    main(config)

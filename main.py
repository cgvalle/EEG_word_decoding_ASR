import os; os.system('clear')
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.input_processing import input_processing
from src.spanish_dataset import SpanishDataset
from src.text_transform import TextTransform
from src.eeg_datasetv2 import EEGSetupV2, EEGDataset, EEGDatasetRaw
from src.text_transform_eeg import TextTransformEEG
from src.input_processing_eeg import input_processing_eeg, input_processing_eeg_raw
from torch.utils.data import DataLoader
import textgrid
from collections import defaultdict
from torch.utils.data import Dataset
import random
import argparse
import pandas as pd
import mne

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', type=str)
parser.add_argument('-r', '--run', type=int, default=1)
parser.add_argument('--task', type=str)
parser.add_argument('--label_shuffle', action=argparse.BooleanOptionalAction)
parser.add_argument('--word_type',  action=argparse.BooleanOptionalAction)
parser.add_argument('--word_position',  action=argparse.BooleanOptionalAction)
parser.add_argument('--name', type=str)



args = parser.parse_args()


tipo={
    "articles": ["el", "la", "los", "una", "ningun"],
    "noun": ["inteligencia", "artificial", "estomago", "ejercicios", "gimnasio", "invierno", "plazo", "temblor", "esposa", "edificios", "tele", "hermanos", "música", "educación", "vacaciones", "vino", "azúcar", "convivencia", "mochila", "tienda", "micro", "metro", "marido", "año", "prueba", "abuela", "altos", "cucharadas", "dia", "dios", "dos", "ejercicio", "resfriado", "siete", "vecinos", "vida", "azucar", "dias", "educacion", "lado", "musica"],
    "adjectives": ["mas","real", "terrible", "flojo", "fuerte", "nervioso", "malo", "pocos", "buena", "igual", "mayor", "muy", "mismo", "todo", "todos"],
    "pronouns": ["me", "yo", "esa", "mi", "ellos", "te", "mis", "eso", "este", "mia"],
    "adverbs": ["recién", "más", "temprano", "no", "todavía", "antes", "cuando", "nunca", "cada", "todavia", "recien"],
    "prepositions": ["para", "a", "con", "de", "por", "en", "al"],
    "conjunctions": ["y", "que", "pero", "si"],
    "verbs": ["gracias", "dijeron", "es", "era", "soy", "voy", "caminando", "estuve", "vaya", "saliendo", "llamo", "hay", "decir", "tiene", "faltan", "salir", "gusta", "tengo", "anda", "puedas", "comprar", "estoy", "esperando", "llamen", "pasa", "pongo", "vivo", "dan", "miedo", "querían", "jubilamos", "estar", "estaba", "fue", "fui", "hacer", "ir", "puede", "tuve", "se", "posponer", "querian"],
}

word2tipo = {}

for key in tipo.keys():
    for word in tipo[key]:
        word2tipo[word] = key
        



        





class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class),
        )
        
        self.final_maxpool = nn.MaxPool1d(3, stride=4)
        self.final_fc = nn.Linear(62, 1)    
        

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        # Added layers
        x = x.transpose(1, 2)
        x = self.final_maxpool(x) 
        x = self.final_fc(x)
        return x


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device,early_stopping, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment):
    model.train()
    data_len = len(train_loader.dataset)

    print("Epoch: ", epoch)
    LOSS = []
    ACC = []
    for batch_idx, _data in tqdm(enumerate(train_loader), total=int(data_len/train_loader.batch_size)):
        eeg_window, labels = _data 
        eeg_window = eeg_window.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(eeg_window).squeeze(2)  # (batch, time, n_class)
        _, predicted = torch.max(output, 1)
        ACC.append((predicted == labels).sum().item() / len(predicted)*100)
        #print(eeg_window.shape)
        #print(labels.shape)
        #print(output.shape)
        
        
        
        loss = criterion(output, labels)
        loss.backward()
        LOSS.append(loss.item())

        optimizer.step()
        scheduler.step()
        iter_meter.step()
   
    experiment.add_scalar(tag='loss/train', scalar_value=np.mean(LOSS), global_step=epoch)
    experiment.add_scalar(tag='acc/train', scalar_value=np.mean(ACC), global_step=epoch)
    
    print("Loss:", np.mean(LOSS))



def test(model, device, early_stopping, val_loader, test_loader, criterion, epoch, iter_meter, experiment, test=False):
    if epoch % 5 != 0:
        return
    
    
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    ACC = [] 
    data_len = len(val_loader.dataset)
    PREDICTED  = []
    LABELS = []
    with torch.no_grad():
        for i, _data in tqdm(enumerate(val_loader), total=int(data_len/val_loader.batch_size)):
            spectrograms, labels = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms).squeeze(2)  # (batch, time, n_class)
            _, predicted = torch.max(output, 1)
            ACC.append((predicted == labels).sum().item() / len(predicted)*100)
            
            PREDICTED.extend(predicted.tolist())
            LABELS.extend(labels.tolist())

            loss = criterion(output, labels)
            test_loss += loss.item() / len(val_loader)
    print(predicted[:20])
    print(labels[:20])

    # EARLY STOPPING
    stop = early_stopping.early_stop(test_loss, model)
    if stop:
        print("TERMINANDO ANTES")

    if not test and not stop:
        experiment.add_scalar('loss/val', test_loss, global_step=epoch)
        experiment.add_scalar('acc/val', np.mean(ACC), global_step=epoch)
    else:
        model.load_state_dict(torch.load(log_dir + "/model.pt")['model_state_dict'])
        model.eval()
        test_loss = 0
        ACC = [] 
        data_len = len(test_loader.dataset)
        PREDICTED  = []
        LABELS = []
        with torch.no_grad():
            for i, _data in tqdm(enumerate(test_loader), total=int(data_len/test_loader.batch_size)):
                spectrograms, labels = _data 
                spectrograms, labels = spectrograms.to(device), labels.to(device)

                output = model(spectrograms).squeeze(2)  # (batch, time, n_class)
                _, predicted = torch.max(output, 1)
                ACC.append((predicted == labels).sum().item() / len(predicted)*100)
                
                PREDICTED.extend(predicted.tolist())
                LABELS.extend(labels.tolist())

                loss = criterion(output, labels)
                test_loss += loss.item() / len(test_loader)



        experiment.add_scalar('loss/test', test_loss, global_step=epoch)
        experiment.add_scalar('acc/test', np.mean(ACC), global_step=epoch)
        print("Test loss: ", test_loss)
        print("Test acc: ", np.mean(ACC))
        data = {'predicted': PREDICTED, 'labels': LABELS}
        df = pd.DataFrame(data=data)
        if not args.label_shuffle:
            df.to_csv(log_dir +"/class.csv", index=False)
        
        raise Exception("End Training")
        
    



    #print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))

class EEGSETUP:
    
    def __init__(self, npz_file, 
                 subjects=None, 
                 stims=None, 
                 tasks=None, 
                 label_shuffle=False, 
                 word_duration=0.5,
                 window_size=3000, 
                 train_ratio=0.8, 
                 test_ratio=0.1, 
                 val_ratio=0.1):
        
        self.npz_file = npz_file
        self.subjects = subjects
        self.tasks = tasks
        self.stims = stims
        self.word_duration = word_duration
        self.label_shuffle = label_shuffle
        self.window_size = window_size
        self.train_ratio  =train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.sfreq = 1000
        
        
        self.stim_words = self.load_words()
        self.load_trials()
        
    def load_words(self):
        stims_words = defaultdict(list)
        for i in range(1, 31):
            praat_path = f"src_eeg/praat_annotations/{i}.TextGrid"
            tg = textgrid.TextGrid.fromFile(praat_path)
            
            for word in tg[0]:
                if len(word.mark.strip()) == 0:
                    continue
                stims_words[i].append([word.mark, int(self.sfreq * word.minTime), int(self.sfreq * word.maxTime)])
          
        return stims_words
                
        
    def load_trials(self):
        # Load trials
        self.trials = []
        with np.load(self.npz_file, 'r', allow_pickle=True) as data:
            for key in tqdm(data.keys()):
                if key == 'info':
                    _, self.info = data['info']
                    continue
                    
                subject, task, stim = key.split('_')
                
                    
                # subjects
                if self.subjects is not None:
                    if subject not in self.subjects:
                        continue
                
                # task
                if self.tasks is not None:
                    if task not in self.tasks:
                        continue
                
                # stim
                if self.stims is not None:
                    if int(stim) not in self.stims:
                        continue    
                
                # iterate over repetitions for a stimulus
                num_epochs, num_channels, _ = data[key].shape

                for epoch_index in range(num_epochs):
                    for c in range(num_channels):
                        X = data[key][epoch_index, c, :]
                        INDEX = 1
                        for word, start_time, end_time in self.stim_words[int(stim)]: 
                            start_index = start_time
                            end_index  = int(start_time + self.word_duration * self.sfreq)
                            assert X[start_index:end_index].shape[0] == self.word_duration * self.sfreq
                            
                            
                            
                            if INDEX <= 5:
                                # add word type
                                if args.word_type:
                                    self.trials.append((X[start_index:end_index].astype(np.float32), word2tipo[word.lower().strip()]))
                                elif args.word_position:
                                    self.trials.append((X[start_index:end_index].astype(np.float32), str(INDEX)))
                                else:
                                    self.trials.append((X[start_index:end_index].astype(np.float32), word.lower().strip()))
                            INDEX += 1
        
       
                                

                                
                            
    def balanced_split(self):
        # Split dataset into train, validation and test
        # The dataset is balanced
        # Input: X, Y (numpy arrays)
        # Output: X_train, X_val, X_test, Y_train, Y_val, Y_test.

        # Get the number of samples in the dataset
        n_samples = len(self.trials)
        
        
        # Shuffle the indices of the samples
        indices = np.random.permutation(n_samples)
        
        # Split the shuffled indices into train, validation, and test sets
        train_end = int(self.train_ratio * n_samples)
        val_end = int((self.train_ratio + self.val_ratio) * n_samples)
        train_indices = np.array(indices[:train_end]).astype(int)
        val_indices = np.array(indices[train_end:val_end]).astype(int)
        test_indices = np.array(indices[val_end:]).astype(int)

        
        print("Train: ", len(train_indices))
        print("Test: ", len(test_indices))
        print("Val: ", len(val_indices))
        return train_indices, val_indices, test_indices
                     
        
                
    def get_words(self):
        Y = []
        for _, word_label in self.trials:
            Y.append(word_label)
            
        vocab = sorted(set(Y))
        print("Total trials: ", len(self.trials))
        print("Total unique labels: ", len(vocab))
        
        self.word2index = {word: i for i, word in enumerate(vocab)}
        self.index2word = {i: word for i, word in enumerate(vocab)}
        self.vocab = vocab
        print(self.word2index)
        new_trials = []
        for i, (X, word_label) in enumerate(self.trials):
            if self.label_shuffle:
                word_label = random.choice(vocab)
            new_trials.append((X, self.word2index[word_label]))
        self.trials = new_trials
        print(self.word2index)

        
        if args.word_type:
            # Balance classes
            l = [t[1] for t  in self.trials]
            lowest = []
            for i in range(8):
                print(i, l.count(i))
                lowest.append(l.count(i))

            lowest = np.min(lowest)
            print("min: ", lowest)
            trials = defaultdict(list)
            for x,y in self.trials:
                trials[y].append(x)
            print([len(i) for i in trials.values()])

            new_trials = []
            for y in range(8):
                samples = random.sample(trials[y], lowest)
                for x in samples:
                    new_trials.append((x,y))

            self.trials = new_trials
            random.shuffle(self.trials)



        return  vocab

        
        
eeg_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=1000, 
        n_fft=128, 
        win_length=32, 
        hop_length=4, 
        n_mels=8, 
        f_min=0.5, 
        f_max=70,
        normalized=False,
        center=False,
        power=2,
    ) 
    
import librosa
from matplotlib import pyplot as plt


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    plt.show(block=False)
    plt.savefig('sample_spectrogram.png')
  
class EEGDATASET(Dataset):
    
    def __init__(self,trials):
        self.trials = trials
        self.X = np.array([self.trials[i][0] for i in range(len(trials))], dtype=np.double)
        self.Y = [self.trials[i][1] for i in range(len(trials))]
        
        n_feats = 6
        #freqs = np.linspace(0.5, 70, n_feats + 1)
        freqs = np.array([ 1, 4, 8, 13, 30, 50, 80])  # EEG bands

        # run 1: word_duration 0.5
        # run 2: word_duration 0.3
        
        self.spec_X = []
        for i in tqdm(range(freqs.shape[0] - 1)):
            self.spec_X.append(mne.filter.filter_data(self.X, 1000, freqs[i], freqs[i + 1], method='iir', verbose=False, n_jobs=6))

        self.spec_X = np.array(self.spec_X)
        self.X  = torch.tensor(self.spec_X.transpose(1, 0, 2)).float()
        self.X = self.X.unsqueeze(1)
        print("Shape of X: ", self.X.shape)


    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, index):
        # Get trial
        x = self.X[index]
        y = torch.tensor(self.trials[index][1])


        
                    
        return x, y         

                    
class EarlyStopper:
    def __init__(self,path,  patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.path = path

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save({'model_state_dict': model.state_dict()}, self.path + "/model.pt")

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def main(learning_rate=5e-4, batch_size=16, epochs=10,
        train_url="train-clean-100", test_url="test-clean",
        experiment=None):

    hparams = {
        "n_cnn_layers": 2,
        "n_rnn_layers": 2,
        "rnn_dim": 64,
        "n_feats": 6,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": 3,
        # Dataloader
        "npz_file": 'src_eeg/eeg_npz/language_average_0.5-Nonehz_icaLabel95confidence_60sessions_perception.npz',
        "subjects": [args.subject],
        #"checkpoint": 'asr_big/version_1/model_100.pt',
        "label_shuffle": False,
        "freeze": None,
        "downsample_to": 200,
        "downsample_aug": 200,
        "char_shuffle": False,
        "window_size": 3000,
        "window_stride": 400,
        "utterance_type": 'char',
        "task": args.task,    
        "resample_aug": False,
        "resample_factor": 4,
        # Spectrogram parameters
        "n_fft": 32,
        "win_length": 128,
        "hop_length": 4,  
    }
    hparams['sfreq'] = 1000


    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    #######################
    #### LOAD EEG DATA ####
    #######################

    eeg_setup = EEGSETUP(hparams['npz_file'], 
                        subjects = hparams['subjects'],
                        #stims=[1,2,3,4,5],
                        tasks=[hparams['task']],
                        label_shuffle=args.label_shuffle,
                        window_size=hparams['window_size'],
                        train_ratio=0.8,
                        test_ratio=0.1,
                        val_ratio=0.1,
                        )


    vocab = eeg_setup.get_words()
    train_indices, val_indices, test_indices = eeg_setup.balanced_split()
    

    
    
    print("Vocab: ", vocab)
    print(vocab)
    


    # Create datasets   
    # train 
    train_set = EEGDATASET([eeg_setup.trials[i] for i in train_indices])
    test_set = EEGDATASET([eeg_setup.trials[i] for i in test_indices])
    val_set = EEGDATASET([eeg_setup.trials[i] for i in val_indices])

    print('train set size: ', len(train_set))
    print('test set size: ', len(test_set))
    print('val set size: ', len(val_set))

    ####################
    #### DATALOADERS ###
    ####################
    kwargs = {'num_workers': 6, 
            'pin_memory': True}
    hparams['n_class'] = len(vocab)   
    shuffle = True
    train_loader = DataLoader(dataset=train_set,
                            batch_size=hparams['batch_size'],
                            shuffle=shuffle,
                            **kwargs
                            )

    test_loader = DataLoader(dataset=test_set,
                            batch_size=hparams['batch_size'],
                            shuffle=False,
                            **kwargs
                            )

    val_loader = DataLoader(dataset=val_set,
                            batch_size=hparams['batch_size'],
                            shuffle=False,
                            **kwargs
                            )

    for x_batch, y_batch in train_loader:
        print(x_batch.shape, y_batch.shape)
        break
    

    ####################
    ###### MODELS ######
    ####################
    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)
    
    
    if 'checkpoint' in hparams:
        checkpoint = torch.load(hparams['checkpoint'])
        for key in list(checkpoint['model_state_dict'].keys()):
            checkpoint['model_state_dict'][key.replace('module.','')] = checkpoint['model_state_dict'].pop(key)

        
        print(checkpoint['model_state_dict'].keys())
        checkpoint_n_class = checkpoint['model_state_dict']['classifier.3.weight'].shape[0]
        model.classifier[3] = nn.Linear(hparams['rnn_dim'], checkpoint_n_class)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.classifier[3] = nn.Linear(hparams['rnn_dim'], hparams['n_class'])
        print("Checkpoint loaded")
        model = model.to(device)
    
    #checkpoint = torch.load('original_eeg/models/363_normal.pt')
    #model.load_state_dict(checkpoint)
    
    
    
        
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))


    optimizer = optim.Adam(model.parameters(), hparams['learning_rate'])
    criterion = nn.CrossEntropyLoss() 
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')
    early_stopping = EarlyStopper(patience=hparams['patience'], path=log_dir)
    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        train(model, device, early_stopping, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
        test(model, device, early_stopping, val_loader, test_loader, criterion, epoch, iter_meter, experiment)
        
    test(model, device, val_loader, test_loader, criterion, epoch, iter_meter, experiment, test=True)
 


project_name = "speechrecognition"




log_dir = f"BYFREQBAND/{args.name}/{args.task}/run_{args.run}/{args.subject}"


experiment = SummaryWriter(
            log_dir=log_dir,
            flush_secs=5
)
  
learning_rate = 5e-4 
batch_size = 32  
epochs = 100

libri_train_set = "train-clean-100"
libri_test_set = "test-clean"
model = main(learning_rate, batch_size, epochs, libri_train_set, libri_test_set, experiment)





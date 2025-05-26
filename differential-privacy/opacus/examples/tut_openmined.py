import torch
from torchvision import datasets, transforms
import numpy as np
from opacus import PrivacyEngine
from tqdm import tqdm

# '../mnist': map waarin de dataset opgeslagen wordt
#train=True: dit is de trainingsdata
#download=True: download automatisch als het nog niet bestaat
#transform=...: converteert de afbeeldingen naar een formaat dat het model begrijpt
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist',
			   train=True, download=True,
               #Hier worden twee bewerkingen toegepast op elke afbeelding
               #ToTensor()
               #→ Zet de afbeelding om naar een PyTorch Tensor (getallen tussen 0 en 1)
               transform=transforms.Compose([transforms.ToTensor(),
               #→ Normaliseert de pixels: elk pixel = (pixel - 0.1307) / 0.3081
               #Dit helpt het model sneller en stabieler leren
               #De waarden 0.1307 (mean) en 0.3081 (std) zijn vooraf berekend over de hele MNIST dataset.
               transforms.Normalize((0.1307,), (0.3081,)),]),),
               batch_size=64, shuffle=True, num_workers=1,
               pin_memory=True)
#batch_size=64: het model traint op 64 afbeeldingen tegelijk
#shuffle=True: schudt de data zodat het model niet leert op volgorde
#num_workers=1: aantal subprocessen om data te laden (1 is prima op CPU)
#pin_memory=True: versnelt data-overdracht naar CUDA (geen effect op CPU, maar geen kwaad)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist',
			  train=False,
              transform=transforms.Compose([transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,)),]),),
              batch_size=1024, shuffle=True, num_workers=1,
              pin_memory=True)
# Dit betekent alleen: hoeveel afbeeldingen je tegelijk in het geheugen laadt
#Niet hoeveel totale data er is.. ik bedoel heir over de batch size


# bouwen van een neuraal netwerk in PyTorch 
#
#Neemt een zwart-wit afbeelding (1 kanaal) van 28×28 pixels
#Maakt 16 filters van 8×8 over de afbeelding
#Verplaatst met stapgrootte 2
#padding=3 = maakt het beeld iets groter om randen mee te pakken

model = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 8, 2, padding=3),
							torch.nn.ReLU(), #activatie functie houdt alleen positieve waarden over 
                            torch.nn.MaxPool2d(2, 1), #Kijkt naar een klein vakje (2x2 pixels) en kiest daaruit de grootste waarde
                            torch.nn.Conv2d(16, 32, 4, 2), 
                            torch.nn.ReLU(), 
                            torch.nn.MaxPool2d(2, 1), 
                            torch.nn.Flatten(), #De afbeelding is nu een 3D blok (bijvoorbeeld 32 filters van 4x4 pixels)
#Flatten() maakt daar één lange rij van getallen van zo kunnen we naar een gewone "beslissing" toe werken
                            torch.nn.Linear(32 * 4 * 4, 32), #“Neem 512 getallen als input, en geef er 32 terug.” en 513 is filter * hoogte * breedte
                            torch.nn.ReLU(), 
                            torch.nn.Linear(32, 10)) #Laatste stap: het model geeft 10 getallen

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

#SGD = Stochastic Gradient Descent → helpt het model beter worden bij elke stap
#lr=0.05 = learning rate → hoe snel leert het model (0.05 is best snel)
#model.parameters() = alles wat het model moet leren


#hiermee wwordt de dp toegepast

#PrivacyEngine is een onderdeel van de Opacus library.
#Het zorgt ervoor dat:

#Er ruis (noise) wordt toegevoegd tijdens training
#Je de privacy-bescherming kunt meten (ε en δ)
#Het model niet te veel leert over individuele voorbeelden

privacy_engine = PrivacyEngine(model, #model die ik wil beveiligen 
							   batch_size=64, # Hoeveel voorbeelden tegelijk in één stap (moet kloppen met train_loader)
                               sample_size=60000,   ## Aantal voorbeelden in je training set (MNIST heeft 60.000)
                               alphas=range(2,32), # Niveaus voor privacyberekening (geavanceerde wiskunde, maar nodig)
                               noise_multiplier=1.3, ## Hoeveel ruis je toevoegt (meer ruis = meer privacy, maar minder nauwkeurig)
                               max_grad_norm=1.0,) #Houdt gevoelige updates binnen veilige grenzen

privacy_engine.attach(optimizer)


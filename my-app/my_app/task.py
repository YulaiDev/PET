"""my-app: A Flower / PyTorch app."""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from transformers import AutoTokenizer


class Net(nn.Module):
    """Model DistilBERT """

    def __init__(self):
        super(Net, self).__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition sentiment140 data and preprocess it."""
    # Only initialize `FederatedDataset` once
    print("Loading data")
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(dataset="stanfordnlp/sentiment140", partitioners={"train": partitioner},)
    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test["train"] = partition_train_test["train"].select(range(500)) #select to make traning faster
    partition_train_test["test"] = partition_train_test["test"].select(range(100))

    #preprocess data so it can be fed into the model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    #converting labels for convinience 
    def map_labels(batch):
        label_map = {0: 0, 2: 1, 4: 2}
        batch["sentiment"] = [int(label_map[l]) for l in batch["sentiment"]]
        return batch

    #in this example we are using text data and this is a standard step of preprocessing while using this type of data
    def tokenize(batch):
        """Tokenize strings first"""
        tokenized = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
        return tokenized

    #mapping labels and tokenizing data; converting it to torch type (expected by DistilBert model)
    for split in ["train", "test"]:
        partition_train_test[split] = partition_train_test[split].map(map_labels, batched=True)
        partition_train_test[split] = partition_train_test[split].map(tokenize, batched=True)
        partition_train_test[split].set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "sentiment"],
        )

    #load data
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)

    return trainloader, testloader


#given function, adapted only iin the 'for' loop
def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    net.train()
    running_loss = 0.0
    for batch in trainloader:
        print("Got a batch")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["sentiment"].to(device)
        optimizer.zero_grad()
        outputs = net(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


#given function, adapted only in the 'for' loop 
def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["sentiment"].to(device)
            outputs = net(input_ids=input_ids, attention_mask=attention_mask)
            loss += criterion(outputs, labels).item()
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return {"loss": loss, "accuracy": accuracy}


#get&set weights are given functions; they take part in the process of updating out model
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

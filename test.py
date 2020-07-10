import torch
from data_processing import get_train_set_by_cls, get_test_set_by_cls

classes = [0, 1, 8]

print("Verifying train set.")
train_set = get_train_set_by_cls(classes, flatten=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

train_total = 0
extracted_correct_cls_train = True
for counter, (inp, lab) in enumerate(train_loader):
    train_total += inp.size(0)
    if lab.item() not in classes:
        extracted_correct_cls_train = False
    if counter == 0:
        inp_size, lab_size = inp.size(), lab.size()

print("\tTotal train data = {}".format(train_total))
print("\tInput size = {} (batch size = 1). Label size = {} (batch size = 1).".format(inp_size, lab_size))
print("\tExtracted correct classes: {}".format(extracted_correct_cls_train))

print("Verifying test set.")
test_set = get_test_set_by_cls(classes, flatten=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

test_total = 0
extracted_correct_cls_test = True
for counter, (inp, lab) in enumerate(test_loader):
    test_total += inp.size(0)
    if lab.item() not in classes:
        extracted_correct_cls_test = False
    if counter == 0:
        inp_size, lab_size = inp.size(), lab.size()

print("\tTotal test data = {}".format(test_total))
print("\tInput size = {} (batch size = 1). Label size = {} (batch size = 1).".format(inp_size, lab_size))
print("\tExtracted correct classes: {}".format(extracted_correct_cls_test))
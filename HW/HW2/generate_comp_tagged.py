from basic_model import *


def tagger(model, data_loader):
    tags = []
    for batch_idx, input_data in enumerate(data_loader):
        with torch.no_grad():
            words_idx_tensor, pos_idx_tensor, heads_tensor = input_data
            tag_scores = model(words_idx_tensor, pos_idx_tensor)
            predicted_mst, _ = decode_mst(energy=tag_scores.detach(), length=tag_scores.shape[0], has_labels=False)
            tags.append(predicted_mst[1:])
    return tags

# create data sets
data_dir = "HW2-files/"
path_train = data_dir + "train.labeled"
path_test = data_dir + "test.labeled"
paths_list = [path_train, path_test]
word_cnt, word_dict, pos_dict = get_vocabs(paths_list)
train = PosDataset(word_cnt, word_dict, pos_dict, data_dir, 'train')
# split into validation
train_set, val_set = torch.utils.data.random_split(train, [4000, 1000])
train_dataloader = DataLoader(train_set, shuffle=False)  # TODO return to true after debugging
val_dataloader = DataLoader(val_set, shuffle=False)
test = PosDataset(word_cnt, word_dict, pos_dict, data_dir, 'test')
test_dataloader = DataLoader(test, shuffle=False)
word_vocab_size = len(train.word2idx)
print(word_vocab_size)
tag_vocab_size = len(train.pos_idx_mappings)
print(tag_vocab_size)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# create and load trained model
basic_model = DnnDependencyParser(WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, HIDDEN_DIM, word_vocab_size, tag_vocab_size).to(device)
if path.exists(PATH):
    print("loading model")
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')  # 'cpu' in this case
    basic_model.load_state_dict(torch.load(PATH, map_location=DEVICE))
    # basic_model.load_state_dict(torch.load(PATH))
else:
    print("model not found")
# inference
path_comp = data_dir + "comp.unlabeled"

comp = PosDataset(word_cnt, word_dict, pos_dict, data_dir, 'comp')
comp_dataloader = DataLoader(comp, shuffle=False)   # comp data loader
head_tokens_tags = tagger(basic_model, comp_dataloader)
comp_tagger(path_comp, "our_tagged_version", head_tokens_tags)

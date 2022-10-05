from torch.utils.data.dataloader import default_collate
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')



def collate(batch):
    batch = default_collate(batch)
    #assert(len(batch) == 2)
    if (int(config['data']['augment_data'])):
        batch_size, num_aug, channels, height, width = batch[0].size()
        batch[0] = batch[0].view(
            [batch_size * num_aug, channels, height, width])
        batch[1] = batch[1].view([batch_size * num_aug])
    return batch
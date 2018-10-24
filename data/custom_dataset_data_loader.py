import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None

    if opt.use_celeba:
        from data.celeba_dataset import CelebaDataset
        dataset=CelebaDataset()
    elif opt.use_cub:
        from data.bird_dataset import BirdDataset
        dataset=BirdDataset()
    elif opt.use_style:
        if opt.use_vae:
            from data.style_dataset_vae import StyleDatasetVae
            dataset=StyleDatasetVae()
        else:
            from data.style_dataset import StyleDataset
            dataset=StyleDataset()
    elif opt.unpair_style:
        from data.unpair_style_dataset import UnpairStyle
        dataset=UnpairStyle()
    elif opt.use_fashion:
        from data.fashion_dataset import FashionDataset
        dataset=FashionDataset()
    else:
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
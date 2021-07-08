from dataset import TactileDataset
from torch.utils.data import DataLoader

def get_datasets(data_dir, fold_number, output_size, trial_number, batch_size=32, test=False):
    train_dataset = TactileDataset(
        path=data_dir,
        fold=fold_number,
        trial_number=trial_number,
        output_size=output_size,
        split_name = 'train'
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_dataset = TactileDataset(
        path=data_dir,
        fold=fold_number,
        trial_number=trial_number,
        output_size=output_size,
        split_name = 'validation'
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    
    if not test:
        return train_dataset, train_loader, val_dataset, val_loader
    
    test_dataset = TactileDataset(
        path=data_dir,
        fold=fold_number,
        trial_number=trial_number,
        output_size=output_size,
        split_name = 'test'
    )
    tets_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, tets_loader
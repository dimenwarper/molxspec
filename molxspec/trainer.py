import training_setup
from tqdm import tqdm

def main():
    pbar = tqdm(list(training_setup.SETUPS.items()))
    for name, tsetup in pbar:
        pbar.set_description(f'Tng: {name}')
        tsetup.train()


if __name__ == '__main__':
    main()

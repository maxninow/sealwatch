"""Script to prepare BOSSBase dataset.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import argparse
import conseal as cl
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import requests
import tempfile
from tqdm import tqdm
from typing import Union, Tuple
import zipfile

from . import _fabrika


def download_archive(
    input_url: str,
    output_dir: Path,
    skip_existing: bool = False,
) -> Path:
    """Downloads the ZIP archive with the BOSS dataset.

    :param input_url: URL to download from
    :param output_dir: directory to save the archive to
    :param skip_existing: skip archive if it already exists
    :return: path to the zip archive
    """
    # create data dir
    zip_path = output_dir / 'BOSSbase_1.01.zip'

    # skip if zip exists
    if skip_existing and zip_path.exists():
        logging.info(f'skipping download of {input_url}, {zip_path} already exists')
        return zip_path

    # stream request
    response = requests.get(input_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    logging.info(f'downloading ZIP file of size {total_size} B')
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(zip_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    #
    if total_size != 0 and progress_bar.n != total_size:
        logging.error('could not download file')
        raise RuntimeError("Could not download file")
    #
    logging.info(f'{input_url} file of size {total_size} B saved to {zip_path}')
    return zip_path


def prepare_covers(
    input_archive: Union[str, Path],
    output_dir: Path,
    skip_existing: bool = False,
):
    """Extracts the covers from the ZIP archive.

    :param input_archive: path to ZIP archive
    :param output_dir: directory to save the dataset to
    :param skip_existing: whether to skip already existing images
    """
    # list files
    zip = zipfile.ZipFile(input_archive)
    input_list = zip.namelist()
    input_list = [f for f in input_list if f != 'BOSSbase_1.01/']

    # create data dir
    dir_images = Path('images')
    path_images = output_dir / dir_images
    path_images.mkdir(parents=False, exist_ok=True)

    #
    df = []
    for zip_fname in tqdm(input_list):
        srcname = Path(zip_fname).name
        dstname = f'{Path(zip_fname).stem}.png'
        dst_fname = path_images / dstname

        if skip_existing and dst_fname.exists():
            logging.info(f'skipping {srcname}, exists')
            continue
        else:
            # extract image content
            with zip.open(zip_fname) as fi:
                content = fi.read()

            # save content
            with tempfile.NamedTemporaryFile(suffix='.pgm') as tmp:
                tmp.write(content)
                x = np.array(Image.open(tmp.name))
            Image.fromarray(x).save(dst_fname)

        # load image
        df.append({
            'name': dir_images / dstname,
            'height': x.shape[0],
            'width': x.shape[1],
        })

    #
    df = pd.DataFrame(df)
    if not df.empty:
        df = df.sort_values('name')
        with open(path_images / 'files.csv', 'w') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)


# map argument strings to conseal interface
SIMULATE = {
    'LSBR': (
        lambda x, alpha, coding, seed:
            cl.lsb.simulate(
                x,
                alpha=alpha,
                n=x.size,
                modify=cl.LSB_REPLACEMENT,
                e=CODING[coding](alpha),
                seed=seed,
            )
    ),
    'LSBM': (
        lambda x, alpha, coding, seed:
            cl.lsb.simulate(
                x,
                alpha=alpha,
                n=x.size,
                modify=cl.LSB_MATCHING,
                e=CODING[coding](alpha),
                seed=seed,
            )
    ),
}
COST = {
    'HILL': cl.hill.compute_cost_adjusted,
}
CODING = {
    'optimal': lambda alpha: None,
    'hamming': cl.coding.hamming.efficiency,
    'lsb': lambda alpha: 2,
}


@_fabrika.precovers(iterator='joblib', convert_to='pandas', n_jobs=8)
def _embed_spatial(
    name: str,
    cover_dir: Path,
    stego_dir: Path,
    stego_method: str,
    alpha: float,
    coding: str,
    **kw,
) -> pd.DataFrame:
    """Simulates spatial steganography into the cover.

    The decorator automatizes the dataset creation.
    This function is called for each cover in the directory.

    :name name: image name
    :param cover_dir: directory of covers
    :param stego_dir: directory of stegos
    :param stego_method: steganographic method
    :param alpha: embedding rate
    :param coding: method of steganographic coding, one of optimal, hamming, lsb
    :return: dataframe with results
    """

    cover_basename = os.path.basename(name)
    cover_filepath = cover_dir / cover_basename
    stego_filepath = stego_dir / cover_basename

    # Ensure that cover image exists
    assert os.path.exists(cover_filepath), f"Cover image \"{cover_filepath}\" does not exist."

    # Load cover
    x0 = np.array(Image.open(cover_filepath))

    # Get seed
    seed = cl.tools.password_to_seed(cover_basename) % (2**31-1)

    # Simulate with distortion
    if stego_method in COST:
        (rho_p1, rho_m1) = COST[stego_method](x0)
        (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
            rhos=(rho_p1, rho_m1),
            alpha=alpha,
            e=CODING[coding](alpha),
            n=x0.size,
        )
        if coding == 'optimal':
            alpha_hat = cl.tools.entropy(p_p1, p_m1) / x0.size
        else:
            alpha_hat = np.mean(p_p1 + p_m1) * CODING[coding](alpha)
        cost_hat = np.sum(p_p1 * rho_p1 + p_m1 * rho_m1) / x0.size
        delta = cl.simulate._ternary.simulate(
            ps=(p_p1, p_m1),
            seed=seed,
        ).astype('uint8')

        x1 = x0 + delta

    # Simulate with simulate
    elif stego_method in SIMULATE:
        x1 = SIMULATE[stego_method](x0, alpha, coding=coding, seed=seed)
        cost_hat, alpha_hat = None, None

    #
    else:
        raise NotImplementedError(f'undefined method {stego_method}')

    # Save stego
    Image.fromarray(x1).save(str(stego_filepath))

    # Keep records
    return {
        'name': os.path.relpath(stego_filepath, stego_dir.parent),
        **kw,
        'stego_method': stego_method,
        'alpha': alpha,
        'coding': coding,
        'alpha_hat': alpha_hat,
        'beta_hat': (x1 != x0).mean(),
        'cost_hat': cost_hat,
        'seed': seed,
    }


def prepare_stego_spatial(
    cover_dir: Path,
    output_dir: Path,
    stego_method: str,
    alpha: float,
    coding: str,
    skip_existing: bool = False,
):
    """Simulates spatial steganography into a set of covers.

    :param cover_dir: directory where to find the cover images
    :param output_dir: base directory in which to create the stego directory
    :param stego_method: steganographic method
    :param alpha: embedding rate
    :param coding: coding to use
    :param skip_existing: whether to skip images that already exist
    """
    # Get compressed cover images
    cover_df = pd.read_csv(cover_dir / 'files.csv')

    # Create stego directory
    cover_dir_basename = os.path.basename(cover_dir)
    stego_dirname = Path(f'stego-{stego_method}_{alpha}_{coding}-{cover_dir_basename}')
    stego_dir = output_dir / stego_dirname

    # Stego directory should not exist, unless `skip_num_images` or `take_num_images` are set
    stego_dir.mkdir(parents=False, exist_ok=False, mode=0o775)

    # Iterate over cover images
    logging.info(f'generating {stego_dir}')
    logging.info(f'using adaptive seed')

    df = _embed_spatial(
        cover_df,
        cover_dir=cover_dir,
        stego_dir=stego_dir,
        stego_method=stego_method,
        alpha=alpha,
        coding=coding,
        progress_on=True,
    )
    df = df.sort_values('name')
    # remove existing
    with open(stego_dir / 'files.csv', 'a') as f:
        df.to_csv(f, header=f.tell() == 0, index=False)


def shuffle_split_dataset(
    dataset_path: Union[Path, str],
    splitpoints: Tuple[float],
    shuffle_seed: int = None
) -> Tuple[pd.DataFrame]:
    """Shuffles and splits fabrika dataset into disjoint sets.

    :param dataset_path: dataset path
    :param splitpoints:
    :param shuffle_seed: seed for image shuffling
    :return: set of dataframes, per split
    """
    # process parameters
    dataset_path = Path(dataset_path)
    K = len(splitpoints)

    # get number of images
    dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if any(d.name.startswith('images') for d in dirs):
        precovers = [d for d in dirs if d.name.startswith('images')]
        dir_config = pd.read_csv(precovers[0] / 'files.csv')
    else:
        raise Exception('cover directory not found!')

    # permute
    rng = np.random.default_rng(shuffle_seed)
    names = [Path(d).stem for d in dir_config.name]
    names = np.unique(names)
    names = rng.permutation(names)

    # split function
    N = len(dir_config)
    assert np.sum(splitpoints) == 1, 'splits do not sum up to 1'
    splitpoints = np.hstack([[0], np.cumsum(np.array(splitpoints)*N)]).astype('int')

    # for each dataset
    splits = [pd.DataFrame() for _ in range(K)]
    for dir in dataset_path.iterdir():
        if dir.is_dir():

            # load dataset files
            try:
                dir_config = pd.read_csv(
                    dir / 'files.csv',
                    dtype={'height': int, 'width': int, 'seed': int},
                )
                logging.info(f'using directory {dir} with {len(dir_config)} samples for splitting')
            except FileNotFoundError:
                logging.warning(f'ignoring {dir}, files.csv not present')
                continue
            except ValueError:
                raise

            # reorder by permutation
            dir_config['basename'] = dir_config.name.apply(lambda d: Path(d).stem)
            dir_config = dir_config.set_index('basename')
            try:
                dir_config = dir_config.reindex(index=names)
            except ValueError:
                raise Exception(f'multiple files of the same name in {dir}')

            #
            for i in range(K):
                # get indices
                idx = names[splitpoints[i]:splitpoints[i+1]]
                dir_config_segment = dir_config.loc[idx]
                # drop missing stego files (e.g., when embedding fails)
                dir_config_segment = dir_config_segment.dropna(subset=['name'])
                # get images from this segment
                splits[i] = pd.concat([splits[i], dir_config_segment], ignore_index=True)
    #
    return splits

def prepare_boss(
    input_url: str = 'https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip',
    output_dir: Union[str, Path] = 'data/boss',
    stego_method: str = 'LSBM',
    alpha: float = 0.4,
    coding: str = 'optimal',
    skip_existing: bool = False,
    split: [float] = [0.5, 0.5],
    names: [str] = None,
    shuffle_seed: int = 12345,
):
    """Main function to prepare the BOSSBase dataset.

    :param input_url: URL to download the dataset from
    :param output_dir: Directory to save the dataset
    :param stego_method: Steganographic method to use
    :param alpha: Embedding rate
    :param coding: Steganographic coding method
    :param skip_existing: Whether to skip existing files
    :param split: List of split ratios (e.g., [0.5, 0.5] for train/test)
    :param names: Names of the splits (e.g., ['tr', 'te'])
    :param shuffle_seed: Seed for shuffling the dataset
    """
    # Convert output_dir to Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer split names if not provided
    if names is None:
        if len(split) == 2:
            names = ['tr', 'te']
        elif len(split) == 3:
            names = ['tr', 'va', 'te']
        else:
            raise RuntimeError(
                f'Cannot infer names from split {split}. Provide names explicitly.'
            )

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "fabrika.log"),  # log to log file
            logging.StreamHandler(),  # print to stderr
        ],
    )

    # Download BOSS dataset
    zip_path = download_archive(
        input_url=input_url,
        output_dir=output_dir,
        skip_existing=skip_existing,
    )

    # Prepare covers
    prepare_covers(
        input_archive=zip_path,
        output_dir=output_dir,
        skip_existing=skip_existing,
    )

    # Prepare stegos
    prepare_stego_spatial(
        cover_dir=output_dir / 'images',
        output_dir=output_dir,
        stego_method=stego_method,
        alpha=alpha,
        coding=coding,
    )

    # Shuffle and split dataset
    splits = shuffle_split_dataset(
        dataset_path=output_dir,
        splitpoints=split,
        shuffle_seed=shuffle_seed,
    )

    # Save split CSVs
    for split_df, name in zip(splits, names):
        split_df.to_csv(
            output_dir / f'split_{name}.csv',
            index=False,
        )
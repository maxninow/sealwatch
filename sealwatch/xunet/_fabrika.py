"""Internal module for simple processing of datasets.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import glob
import hashlib
import h5py
import joblib
import logging
import numpy as np
import os
import pandas as pd
import pathlib
from PIL import Image
from tqdm import tqdm
import typing


class ProgressParallel(joblib.Parallel):
    """"""
    def __call__(self, *args, total: int = None, disable: bool = False, **kw):
        """"""
        with tqdm(total=total, disable=disable) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kw)

    def print_progress(self):
        """"""
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def collect_files(
    patterns: typing.Sequence[str],
    fn: typing.Callable,
    pre_fn: typing.Callable = None,
    post_fn: typing.Callable = None,
    iterator: str = 'python',
    ignore_missing: bool = False,
    file_search: bool = False,
    convert_to: bool = 'pandas',
    **kw_deco,
) -> typing.Callable:
    """

    :param iterator:
    :param ignore_missing:
    :param convert_to:
    """

    def iterate(
        dataset: pathlib.Path,
        skip_num_images: int = None,
        take_num_images: int = None,
        shuffle_seed: int = None,
        progress_on: bool = False,
        split: str = None,
        **kw_fn,
    ) -> pd.DataFrame:
        """

        :param dataset:
        :param skip_num_images:
        :param take_num_images:
        :param shuffle_seed:
        :param progress_on:
        :param split:
        """
        # get precovers
        if isinstance(dataset, pd.DataFrame):
            df = dataset
            dataset = pathlib.Path('.')
        elif split is not None:
            dataset = pathlib.Path(dataset)
            df = pd.read_csv(
                dataset / split,
                low_memory=False,
                dtype={
                    'name': str,
                    'height': int,
                    'width': int,
                    'demosaic': str,
                    'color': str,
                    'stego_method': str,
                    'simulator': str,
                    'alpha': float,
                    'color_strategy': str,
                    'channels': str,
                    'beta_hat': float,
                    'device': str,
                })
            # print(f'- split {split} with {len(df)} rows')
        else:
            dataset = pathlib.Path(dataset)
            paths = []
            for pattern in patterns:
                paths += glob.glob(str(dataset / pattern))
            if not file_search:
                df = []
                for path in paths:
                    try:
                        df.append(pd.read_csv(pathlib.Path(path) / 'files.csv'))
                    except Exception:
                        if not ignore_missing:
                            raise
                df = pd.concat(df)
            else:
                df = pd.DataFrame({'name': [
                    p.relative_to(p.parents[1])
                    for p in map(pathlib.Path, paths)
                ]})

        # preprocess
        if pre_fn is not None:
            df = pre_fn(df, **kw_fn)
            if df.empty:
                raise Exception('pre_fn() returned empty dataframe')

        # take first
        df = df.sort_values('name').reset_index(drop=True)
        if shuffle_seed:
            df = df.sample(frac=1., random_state=shuffle_seed)
        if skip_num_images:
            df = df[skip_num_images:]
        if take_num_images:
            df = df[:take_num_images]

        # vanilla Python
        if iterator == 'python':
            res = []
            keys = df.columns.difference(pd.Index(['name']))
            for index, row in tqdm(df.iterrows(), total=len(df), disable=not progress_on):
                row = row.to_dict()
                fname = row.pop('name')
                res.append(fn(
                    dataset / fname,
                    **(row | kw_fn)
                ))

        # joblib
        elif iterator == 'joblib':
            gen = (
                joblib.delayed(fn)(
                    dataset / row['name'],
                    **(row.drop('name').to_dict() | kw_fn),
                )
                for index, row in df.iterrows()
            )
            res = ProgressParallel(**kw_deco)(gen, total=len(df), disable=not progress_on)

        # provide entire dataframe
        elif iterator is None:
            if progress_on:
                tqdm.tqdm.pandas()
                df['name'] = df['name'].progress_apply(lambda d: str(dataset / d))
            else:
                df['name'] = df['name'].apply(lambda d: str(dataset / d))
            res = fn(df, **kw_fn)

        else:
            raise NotImplementedError(f'unknown iterator {iterator}')

        # convert
        if convert_to is None:
            pass
        elif convert_to == 'pandas':
            res = pd.DataFrame(res)
        elif convert_to == 'numpy':
            res = np.array(res)
        else:
            raise NotImplementedError(f'unknown convertor {convert_to}')

        # postprocess
        if post_fn is not None:
            res = post_fn(res, **kw_fn)

        return res

    return iterate


def filter_precovers(
    df,
    height: int = None,
    demosaic: str = None,
    grayscale: str = None,
    quality: int = None,
    *args,
    **kw,
) -> pd.DataFrame:
    """

    :param df:
    :param height:
    :param demosaic:
    :param grayscale:
    :param quality:
    :return:
    """
    if demosaic is not None:
        df = df[df['demosaic'] == demosaic]
    if grayscale is not None:
        color = 'grayscale' if grayscale else 'sRGB'
        df = df[df['color'] == color]
    if height is not None:
        df = df[df['height'] == height]
    if 'stego_method' in df:
        df = df[df['stego_method'].isna()]
    if 'quality' in df:
        if quality is None:
            df = df[df['quality'].isna()]
        else:
            df = df[df['quality'] == quality]
    print(df)
    return df


def precovers(**kw_deco):
    """"""
    def _precovers(fn: typing.Callable):
        """"""
        return collect_files(['images*'], fn=fn, pre_fn=filter_precovers, **kw_deco)
    return _precovers


def raws(patterns: typing.List[str] = ['data/*.*'], **kw_deco):
    """"""
    def _raws(fn: typing.Callable):
        """"""
        def pre_fn(
            df: pd.DataFrame,
            **kw,
        ) -> pd.DataFrame:
            """"""
            return df

        return collect_files(patterns, fn=fn, pre_fn=pre_fn, file_search=True, **kw_deco)
    return _raws


def covers(**kw_deco):
    """"""
    def _covers(fn: typing.Callable):
        """"""
        def pre_fn(
            df: pd.DataFrame,
            quality: int = None,
            samp_factor: str = None,
            **kw,
        ) -> pd.DataFrame:
            """"""
            df = df[~df['quality'].isna()]
            if 'mse' in df:
                df = df[df['mse'].isna()]
            if 'stego_method' in df:
                df = df[df['stego_method'].isna()]
            if quality is not None:
                df = df[df['quality'] == quality]
            if samp_factor is not None:
                df = df[df['samp_factor'] == samp_factor]
            return df

        return collect_files(['jpegs*'], fn=fn, pre_fn=pre_fn, **kw_deco)
    return _covers


def filter_stego_spatial(
    df: pd.DataFrame,
    stego_method: int = None,
    alpha: str = None,
    coding: str = None,
    demosaic: str = None,
    quality: int = None,
    **kw,
) -> pd.DataFrame:
    """"""
    if demosaic is not None:
        df = df[df['demosaic'] == demosaic]
    if stego_method is not None:
        df = df[df['stego_method'] == stego_method]
    if alpha is not None:
        df = df[df['alpha'] == alpha]
    if coding is not None:
        df = df[df['coding'] == coding]
    if 'quality' in df:
        if quality is None:
            df = df[df['quality'].isna()]
        else:
            df = df[df['quality'] == quality]
    return df


def stego_spatial(**kw_deco):
    """"""
    def _stego_spatial(fn: typing.Callable):
        """"""
        return collect_files(['stego*'], fn=fn, pre_fn=filter_stego_spatial, **kw_deco)
    return _stego_spatial


def filter_stego_jpeg(
    df: pd.DataFrame,
    stego_method: int = None,
    alpha: str = None,
    coding: str = None,
    demosaic: str = None,
    quality: int = None,
    **kw,
) -> pd.DataFrame:
    """"""
    df = df[~df['quality'].isna()]
    df = df[df['mse'].isna()]
    if 'stego_method' in df.columns:
        df = df[~df['stego_method'].isna()]
    if demosaic is not None and 'demosaic' in df.columns:
        df = df[df['demosaic'] == demosaic]
    if stego_method is not None and 'stego_method' in df.columns:
        df = df[df['stego_method'] == stego_method]
    if alpha is not None and 'alpha' in df.columns:
        df = df[df['alpha'] == alpha]
    if coding is not None and 'coding' in df.columns:
        df = df[df['coding'] == coding]
    if quality is not None and 'quality' in df.columns:
        df = df[df['quality'] == quality]
    return df


def stego_jpeg(**kw_deco):
    """"""
    def _stego_jpeg(fn: typing.Callable):
        """"""
        return collect_files(['stego*'], fn=fn, pre_fn=filter_stego_jpeg, **kw_deco)
    return _stego_jpeg


def cover_stego_spatial(paired=True, **kw_deco):
    """"""
    def _cover_stego_spatial(fn: typing.Callable):
        """"""
        def pre_fn(
            df: pd.DataFrame,
            stego_method: int = None,
            alpha: str = None,
            coding: str = None,
            demosaic: str = None,
            **kw,
        ) -> pd.DataFrame:
            """"""
            # filter cover types
            if demosaic is not None:
                df = df[df['demosaic'] == demosaic]
            if 'quality' in df:
                df = df[df['quality'].isna()]

            # split cover and stegos
            df_c = df[df['stego_method'].isna()]
            df_s = df[~df['stego_method'].isna()]

            # filter stego
            if stego_method is not None:
                df_s = df_s[df_s['stego_method'] == stego_method]
            if alpha is not None:
                df_s = df_s[df_s['alpha'] == alpha]
            if coding is not None:
                df_s = df_s[df_s['coding'] == coding]

            # paired
            if paired:
                df_c['stem'] = df_c['name'].apply(lambda f: pathlib.Path(f).stem)
                df_s['stem'] = df_s['name'].apply(lambda f: pathlib.Path(f).stem)
                df = df_c.merge(df_s, how='left', on=['stem'], suffixes=('_c', '_s'))
                df = df.drop('stem', axis=1)
                df['name'] = df['name_c']

            else:
                raise NotImplementedError

            return df

        def post_fn(
            df: pd.DataFrame,
            *args,
            **kw,
        ) -> pd.DataFrame:
            """"""
            df['stem'] = df['name_c'].apply(lambda f: pathlib.Path(f).stem)
            df = df.sort_values(['stem', 'name_c'])
            df = df.drop('stem', axis=1)
            return df

        return collect_files(
            ['images*', 'stego*'],
            fn=fn,
            pre_fn=pre_fn,
            post_fn=post_fn,
            **kw_deco,
        )
    return _cover_stego_spatial


def subset_names(path, K, seed):
    """"""
    # get
    split = pd.read_csv(path)
    split['stem'] = split['name'].apply(lambda f: pathlib.Path(f).stem)
    names = split['stem'].unique()
    # shuffle and select
    rng = np.random.default_rng(seed)
    names = rng.permutation(names)
    names = names[:K]
    #
    return list(set(names))


def load_features(path, names):
    """"""
    with h5py.File(path, "r") as f:
        # extract
        filenames = np.array([pathlib.Path(f.decode('utf-8')).stem for f in f['filenames']])
        features = np.array(f['features'])
        # sort by filenames
        order = np.argsort(filenames)
        features = features[order]
        filenames = filenames[order]
        # select samples
        index = [i for i, f in enumerate(filenames) if f in names]
        logging.info(f'loaded {len(index)} features from {pathlib.Path(path).name}')
        features = features[index]
        # reorder
        return features

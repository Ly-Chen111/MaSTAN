from .ovisoe import build as build_ovisoe


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == 'ovisoe':
        return build_ovisoe(image_set, args)
    raise ValueError(f'dataset {dataset_file} not supported')

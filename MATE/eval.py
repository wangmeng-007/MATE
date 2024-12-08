import os
import argparse
import logging
from lib import evaluation
import arguments
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='f30k',
                        help='coco or f30k')
    parser.add_argument('--data_path', default='/home/s1/Documents/ESA-main1/data/coco')
    parser.add_argument('--transformer_learning_rate', default=0.0001, type=float,
                        help='Initial learning rate for the transformer.')
    parser.add_argument('--num_layers', default=6, type=int,
                        help='Number of layers in the transformer encoder.')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--evaluate_cxc', action='store_true')
    parser.add_argument('--model_name', default='/home/s1/Documents/ESA-main1/ESA_BERT/runs/runX/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--nhead', default=16, type=int,
                        help='Number of attention heads in the transformer encoder.')

    opt = parser.parse_args()

    if opt.dataset == 'coco':
        weights_bases = [
            # 'runs/release_weights/coco_butd_region_bert',
            # 'runs/release_weights/coco_butd_grid_bert',
            # 'runs/release_weights/coco_wsl_grid_bert',
            opt.model_name    
        ]
    elif opt.dataset == 'f30k':
        weights_bases = [
            # 'runs/release_weights/f30k_butd_region_bert',
            # 'runs/release_weights/f30k_butd_grid_bert',
            # 'runs/release_weights/f30k_wsl_grid_bert',
            opt.model_name 
        ]
    else:
        raise ValueError('Invalid dataset argument {}'.format(opt.dataset))

    for base in weights_bases:
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        if opt.save_results:  # Save the final results for computing ensemble results
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'coco':
            if not opt.evaluate_cxc:
                # Evaluate COCO 5-fold 1K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
                # Evaluate COCO 5K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
            else:
                # Evaluate COCO-trained models on CxC
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True, cxc=True)
        elif opt.dataset == 'f30k':
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)


if __name__ == '__main__':
    main()

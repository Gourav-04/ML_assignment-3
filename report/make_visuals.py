"""Run from the experiment directory: python PATH/TO/make_visuals.py
The PDF already includes these panels. Optional flags point to a different run."""
import argparse, csv, sys, tarfile, pickle, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def draw(results, archive, project):
    import torch
    sys.path.insert(0,str(project.resolve()))
    from common.models import Autoencoder, classifier
    dest=Path(__file__).resolve().parent/'figures'
    dest.mkdir(exist_ok=True)
    labels="airplane automobile bird cat deer dog frog horse ship truck".split()
    with tarfile.open(archive) as tar:
        batch=pickle.load(tar.extractfile("cifar-10-batches-py/test_batch"),encoding="bytes")
    # Use the first four test images in their official order. All choices are already fixed.
    x=torch.from_numpy(batch[b'data'][:4].reshape(-1,3,32,32)).float()/255
    ids=np.arange(1024)
    pixel=np.random.default_rng(26018).permutation(1024)
    blocks=ids.reshape(8,4,8,4).transpose(0,2,1,3).reshape(64,4,4)
    tiles=blocks[np.random.default_rng(26018).permutation(64)].reshape(8,8,4,4).transpose(0,2,1,3).ravel()
    def show(ax,img):
        ax.imshow(img.permute(1,2,0).numpy(),interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    fig,axes=plt.subplots(2,3,figsize=(4.1,2.8),layout='constrained')
    for i in range(2):
        for j,(name,perm) in enumerate([('Original',ids),('Pixel shuffle',pixel),('Tile shuffle',tiles)]):
            show(axes[i,j],x[i].reshape(3,-1)[:,perm].reshape(3,32,32))
            if i==0:axes[i,j].set_title(name,fontsize=9)
        axes[i,0].set_ylabel(labels[batch[b'labels'][i]],fontsize=8)
    fig.savefig(dest/'interventions.pdf',bbox_inches='tight');plt.close(fig)
    def load(model,name):
        model.load_state_dict(torch.load(results/(name+'.pt'),weights_only=True,map_location='cpu'));return model.eval()
    width=json.loads((results/'selected.json').read_text())['bottleneck']
    cfg=json.loads((project/'config.json').read_text())
    cnn=load(classifier("cnn"),"cnn_clean")
    ae=load(Autoencoder(width),f'ae_{width}');dae=load(Autoencoder(width),f'dae_{width}')
    noise=np.random.default_rng(26018).standard_normal((256,3,32,32),dtype=np.float32)[:4]
    noisy=(x+cfg['noise_sd']*torch.from_numpy(noise)).clamp(0,1)
    fig,axs=plt.subplots(4,4,figsize=(5.1,5.0),layout='constrained')
    with torch.inference_mode():
        for col,(name,view) in enumerate([('Clean',x),('Noisy',noisy),('Clean AE',ae(noisy)),('Denoiser',dae(noisy))]):
            predictions=cnn(view).argmax(1)
            for i in range(4):
                ax=axs[i,col];show(ax,view[i]);ax.set_xlabel(labels[int(predictions[i])],fontsize=8,labelpad=1)
                if col==0:ax.set_ylabel(f"{i}: "+labels[batch[b'labels'][i]],fontsize=8)
                if i==0:ax.set_title(name,fontsize=9)
    fig.savefig(dest/'denoising_examples.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(4.2,2.45),layout='constrained')
    for k in cfg['bottlenecks']:
        with (results/f'ae_{k}.csv').open() as f: records=list(csv.DictReader(f))
        ax.plot([int(a['epoch']) for a in records],[float(a['val_loss']) for a in records],marker='o',ms=3,label=f'Width {k}')
    ax.set(xlabel='Epoch',ylabel='Validation MSE',xticks=range(1,cfg['ae_epochs']+1))
    ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.savefig(dest/'ae_learning.pdf');plt.close(fig)
    cm=np.loadtxt(results/'svm_confusion.csv',delimiter=',')
    fig,ax=plt.subplots(figsize=(4.3,3.5),layout='constrained');im=ax.imshow(cm/10,cmap='Blues')
    ax.set(xticks=range(10),yticks=range(10),xticklabels=labels,yticklabels=labels,xlabel='Predicted class',ylabel='True class')
    ax.tick_params(axis='both',labelsize=7);plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
    fig.colorbar(im,ax=ax,label='Share of true class (%)',shrink=.8)
    fig.savefig(dest/'svm_confusion.pdf');plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n,default in [('results','results'),('archive','data/cifar-10-python.tar.gz'),('project','.')]:
        parser.add_argument('--'+n,type=Path,default=Path(default))
    draw(**vars(parser.parse_args()))

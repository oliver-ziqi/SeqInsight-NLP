"""Check actual notebook model definitions without downloading data or retraining."""
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_definitions():
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'seqinsight.ipynb').read_text(encoding='utf-8'))
    scope={'torch':torch,'nn':nn,'F':F,'device':torch.device('cpu')}
    classes=('myRNNBlock','myLSTMBlock','Classifier','textGenModel')
    for cell in notebook['cells']:
        source=''.join(cell.get('source',[]))
        if cell['cell_type']=='code' and (any('class '+name+'(' in source for name in classes) or source.lstrip().startswith('def generate_text(')):
            exec(compile(source,'seqinsight.ipynb','exec'),scope)
    return scope

def main():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    scope=load_definitions()
    for backend in ('rnn','lstm'):
        scope['block_type']=backend
        tokens=torch.tensor([[2,3,4,0],[5,6,0,0]])
        model=scope['Classifier'](32,8,12,2,2,0.1)
        logits=model(tokens)
        assert logits.shape==(2,2)
        loss=nn.CrossEntropyLoss()(logits,torch.tensor([0,1]))
        loss.backward()
        assert torch.isfinite(loss)
        generator=scope['textGenModel'](32,8,12,2)
        out,_=generator(tokens,generator.init_hidden(2))
        assert out.shape==(2,4,32)
        generation_loss=nn.CrossEntropyLoss()(out.reshape(-1,32),tokens.reshape(-1))
        generation_loss.backward()
        assert torch.isfinite(generation_loss)
        print(backend+': classifier and generator forward/backward passed')
        for kind in ('classifier','gentext'):
            weights=torch.load(str(root/'models'/(backend+'_'+kind+'.pth')),map_location='cpu',weights_only=True)
            vocabulary,embedding=weights['embedding.weight'].shape
            hidden_size=weights['model.weight_hh.0'].shape[1]
            layers=sum(key.startswith('model.weight_ih.') for key in weights)
            if kind=='classifier':
                trained=scope['Classifier'](vocabulary,embedding,hidden_size,layers,2,0.2)
                trained.load_state_dict(weights)
                trained.eval()
                with torch.no_grad(): assert trained(tokens).shape==(2,2)
            else:
                trained=scope['textGenModel'](vocabulary,embedding,hidden_size,layers)
                trained.load_state_dict(weights)
                trained.eval()
                with torch.no_grad():
                    prediction,_=trained(tokens,trained.init_hidden(2))
                    assert prediction.shape==(2,4,vocabulary)
            print(backend+' '+kind+': retained checkpoint loaded and forward passed')
    import pandas as pd
    corpus=pd.read_parquet(root/'corpuses/short.parquet')
    assert 'short_story' in corpus.columns
    tokenizer=json.loads((root/'tokenizer/word_level.json').read_text(encoding='utf-8'))
    assert tokenizer['model']['type']=='WordLevel'
    assert len(tokenizer['model']['vocab'])==3072
    assert len(json.loads((root/'tokenizer/sst2_vocab.json').read_text(encoding='utf-8')))==14764
    print('Parquet corpus and tokenizer passed; no historical metrics were recomputed')

if __name__=='__main__':
    main()

"""Run retained notebook checkpoints without running training cells."""
from pathlib import Path
import argparse
import json
import torch
from smoke_check import load_definitions

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=['sentiment','generate'],default='sentiment')
    parser.add_argument('--backend',choices=['rnn','lstm'],default='rnn')
    parser.add_argument('--text',default='This movie was wonderful.')
    parser.add_argument('--length',type=int,default=40)
    parser.add_argument('--temperature',type=float,default=0.8)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    root=Path(__file__).resolve().parents[1]
    scope=load_definitions()
    scope['block_type']=args.backend
    kind='classifier' if args.task=='sentiment' else 'gentext'
    weights=torch.load(str(root/'models'/(args.backend+'_'+kind+'.pth')),map_location='cpu',weights_only=True)
    vocab_size,embedding=weights['embedding.weight'].shape
    hidden=weights['model.weight_hh.0'].shape[1]
    layers=sum(key.startswith('model.weight_ih.') for key in weights)
    if args.task=='sentiment':
        from torchtext.data.utils import get_tokenizer
        vocabulary=json.loads((root/'tokenizer/sst2_vocab.json').read_text(encoding='utf-8'))
        words=['<BOS>']+get_tokenizer('basic_english')(args.text)+['<EOS>']
        ids=torch.tensor([[vocabulary.get(word,vocabulary['<UNK>']) for word in words]])
        model=scope['Classifier'](vocab_size,embedding,hidden,layers,2,0.2)
        model.load_state_dict(weights)
        model.eval()
        with torch.no_grad(): probability=model(ids).softmax(dim=1)[0]
        result={'label':'positive' if probability.argmax().item()==1 else 'negative','probability_positive':probability[1].item()}
        print(json.dumps(result))
    else:
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(root/'tokenizer/word_level.json'))
        if not tokenizer.encode(args.text).ids: parser.error('Generation prompt must encode at least one token.')
        model=scope['textGenModel'](vocab_size,embedding,hidden,layers)
        model.load_state_dict(weights)
        print(scope['generate_text'](model,tokenizer,args.text,length=args.length,temperature=args.temperature))

if __name__=='__main__':
    main()
